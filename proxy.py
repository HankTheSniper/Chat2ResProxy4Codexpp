"""
Codex Responses <-> Chat Completions 代理

对 Codex 暴露标准的 /v1/responses（含 SSE 流），内部转发到上游的
/v1/chat/completions，并把上游的 chat 流「正确地」重组成合规的 responses 事件。

修复的三类问题：
  1) reasoning 泄漏进正文  -> reasoning 单独走 reasoning_summary 事件，绝不并入 output_text
  2) 首字母重复(NN...)      -> delta 只累加一次，不重复 seed
  3) tool call 提前中断     -> 只有上游真正 finish 后才发 response.completed；
                              function_call 作为独立 output item 完整发出
"""
import os
import json
import time
import uuid
import asyncio
from typing import Any, AsyncGenerator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# 上游临时过载(503/429/5xx)时自动退避重试的次数
MAX_RETRIES = 5

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY", "")
# 模型改写：Codex 端用它认识的名字(gpt-5-codex)才挂工具，
# 但中转不认这个名，故在此改写成中转真实模型名。
UPSTREAM_MODEL = os.environ.get("UPSTREAM_MODEL", "").strip()

# 上游对话协议：
#   "responses"(默认) —— 上游原生支持 /v1/responses，直接透传（保留工具自动调用）。
#   "chat"            —— 上游只会 /v1/chat/completions，走内部 responses<->chat 转换。
# 关键：许多“中转”其实是 responses 原生模型。把它强转成 chat/completions 会让
# 模型在 tool_choice=auto 下丢失自动调用工具的能力（只吐正文，不发 function_call），
# 这正是「gpt-5.6/5.5 不调用工具」的根因。故默认透传原生 responses。
UPSTREAM_WIRE_API = os.environ.get("UPSTREAM_WIRE_API", "responses").strip().lower()

app = FastAPI()


@app.on_event("startup")
async def _silence_conn_reset_noise():
    """Windows 上 Codex 频繁开关连接会触发 ConnectionResetError(WinError 10054)，
    这是无害的套接字清理噪音（请求已成功返回），静音它以免淹没真正的错误日志。"""
    loop = asyncio.get_running_loop()
    prev = loop.get_exception_handler()

    def handler(lp, context):
        exc = context.get("exception")
        if isinstance(exc, ConnectionResetError):
            return
        if prev:
            prev(lp, context)
        else:
            lp.default_exception_handler(context)

    loop.set_exception_handler(handler)


def _now() -> int:
    return int(time.time())


def sse(event: str, data: dict) -> str:
    """格式化一条 SSE 事件。responses 流使用带 event: 名的 SSE。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def responses_to_chat(payload: dict) -> dict:
    """把 Codex 发来的 /v1/responses 请求体翻译成 /v1/chat/completions。"""
    messages: list[dict] = []

    # responses 用 instructions 承载 system 提示
    if payload.get("instructions"):
        messages.append({"role": "system", "content": payload["instructions"]})

    # input 可能是字符串，也可能是 responses 的 item 数组
    inp = payload.get("input", [])
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    else:
        for item in inp:
            itype = item.get("type", "message")
            if itype == "message":
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, list):
                    text = "".join(
                        c.get("text", "")
                        for c in content
                        if c.get("type") in ("input_text", "output_text", "text")
                    )
                else:
                    text = content
                messages.append({"role": role, "content": text})
            elif itype == "function_call":
                # 历史里的模型工具调用
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": item.get("call_id", ""),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "") or "{}",
                        },
                    }],
                })
            elif itype == "function_call_output":
                # 历史里的工具执行结果
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": item.get("output", ""),
                })

    chat: dict[str, Any] = {
        "model": UPSTREAM_MODEL or payload.get("model"),
        "messages": messages,
        "stream": bool(payload.get("stream", False)),
    }

    # 工具定义：responses 的 tools 是扁平结构，chat 需要 {"type","function":{...}}
    tools = payload.get("tools")
    if tools:
        chat_tools = []
        for t in tools:
            if t.get("type") == "function":
                if "function" in t:
                    chat_tools.append(t)
                else:
                    chat_tools.append({
                        "type": "function",
                        "function": {
                            "name": t.get("name"),
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {}),
                        },
                    })
        if chat_tools:
            chat["tools"] = chat_tools
            # 关闭并行工具调用：避免模型用 multi_tool_use.parallel 构造，
            # 该构造在中转链路上会被当成文本泄漏（<tool_call> 文字），导致工具不执行。
            chat["parallel_tool_calls"] = False
            if payload.get("tool_choice"):
                chat["tool_choice"] = payload["tool_choice"]

    for k in ("temperature", "top_p", "max_output_tokens"):
        if k in payload:
            chat["max_tokens" if k == "max_output_tokens" else k] = payload[k]

    return chat

async def stream_responses(chat_body: dict, model: str) -> AsyncGenerator[str, None]:
    """
    调上游 chat/completions(stream)，把 chat 增量重组成合规的 responses SSE。
    状态机保证：正文、reasoning、tool call 各走各的 output item，
    且只有上游 finish 后才发 response.completed。
    """
    resp_id = "resp_" + uuid.uuid4().hex
    created = _now()
    seq = 0

    def nxt() -> int:
        nonlocal seq
        seq += 1
        return seq

    # 生命周期起始事件
    base_resp = {"id": resp_id, "object": "response", "created_at": created,
                 "model": model, "status": "in_progress", "output": []}
    yield sse("response.created", {"type": "response.created", "sequence_number": nxt(), "response": base_resp})
    yield sse("response.in_progress", {"type": "response.in_progress", "sequence_number": nxt(), "response": base_resp})

    out_index = 0
    text_open = False          # 正文 message item 是否已开
    text_item_id = ""
    text_out_index = 0         # 正文 item 所在的 output_index
    text_buf = ""              # 累积正文全文（填入 .done / completed 的终值）
    reasoning_open = False     # reasoning item 是否已开
    reasoning_item_id = ""
    reasoning_out_index = 0     # reasoning item 所在的 output_index
    reasoning_buf = ""         # 累积 reasoning 全文
    # tool call 聚合： index -> {id,name,args,out_index,item_id,opened}
    tool_calls: dict[int, dict] = {}
    finish_reason = None
    completed_items: list[dict] = []   # 已收尾的 output item，供 completed 汇总

    headers = {"Authorization": f"Bearer {UPSTREAM_API_KEY}", "Content-Type": "application/json"}
    url = f"{UPSTREAM_BASE_URL}/chat/completions"

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        # 连接上游，遇到临时过载(503 cpu overloaded / 429 / 5xx)自动退避重试
        attempt = 0
        while True:
            attempt += 1
            up_ctx = client.stream("POST", url, headers=headers, json=chat_body)
            up = await up_ctx.__aenter__()
            if up.status_code >= 400:
                body_bytes = await up.aread()
                err_text = body_bytes.decode("utf-8", "replace")
                await up_ctx.__aexit__(None, None, None)
                retryable = (up.status_code in (429, 500, 502, 503, 504)
                             or "overload" in err_text.lower())
                if retryable and attempt <= MAX_RETRIES:
                    delay = min(2 * attempt, 12)
                    print(f"[RETRY {attempt}/{MAX_RETRIES}] upstream {up.status_code}, "
                          f"{delay}s 后重试: {err_text[:120]}", flush=True)
                    await asyncio.sleep(delay)
                    continue
                print(f"[UPSTREAM ERROR] {up.status_code}: {err_text}", flush=True)
                fail_resp = {"id": resp_id, "object": "response", "created_at": created,
                             "model": model, "status": "failed",
                             "error": {"code": f"upstream_{up.status_code}", "message": err_text}}
                yield sse("response.failed", {"type": "response.failed",
                          "sequence_number": nxt(), "response": fail_resp})
                return
            break
        try:
            async for line in up.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                # ---- reasoning：单独通道，绝不并入正文（修复泄漏） ----
                rc = delta.get("reasoning_content") or delta.get("reasoning")
                if rc:
                    if not reasoning_open:
                        reasoning_item_id = "rs_" + uuid.uuid4().hex
                        reasoning_out_index = out_index
                        item = {"type": "reasoning", "id": reasoning_item_id, "summary": []}
                        yield sse("response.output_item.added", {"type": "response.output_item.added",
                                  "sequence_number": nxt(), "output_index": out_index, "item": item})
                        yield sse("response.reasoning_summary_part.added", {
                            "type": "response.reasoning_summary_part.added", "sequence_number": nxt(),
                            "item_id": reasoning_item_id, "output_index": out_index,
                            "summary_index": 0, "part": {"type": "summary_text", "text": ""}})
                        reasoning_open = True
                    reasoning_buf += rc
                    yield sse("response.reasoning_summary_text.delta", {
                        "type": "response.reasoning_summary_text.delta", "sequence_number": nxt(),
                        "item_id": reasoning_item_id, "output_index": reasoning_out_index,
                        "summary_index": 0, "delta": rc})

                # reasoning 结束、正文/工具开始前，先收掉 reasoning item
                content_piece = delta.get("content")
                tool_deltas = delta.get("tool_calls")
                if reasoning_open and (content_piece or tool_deltas or finish_reason):
                    yield sse("response.reasoning_summary_text.done", {
                        "type": "response.reasoning_summary_text.done", "sequence_number": nxt(),
                        "item_id": reasoning_item_id, "output_index": reasoning_out_index,
                        "summary_index": 0, "text": reasoning_buf})
                    yield sse("response.reasoning_summary_part.done", {
                        "type": "response.reasoning_summary_part.done", "sequence_number": nxt(),
                        "item_id": reasoning_item_id, "output_index": reasoning_out_index,
                        "summary_index": 0, "part": {"type": "summary_text", "text": reasoning_buf}})
                    done_reason = {"type": "reasoning", "id": reasoning_item_id,
                                   "summary": [{"type": "summary_text", "text": reasoning_buf}]}
                    yield sse("response.output_item.done", {"type": "response.output_item.done",
                              "sequence_number": nxt(), "output_index": reasoning_out_index,
                              "item": done_reason})
                    completed_items.append(done_reason)
                    reasoning_open = False
                    out_index += 1

                # ---- 正文 content：只累加一次（修复 NN 重复） ----
                if content_piece:
                    if not text_open:
                        text_item_id = "msg_" + uuid.uuid4().hex
                        text_out_index = out_index
                        text_buf = ""
                        item = {"type": "message", "id": text_item_id, "status": "in_progress",
                                "role": "assistant", "content": []}
                        yield sse("response.output_item.added", {"type": "response.output_item.added",
                                  "sequence_number": nxt(), "output_index": text_out_index, "item": item})
                        yield sse("response.content_part.added", {"type": "response.content_part.added",
                                  "sequence_number": nxt(), "item_id": text_item_id, "output_index": text_out_index,
                                  "content_index": 0, "part": {"type": "output_text", "text": ""}})
                        text_open = True
                    text_buf += content_piece
                    yield sse("response.output_text.delta", {"type": "response.output_text.delta",
                              "sequence_number": nxt(), "item_id": text_item_id, "output_index": text_out_index,
                              "content_index": 0, "delta": content_piece})

                # ---- tool_calls：聚合分片，作为独立 output item ----
                if tool_deltas:
                    for td in tool_deltas:
                        idx = td.get("index", 0)
                        tc = tool_calls.get(idx)
                        if tc is None:
                            tc = {"id": td.get("id") or ("call_" + uuid.uuid4().hex),
                                  "name": "", "args": "", "out_index": None,
                                  "item_id": "fc_" + uuid.uuid4().hex, "opened": False}
                            tool_calls[idx] = tc
                        fn = td.get("function", {})
                        if fn.get("name"):
                            tc["name"] = fn["name"]
                        if not tc["opened"] and tc["name"]:
                            # 若正文还开着，先收掉正文再开工具
                            if text_open:
                                yield sse("response.output_text.done", {"type": "response.output_text.done",
                                          "sequence_number": nxt(), "item_id": text_item_id,
                                          "output_index": text_out_index, "content_index": 0, "text": text_buf})
                                yield sse("response.content_part.done", {"type": "response.content_part.done",
                                          "sequence_number": nxt(), "item_id": text_item_id,
                                          "output_index": text_out_index, "content_index": 0,
                                          "part": {"type": "output_text", "text": text_buf}})
                                msg_item = {"type": "message", "id": text_item_id, "status": "completed",
                                            "role": "assistant",
                                            "content": [{"type": "output_text", "text": text_buf}]}
                                yield sse("response.output_item.done", {"type": "response.output_item.done",
                                          "sequence_number": nxt(), "output_index": text_out_index, "item": msg_item})
                                completed_items.append(msg_item)
                                text_open = False
                                out_index += 1
                            tc["out_index"] = out_index
                            out_index += 1
                            tc["opened"] = True
                            yield sse("response.output_item.added", {"type": "response.output_item.added",
                                      "sequence_number": nxt(), "output_index": tc["out_index"],
                                      "item": {"type": "function_call", "id": tc["item_id"],
                                               "call_id": tc["id"], "name": tc["name"], "arguments": ""}})
                        if fn.get("arguments"):
                            tc["args"] += fn["arguments"]
                            yield sse("response.function_call_arguments.delta", {
                                "type": "response.function_call_arguments.delta", "sequence_number": nxt(),
                                "item_id": tc["item_id"], "output_index": tc["out_index"],
                                "delta": fn["arguments"]})
        finally:
            await up_ctx.__aexit__(None, None, None)

    # ---- 收尾：只在上游流真正结束后才做（修复 tool call 提前中断） ----
    # 若 reasoning 还开着（整轮只有 reasoning、无正文/工具），也要收掉
    if reasoning_open:
        yield sse("response.reasoning_summary_text.done", {
            "type": "response.reasoning_summary_text.done", "sequence_number": nxt(),
            "item_id": reasoning_item_id, "output_index": reasoning_out_index,
            "summary_index": 0, "text": reasoning_buf})
        yield sse("response.reasoning_summary_part.done", {
            "type": "response.reasoning_summary_part.done", "sequence_number": nxt(),
            "item_id": reasoning_item_id, "output_index": reasoning_out_index,
            "summary_index": 0, "part": {"type": "summary_text", "text": reasoning_buf}})
        done_reason = {"type": "reasoning", "id": reasoning_item_id,
                       "summary": [{"type": "summary_text", "text": reasoning_buf}]}
        yield sse("response.output_item.done", {"type": "response.output_item.done",
                  "sequence_number": nxt(), "output_index": reasoning_out_index, "item": done_reason})
        completed_items.append(done_reason)
        reasoning_open = False

    # 收掉仍开着的正文 item，填入累积的完整文本（修复 Codex 收不到内容）
    if text_open:
        yield sse("response.output_text.done", {"type": "response.output_text.done",
                  "sequence_number": nxt(), "item_id": text_item_id,
                  "output_index": text_out_index, "content_index": 0, "text": text_buf})
        yield sse("response.content_part.done", {"type": "response.content_part.done",
                  "sequence_number": nxt(), "item_id": text_item_id,
                  "output_index": text_out_index, "content_index": 0,
                  "part": {"type": "output_text", "text": text_buf}})
        msg_item = {"type": "message", "id": text_item_id, "status": "completed",
                    "role": "assistant", "content": [{"type": "output_text", "text": text_buf}]}
        yield sse("response.output_item.done", {"type": "response.output_item.done",
                  "sequence_number": nxt(), "output_index": text_out_index, "item": msg_item})
        completed_items.append(msg_item)
        text_open = False

    # 关闭所有工具调用 item（参数补全后一次性 done）
    for idx in sorted(tool_calls.keys()):
        tc = tool_calls[idx]
        if not tc["opened"]:
            continue
        yield sse("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done", "sequence_number": nxt(),
            "item_id": tc["item_id"], "output_index": tc["out_index"], "arguments": tc["args"]})
        done_item = {"type": "function_call", "id": tc["item_id"], "status": "completed",
                     "call_id": tc["id"], "name": tc["name"], "arguments": tc["args"]}
        yield sse("response.output_item.done", {"type": "response.output_item.done",
                  "sequence_number": nxt(), "output_index": tc["out_index"], "item": done_item})
        completed_items.append(done_item)

    # completed 的 output 汇总所有已收尾 item（Codex 靠它重建完整消息）
    final_resp = {"id": resp_id, "object": "response", "created_at": created,
                  "model": model, "status": "completed", "output": completed_items}
    yield sse("response.completed", {"type": "response.completed",
              "sequence_number": nxt(), "response": final_resp})


async def native_responses_stream(payload: dict, model: str) -> AsyncGenerator[str, None]:
    """把 /v1/responses 请求「原样」转发到上游原生 /v1/responses，
    并把上游的 SSE 字节流逐块透传回 Codex。保留 reasoning / function_call /
    自动工具调用等一切原生语义（不做任何 chat 转换）。"""
    url = f"{UPSTREAM_BASE_URL}/responses"
    headers = {"Authorization": f"Bearer {UPSTREAM_API_KEY}",
               "Content-Type": "application/json",
               "Accept": "text/event-stream"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        attempt = 0
        while True:
            attempt += 1
            up_ctx = client.stream("POST", url, headers=headers, json=payload)
            up = await up_ctx.__aenter__()
            if up.status_code >= 400:
                body_bytes = await up.aread()
                err_text = body_bytes.decode("utf-8", "replace")
                await up_ctx.__aexit__(None, None, None)
                retryable = (up.status_code in (429, 500, 502, 503, 504)
                             or "overload" in err_text.lower())
                if retryable and attempt <= MAX_RETRIES:
                    delay = min(2 * attempt, 12)
                    print(f"[RETRY {attempt}/{MAX_RETRIES}] upstream {up.status_code}, "
                          f"{delay}s 后重试: {err_text[:120]}", flush=True)
                    await asyncio.sleep(delay)
                    continue
                # 不可重试：把上游错误包装成合规的 response.failed 事件，Codex 才能显示
                print(f"[UPSTREAM ERROR] {up.status_code}: {err_text}", flush=True)
                fail_resp = {"id": "resp_" + uuid.uuid4().hex, "object": "response",
                             "created_at": _now(), "model": model, "status": "failed",
                             "error": {"code": f"upstream_{up.status_code}", "message": err_text}}
                yield sse("response.failed", {"type": "response.failed",
                          "sequence_number": 0, "response": fail_resp})
                return
            break
        try:
            # 逐块透传上游 SSE 字节（httpx 已解压，SSE 分帧完整保留）
            async for chunk in up.aiter_bytes():
                if chunk:
                    yield chunk.decode("utf-8", "replace")
        finally:
            await up_ctx.__aexit__(None, None, None)


async def native_responses_once(payload: dict) -> JSONResponse:
    """非流式：直接转发到上游原生 /v1/responses，原样回传 JSON。"""
    url = f"{UPSTREAM_BASE_URL}/responses"
    headers = {"Authorization": f"Bearer {UPSTREAM_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        attempt = 0
        while True:
            attempt += 1
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                retryable = (r.status_code in (429, 500, 502, 503, 504)
                             or "overload" in r.text.lower())
                if retryable and attempt <= MAX_RETRIES:
                    delay = min(2 * attempt, 12)
                    print(f"[RETRY {attempt}/{MAX_RETRIES}] upstream {r.status_code}, "
                          f"{delay}s 后重试: {r.text[:120]}", flush=True)
                    await asyncio.sleep(delay)
                    continue
                print(f"[UPSTREAM ERROR] {r.status_code}: {r.text}", flush=True)
            break
    try:
        return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception:
        return JSONResponse(status_code=r.status_code, content={"error": r.text})


@app.get("/health")
async def health():
    return {"ok": True, "upstream": UPSTREAM_BASE_URL, "wire_api": UPSTREAM_WIRE_API}


@app.get("/v1/models")
async def list_models():
    """转发上游的模型列表，供 Codex++ 拉取可用模型。"""
    headers = {"Authorization": f"Bearer {UPSTREAM_API_KEY}"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        r = await client.get(f"{UPSTREAM_BASE_URL}/models", headers=headers)
    return JSONResponse(status_code=r.status_code, content=r.json())


@app.post("/v1/responses")
async def responses_endpoint(request: Request):
    payload = await request.json()
    model = payload.get("model")
    stream = bool(payload.get("stream", False))
    # 调试：打印 Codex 实际发来的工具类型
    _tools = payload.get("tools") or []
    _ttypes = [(t.get("type"), t.get("name")) for t in _tools]
    print(f"[REQ] wire={UPSTREAM_WIRE_API} model={model} stream={stream} tools={_ttypes}", flush=True)

    # ---- 默认：原生 responses 透传（保留自动工具调用） ----
    if UPSTREAM_WIRE_API != "chat":
        # 只改写模型名，其余字段原样转发
        if UPSTREAM_MODEL:
            payload["model"] = UPSTREAM_MODEL
        if stream:
            return StreamingResponse(
                native_responses_stream(payload, model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return await native_responses_once(payload)

    # ---- 兼容路径：上游只会 chat/completions，走内部转换 ----
    chat_body = responses_to_chat(payload)

    if stream:
        chat_body["stream"] = True
        return StreamingResponse(
            stream_responses(chat_body, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # 非流式：调上游一次，拼成 responses 对象
    chat_body["stream"] = False
    headers = {"Authorization": f"Bearer {UPSTREAM_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        r = await client.post(f"{UPSTREAM_BASE_URL}/chat/completions", headers=headers, json=chat_body)
    if r.status_code >= 400:
        return JSONResponse(status_code=r.status_code, content={"error": r.text})

    data = r.json()
    msg = (data.get("choices") or [{}])[0].get("message", {})
    output: list[dict] = []
    if msg.get("content"):
        output.append({"type": "message", "id": "msg_" + uuid.uuid4().hex,
                       "status": "completed", "role": "assistant",
                       "content": [{"type": "output_text", "text": msg["content"]}]})
    for tc in msg.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        output.append({"type": "function_call", "id": "fc_" + uuid.uuid4().hex,
                       "status": "completed", "call_id": tc.get("id", ""),
                       "name": fn.get("name", ""), "arguments": fn.get("arguments", "") or "{}"})
    return JSONResponse(content={
        "id": "resp_" + uuid.uuid4().hex, "object": "response", "created_at": _now(),
        "model": model, "status": "completed", "output": output,
        "usage": data.get("usage", {}),
    })

