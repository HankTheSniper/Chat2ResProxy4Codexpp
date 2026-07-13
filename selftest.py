"""端到端自测（无需真实上游）：

  1) native 透传路径（默认 UPSTREAM_WIRE_API=responses）
     假上游实现原生 /v1/responses(SSE)，验证代理把 reasoning / 正文 /
     function_call 事件原样透传，且 tool_choice=auto 时工具调用能穿过。

  2) chat 兼容路径（UPSTREAM_WIRE_API=chat）
     假上游实现 /v1/chat/completions(SSE)，验证内部 responses<->chat 转换
     修掉三个老 bug：reasoning 泄漏、正文重复、tool call 提前中断。

两条路径各起一套「假上游 + 代理」，端口互不冲突。
"""
import json, threading, time, os, importlib
import uvicorn, httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


def _serve(app, port):
    threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=port, log_level="error"),
        daemon=True,
    ).start()


def _collect(proxy_port, body):
    events = []
    with httpx.Client(timeout=30) as c:
        with c.stream("POST", f"http://127.0.0.1:{proxy_port}/v1/responses", json=body) as r:
            cur = None
            for line in r.iter_lines():
                if line.startswith("event:"):
                    cur = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    events.append((cur, json.loads(line.split(":", 1)[1].strip())))
    return events


# ---------------- 假上游 A：原生 responses ----------------
native_up = FastAPI()

@native_up.post("/v1/responses")
async def native(request: Request):
    def gen():
        def ev(name, data):
            return f"event: {name}\ndata: {json.dumps(data)}\n\n"
        rid = "resp_x"
        yield ev("response.created", {"type": "response.created", "response": {"id": rid}})
        # 一个 function_call item，参数分片
        yield ev("response.output_item.added", {"type": "response.output_item.added", "output_index": 0,
                 "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_file", "arguments": ""}})
        yield ev("response.function_call_arguments.delta", {"type": "response.function_call_arguments.delta",
                 "item_id": "fc_1", "output_index": 0, "delta": '{"path":'})
        yield ev("response.function_call_arguments.delta", {"type": "response.function_call_arguments.delta",
                 "item_id": "fc_1", "output_index": 0, "delta": '"a.css"}'})
        yield ev("response.function_call_arguments.done", {"type": "response.function_call_arguments.done",
                 "item_id": "fc_1", "output_index": 0, "arguments": '{"path":"a.css"}'})
        yield ev("response.completed", {"type": "response.completed",
                 "response": {"id": rid, "status": "completed",
                              "output": [{"type": "function_call", "call_id": "call_1",
                                          "name": "read_file", "arguments": '{"path":"a.css"}'}]}})
    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------- 假上游 B：chat/completions ----------------
chat_up = FastAPI()

@chat_up.post("/v1/chat/completions")
async def cc(request: Request):
    def gen():
        def chunk(delta, finish=None):
            return f"data: {json.dumps({'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})}\n\n"
        yield chunk({"reasoning_content": "Need inspect"})
        yield chunk({"reasoning_content": " relevant CSS."})
        yield chunk({"content": "Hello"})
        yield chunk({"content": " world"})
        yield chunk({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": ""}}]})
        yield chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"path":'}}]})
        yield chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"a.css"}'}}]})
        yield chunk({}, finish="tool_calls")
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


def test_native():
    os.environ["UPSTREAM_BASE_URL"] = "http://127.0.0.1:9101/v1"
    os.environ["UPSTREAM_API_KEY"] = "test"
    os.environ["UPSTREAM_WIRE_API"] = "responses"
    os.environ.pop("UPSTREAM_MODEL", None)
    import proxy; importlib.reload(proxy)
    _serve(proxy.app, 8131)
    time.sleep(1.2)
    body = {"model": "gpt-5-codex", "stream": True, "tool_choice": "auto",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read a.css"}]}],
            "tools": [{"type": "function", "name": "read_file", "parameters": {}}]}
    events = _collect(8131, body)
    args = "".join(e["delta"] for t, e in events if t == "response.function_call_arguments.delta")
    has_fc = any(t == "response.output_item.added" and e.get("item", {}).get("type") == "function_call" for t, e in events)
    ok = has_fc and args == '{"path":"a.css"}' and "response.completed" in [t for t, _ in events]
    print(f"[native 透传] function_call={has_fc} args={args!r} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_chat():
    os.environ["UPSTREAM_BASE_URL"] = "http://127.0.0.1:9102/v1"
    os.environ["UPSTREAM_API_KEY"] = "test"
    os.environ["UPSTREAM_WIRE_API"] = "chat"
    os.environ.pop("UPSTREAM_MODEL", None)
    import proxy; importlib.reload(proxy)
    _serve(proxy.app, 8132)
    time.sleep(1.2)
    body = {"model": "test", "stream": True,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}]}
    events = _collect(8132, body)
    text = "".join(e["delta"] for t, e in events if t == "response.output_text.delta")
    reasoning = "".join(e["delta"] for t, e in events if t == "response.reasoning_summary_text.delta")
    tool_args = "".join(e["delta"] for t, e in events if t == "response.function_call_arguments.delta")
    ok = (text == "Hello world" and reasoning == "Need inspect relevant CSS."
          and tool_args == '{"path":"a.css"}' and "response.completed" in [t for t, _ in events])
    print(f"[chat 转换] text={text!r} reasoning-ok={reasoning=='Need inspect relevant CSS.'} args={tool_args!r} -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _serve(native_up, 9101)
    _serve(chat_up, 9102)
    time.sleep(1.5)
    r1 = test_native()
    r2 = test_chat()
    print("\n=== 总结 ===")
    print("PASS" if (r1 and r2) else "FAIL")
