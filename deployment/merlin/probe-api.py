"""Check a real tool roundtrip using the same generation settings as the agent."""
import asyncio
import json
import time
import uuid
from pathlib import Path

from Emerge.config.schema import Config
from Emerge.providers.factory import create_provider


async def probe(config):
    provider = create_provider(config)
    start = time.monotonic()
    report = {"i": 0, "ok": False}
    try:
        messages = [{"role": "user", "content":
            "Call check with value OK exactly once, then reply only with the token returned by the tool."}]
        tools = [{"type": "function", "function": {"name": "check", "description": "Read a test token.",
            "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                           "required": ["value"], "additionalProperties": False}}}]
        # Inherit effort and token budget: a 128-token override can truncate reasoning.
        first = await provider.chat_with_retry(messages, tools=tools)
        report["first_finish"] = first.finish_reason
        if (first.finish_reason != "tool_calls" or len(first.tool_calls) != 1
                or first.tool_calls[0].name != "check" or first.tool_calls[0].arguments != {"value": "OK"}):
            return report
        call = first.tool_calls[0]
        token = "probe-" + uuid.uuid4().hex[:12]
        messages.extend([
            {"role": "assistant", "content": first.content, "tool_calls": [call.to_openai_tool_call()]},
            {"role": "tool", "tool_call_id": call.id, "content": json.dumps({"token": token})},
        ])
        final = await provider.chat_with_retry(messages, tools=tools)
        report.update(ok=final.finish_reason == "stop" and (final.content or "").strip() == token,
                      finish=final.finish_reason, tool_result_verified=(final.content or "").strip() == token)
        return report
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        return report
    finally:
        await provider.aclose()
        report["seconds"] = round(time.monotonic() - start, 2)


async def main():
    config = Config.model_validate(json.loads(Path('/home/tiger/.config/emerge-robodojo/agent.json').read_text()))
    result = await probe(config)
    print(json.dumps({"concurrency": 1, "passed": int(result["ok"]), "results": [result]}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
