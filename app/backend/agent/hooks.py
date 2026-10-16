"""Hooks: the harness calls these around every tool call and every user message.

PreToolUse       - allow only this app's tools (and Skill); resolve loose scan names
                   ("R 1.7V 300k") to one exact id, or deny with the candidates so the
                   model asks instead of guessing; announce the tool in the chat.
PostToolUse      - mark the tool done in the chat; append to the audit log.
PostToolUseFailure - mark it failed.
UserPromptSubmit - inject the long-term memory digest and this chat's state (crop and
                   contrast choices, last scan, last detection) into the prompt.
"""
import json
import time

from claude_agent_sdk import HookMatcher

from ..ect import scan as sc
from ..ect.paths import LOG_DIR, ensure_state_dirs
from . import memory
from .tools import LABELS, MAX_COMPARE, SERVER_NAME, TOOL_NAMES

PREFIX = f"mcp__{SERVER_NAME}__"


def _short(name):
    return name[len(PREFIX):] if name.startswith(PREFIX) else name


def _deny(reason):
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


def _log(record):
    ensure_state_dirs()
    with open(LOG_DIR / "tools.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def build_hooks(ctx):
    async def guard(inp, tool_use_id, context):
        name = inp["tool_name"]
        args = dict(inp.get("tool_input") or {})
        if name == "Skill":
            skill = args.get("skill") or args.get("name") or ""
            await ctx.emit("tool", {"id": tool_use_id, "name": "Skill", "status": "start",
                                    "label": f"Đọc kỹ năng: {skill}"})
            return {}
        short = _short(name)
        if not name.startswith(PREFIX) or short not in TOOL_NAMES:
            return _deny(f"'{name}' is not available in this app; use the {SERVER_NAME} tools.")
        try:
            if args.get("scan"):
                args["scan"] = sc.resolve(args["scan"])
            if short == "compare_scans":
                scans = args.get("scans") or []
                if len(scans) > MAX_COMPARE:
                    return _deny(f"compare at most {MAX_COMPARE} scans at once; ask the user which ones.")
                args["scans"] = [sc.resolve(s) for s in scans]
        except ValueError as e:
            return _deny(f"{e} Ask the user which scan they mean, or call list_scans.")
        await ctx.emit("tool", {"id": tool_use_id, "name": short, "status": "start", "label": LABELS[short],
                                "input": args})
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                       "updatedInput": args}}

    async def done(inp, tool_use_id, context):
        short = _short(inp["tool_name"])
        await ctx.emit("tool", {"id": tool_use_id, "name": short, "status": "end",
                                "ms": inp.get("duration_ms")})
        _log({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "session": ctx.session_id, "tool": short,
              "input": inp.get("tool_input"), "ms": inp.get("duration_ms")})
        return {}

    async def failed(inp, tool_use_id, context):
        short = _short(inp["tool_name"])
        await ctx.emit("tool", {"id": tool_use_id, "name": short, "status": "error",
                                "error": str(inp.get("error", ""))[:300]})
        _log({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "session": ctx.session_id, "tool": short,
              "input": inp.get("tool_input"), "error": inp.get("error")})
        return {}

    async def inject(inp, tool_use_id, context):
        st = ctx.state
        parts = []
        dig = memory.digest()
        if dig:
            parts.append("<long_term_memory>\n" + dig + "\n</long_term_memory>")
        chat = []
        if st.get("last_scan"):
            chat.append(f"last scan: {st['last_scan']}")
        if st.get("crop"):
            chat.append("crop chosen in this chat: " + json.dumps(st["crop"], ensure_ascii=False))
        if st.get("enhance"):
            chat.append("contrast chosen in this chat: " + json.dumps(st["enhance"], ensure_ascii=False))
        for sid, boxes in (st.get("detections") or {}).items():
            chat.append(f"last detection on {sid}: {len(boxes)} cracks (numbered #1..#{len(boxes)})")
        if st.get("last_image"):
            chat.append(f"image on the right panel now: {st['last_image']}")
        if chat:
            parts.append("<chat_state>\n" + "\n".join(chat) + "\n</chat_state>")
        if not parts:
            return {}
        return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                       "additionalContext": "\n".join(parts)}}

    return {
        "PreToolUse": [HookMatcher(matcher=None, hooks=[guard])],
        "PostToolUse": [HookMatcher(matcher=None, hooks=[done])],
        "PostToolUseFailure": [HookMatcher(matcher=None, hooks=[failed])],
        "UserPromptSubmit": [HookMatcher(matcher=None, hooks=[inject])],
    }
