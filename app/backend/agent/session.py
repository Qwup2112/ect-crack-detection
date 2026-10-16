"""One chat = one ClaudeSDKClient (the Claude Code harness) + an event stream for the UI.

Event kinds sent to the frontend:
  delta {text}            streamed piece of the assistant's reply
  text {text}             a whole reply block (only when it was not streamed)
  tool {id, name, label, status: start|end|error}   from the hooks
  image {id, url, title, caption}                   from the tools
  done {turns, ms, error, detail}
  error {message}
"""
import asyncio
import json
import os
import uuid
from pathlib import Path

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage,
                              StreamEvent, TextBlock)

from ..ect.paths import AGENT_HOME, PROJECT_ROOT
from .hooks import build_hooks
from .tools import SERVER_NAME, TOOL_NAMES, Ctx, build_server

MODEL = os.environ.get("ECT_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("ECT_EFFORT", "medium")
SKILLS = ["cat-vien", "bat-tong", "nhan-dien-vet-nut", "ve-hinh"]
PROMPT_FILE = Path(__file__).with_name("system_prompt.md")

# The harness reads CLAUDE.md files from the agent's folder AND its parents. The project root's
# CLAUDE.md is written for developers, not for this agent: exclude it. Claude Code's own auto
# memory is off too - the app keeps its memory in app/state/memory.json (agent/memory.py).
_root = PROJECT_ROOT.as_posix()
AGENT_SETTINGS = json.dumps({
    "claudeMdExcludes": [f"{_root}/CLAUDE.md", f"{_root}/CLAUDE.local.md", f"{_root}/.claude/CLAUDE.md",
                         str(PROJECT_ROOT / "CLAUDE.md"), str(PROJECT_ROOT / ".claude" / "CLAUDE.md")],
    "autoMemoryEnabled": False,
})


class ChatSession:
    def __init__(self, sid=None):
        self.id = sid or uuid.uuid4().hex[:12]
        self.state = {}
        self.client = None
        self.queue = None
        self.lock = asyncio.Lock()
        self.ctx = Ctx(self.id, self.emit, self.state)

    async def emit(self, kind, data):
        if self.queue is not None:
            await self.queue.put((kind, data))

    def options(self):
        return ClaudeAgentOptions(
            system_prompt={"type": "file", "path": str(PROMPT_FILE)},
            mcp_servers={SERVER_NAME: build_server(self.ctx)},
            strict_mcp_config=True,                 # ignore any other MCP servers on this machine
            tools=["Skill"],                        # no Bash/Read/Edit...: only Skill + our tools
            allowed_tools=[f"mcp__{SERVER_NAME}__{n}" for n in TOOL_NAMES],
            skills=SKILLS,
            setting_sources=["project"],            # agent_home/.claude only, not the user's settings
            settings=AGENT_SETTINGS,
            cwd=str(AGENT_HOME),
            permission_mode="dontAsk",              # anything not pre-approved is refused, never prompted
            model=MODEL,
            effort=EFFORT,
            include_partial_messages=True,
            hooks=build_hooks(self.ctx),
            max_turns=40,
        )

    async def close(self):
        if self.client is not None:
            try:
                await self.client.disconnect()
            finally:
                self.client = None

    async def ask(self, text):
        async with self.lock:
            q = self.queue = asyncio.Queue()
            if self.client is None:
                self.client = ClaudeSDKClient(options=self.options())
                await self.client.connect()
            client = self.client

            async def pump():
                streamed = False
                try:
                    await client.query(text)
                    async for msg in client.receive_response():
                        if isinstance(msg, StreamEvent):
                            ev = msg.event
                            if msg.parent_tool_use_id is None and ev.get("type") == "content_block_delta" \
                                    and ev.get("delta", {}).get("type") == "text_delta":
                                streamed = True
                                await q.put(("delta", {"text": ev["delta"]["text"]}))
                        elif isinstance(msg, AssistantMessage):
                            if msg.parent_tool_use_id is None and not streamed:
                                for b in msg.content:
                                    if isinstance(b, TextBlock) and b.text.strip():
                                        await q.put(("text", {"text": b.text}))
                            streamed = False
                        elif isinstance(msg, ResultMessage):
                            await q.put(("done", {"turns": msg.num_turns, "ms": msg.duration_ms,
                                                  "error": msg.is_error, "detail": msg.errors}))
                except Exception as e:
                    await q.put(("error", {"message": f"{type(e).__name__}: {e}"}))
                finally:
                    await q.put(None)

            task = asyncio.create_task(pump())
            try:
                while True:
                    item = await q.get()
                    if item is None:
                        break
                    yield item
            finally:
                if not task.done():                 # the browser went away mid-answer
                    try:
                        await client.interrupt()
                    except Exception:
                        pass
                    try:
                        await asyncio.wait_for(task, 30)
                    except Exception:
                        task.cancel()
                self.queue = None
