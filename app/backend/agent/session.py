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
import subprocess
import uuid
from pathlib import Path

import anyio
from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ClaudeSDKError,
                              ResultMessage, StreamEvent, TextBlock)

from ..ect.paths import AGENT_HOME, PROJECT_ROOT
from .hooks import build_hooks
from .sysmem import free_commit_mb
from .tools import SERVER_NAME, TOOL_NAMES, Ctx, build_server

MODEL = os.environ.get("ECT_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("ECT_EFFORT", "medium")
SKILLS = ["auto-analysis", "border-crop", "contrast", "crack-detection", "drawing", "python-analysis"]
PROMPT_FILE = Path(__file__).with_name("system_prompt.md")

# One Claude Code CLI process commits ~0.5 GB. With less free commit memory than this it crashes
# at start-up (exit 0xC0000409 / 0x80000003, nothing on stderr): a crash is then blamed on memory.
LOW_MEMORY_MB = 1000

# The desktop program runs under pythonw, which has no console: Windows would give the CLI (a
# console program) its own terminal window, and closing that window kills the agent. The SDK
# starts the CLI through anyio.open_process, so default that to "no window".
if os.name == "nt":
    _open_process = anyio.open_process

    async def _open_process_no_window(*args, creationflags=0, **kwargs):
        return await _open_process(*args, creationflags=creationflags | subprocess.CREATE_NO_WINDOW, **kwargs)

    anyio.open_process = _open_process_no_window


def explain_failure(e):
    """Error text for the chat when the CLI fails: name the usual cause (memory) in plain words."""
    raw = f"{type(e).__name__}: {e}".strip()
    free = free_commit_mb()
    if isinstance(e, ClaudeSDKError) and free is not None and free < LOW_MEMORY_MB:
        return (f"The assistant could not run because this PC is low on memory (about {free} MB free, the "
                f"assistant needs about {LOW_MEMORY_MB} MB). Close some programs (browser, Discord…) and send "
                f"the question again; the chat is kept.\n\nDetails: {raw}")
    return raw

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
        self.cli_session = None                     # Claude Code's own session id, to resume after close()
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
            resume=self.cli_session,                # continue this chat after its CLI was stopped
            permission_mode="dontAsk",              # anything not pre-approved is refused, never prompted
            model=MODEL,
            effort=EFFORT,
            include_partial_messages=True,
            hooks=build_hooks(self.ctx),
            max_turns=40,
            stderr=self._cli_stderr,
        )

    def _cli_stderr(self, line):
        """The bundled Claude Code CLI's stderr: kept for diagnosing start-up failures."""
        self.stderr_tail = (getattr(self, "stderr_tail", []) + [line])[-40:]
        try:
            from ..ect.paths import LOG_DIR, ensure_state_dirs
            ensure_state_dirs()
            with open(LOG_DIR / "agent_cli.log", "a", encoding="utf-8") as f:
                f.write(f"[{self.id}] {line.rstrip()}\n")
        except Exception:
            pass

    async def close(self):
        """Stop the CLI process. The chat itself survives: the next ask() resumes it."""
        client, self.client = self.client, None
        if client is not None:
            try:
                await client.disconnect()
            except Exception:
                pass                                # already dead (crashed or killed)

    async def ask(self, text):
        async with self.lock:
            if self.client is None:
                client = ClaudeSDKClient(options=self.options())
                try:
                    await client.connect()
                except Exception as e:              # usually: the CLI crashed for lack of memory
                    try:
                        await client.disconnect()
                    except Exception:
                        pass
                    yield "error", {"message": explain_failure(e)}
                    return
                self.client = client
            client = self.client
            q = self.queue = asyncio.Queue()
            failed = []

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
                            self.cli_session = msg.session_id or self.cli_session
                            await q.put(("done", {"turns": msg.num_turns, "ms": msg.duration_ms,
                                                  "error": msg.is_error, "detail": msg.errors}))
                except Exception as e:
                    failed.append(e)
                    await q.put(("error", {"message": explain_failure(e)}))
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
                if failed and self.client is client:    # the CLI broke: start a fresh one next time
                    await self.close()
