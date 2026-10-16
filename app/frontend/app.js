// ECT Crack Chat - temporary UI. Talks to the backend through:
//   POST /api/chat {message, session_id}  -> text/event-stream
//        events: session{id} delta{text} text{text} tool{id,name,label,status} image{id,url,title,caption}
//                done{turns,ms,error} error{message}
//   GET  /api/session/{id}/images, POST /api/session/{id}/reset, GET /api/health
const $ = (s) => document.querySelector(s);
const messages = $("#messages"), input = $("#input"), sendBtn = $("#send"), statusEl = $("#status");
const mainImg = $("#mainImg"), placeholder = $("#placeholder"), strip = $("#strip");

let sessionId = localStorage.getItem("ect.session") || null;
let transcript = JSON.parse(localStorage.getItem("ect.transcript") || "[]");
let busy = false, controller = null, shown = [];

// ---------------------------------------------------------------- rendering helpers
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function md(text) {             // small, safe markdown: paragraphs, lists, **bold**, `code`
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
  const out = [];
  let list = null;
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const li = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
    if (li) { if (!list) { list = []; out.push(list); } list.push(inline(li[1])); continue; }
    list = null;
    if (line.trim()) out.push(inline(line));
  }
  return out.map((b) => Array.isArray(b) ? `<ul>${b.map((i) => `<li>${i}</li>`).join("")}</ul>` : `<p>${b}</p>`).join("");
}

function scrollDown() { messages.scrollTop = messages.scrollHeight; }
function hideEmpty() { const e = $("#empty"); if (e) e.remove(); }

function addMsg(role, text) {
  hideEmpty();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  if (role === "user") el.textContent = text; else el.innerHTML = md(text);
  messages.appendChild(el);
  scrollDown();
  return el;
}

function setStatus(kind, title) { statusEl.className = `status ${kind}`; statusEl.title = title || ""; }

function showImage(meta, addToStrip = true) {
  mainImg.src = meta.url; mainImg.alt = meta.title || ""; mainImg.hidden = false; mainImg.classList.remove("zoom");
  placeholder.hidden = true;
  $("#imgTitle").textContent = meta.title || "Hình";
  $("#caption").textContent = meta.caption || "";
  const full = $("#openFull"); full.href = meta.url; full.hidden = false;
  if (addToStrip && !shown.some((m) => m.id === meta.id)) {
    shown.push(meta);
    const t = document.createElement("img");
    t.src = meta.url; t.alt = meta.title || ""; t.title = meta.title || ""; t.dataset.id = meta.id;
    t.onclick = () => showImage(meta, false);
    strip.appendChild(t);
    t.scrollIntoView({ inline: "end", behavior: "smooth" });
  }
  strip.querySelectorAll("img").forEach((i) => i.classList.toggle("active", i.dataset.id === meta.id));
}
mainImg.onclick = () => mainImg.classList.toggle("zoom");

// ---------------------------------------------------------------- chat
async function send(text) {
  if (busy || !text.trim()) return;
  addMsg("user", text);
  transcript.push({ role: "user", text });
  input.value = "";
  setBusy(true);

  let bot = null, botText = "";
  const tools = {};
  const flushBot = () => { if (bot && botText.trim()) transcript.push({ role: "bot", text: botText }); bot = null; botText = ""; };
  const appendText = (t) => {
    if (!bot) { bot = addMsg("bot", ""); }
    botText += t; bot.innerHTML = md(botText); scrollDown();
  };

  controller = new AbortController();
  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }), signal: controller.signal,
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        const ev = (chunk.match(/^event: (.*)$/m) || [])[1];
        const dataLine = (chunk.match(/^data: (.*)$/m) || [])[1];
        if (!ev || dataLine === undefined) continue;
        const data = JSON.parse(dataLine);
        if (ev === "session") { sessionId = data.id; localStorage.setItem("ect.session", sessionId); }
        else if (ev === "delta" || ev === "text") appendText(data.text);
        else if (ev === "tool") {
          if (data.status === "start") {
            flushBot();
            const el = document.createElement("div");
            el.className = "tool"; el.innerHTML = `<span class="dot"></span><span>${esc(data.label || data.name)}…</span>`;
            messages.appendChild(el); tools[data.id] = el; scrollDown();
          } else if (tools[data.id]) {
            tools[data.id].classList.add(data.status === "error" ? "error" : "end");
            const ms = data.ms ? ` · ${(data.ms / 1000).toFixed(1)} s` : "";
            tools[data.id].lastChild.textContent = tools[data.id].lastChild.textContent.replace(/…$/, "") + (data.status === "error" ? " — lỗi" : ms);
          }
        } else if (ev === "image") {
          showImage(data);
          const chip = document.createElement("div");
          chip.className = "tool end image"; chip.innerHTML = `<span class="dot"></span><span>🖼 ${esc(data.title)}</span>`;
          chip.onclick = () => showImage(data, false);
          messages.appendChild(chip); scrollDown();
        } else if (ev === "done") {
          flushBot();
          if (data.error) addMsg("bot error", "Phiên trả lời kết thúc với lỗi: " + (data.detail || []).join("; "));
        } else if (ev === "error") { flushBot(); addMsg("bot error", data.message); }
      }
    }
    setStatus("ok", "Sẵn sàng");
  } catch (e) {
    if (e.name === "AbortError") addMsg("bot error", "Đã dừng.");
    else { addMsg("bot error", `Không kết nối được backend: ${e.message}`); setStatus("err", e.message); }
  } finally {
    flushBot();
    localStorage.setItem("ect.transcript", JSON.stringify(transcript.slice(-60)));
    setBusy(false);
  }
}

function setBusy(b) {
  busy = b;
  sendBtn.textContent = b ? "Dừng" : "Gửi";
  sendBtn.classList.toggle("stop", b);
  if (b) setStatus("busy", "Đang trả lời…");
}

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  if (busy) { controller && controller.abort(); return; }
  send(input.value);
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!busy) send(input.value); }
});
document.querySelectorAll(".suggestions button").forEach((b) => (b.onclick = () => send(b.textContent)));

$("#newChat").onclick = async () => {
  if (busy) return;
  if (sessionId) fetch(`/api/session/${sessionId}/reset`, { method: "POST" });
  localStorage.removeItem("ect.session"); localStorage.removeItem("ect.transcript");
  location.reload();
};

// ---------------------------------------------------------------- start-up
(async () => {
  for (const m of transcript) addMsg(m.role, m.text);
  if (sessionId) {
    try {
      const imgs = await (await fetch(`/api/session/${sessionId}/images`)).json();
      imgs.forEach((m) => showImage(m));
    } catch {}
  }
  try {
    const h = await (await fetch("/api/health")).json();
    setStatus("ok", `Sẵn sàng · ${h.model} · ${h.readable}/${h.scans} file đọc được`);
  } catch { setStatus("err", "Backend chưa chạy"); }
  input.focus();
})();
