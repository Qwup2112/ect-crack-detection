# UI design prompt for Google AI Studio

**How to use it**

1. Open [Google AI Studio](https://aistudio.google.com) and start a new chat (or the Build section).
2. Paste the whole prompt block below and send it.
3. AI Studio returns a single `index.html` file. Save it over `app/frontend/index.html`.
4. Restart `run_app.bat`. The new UI works with the existing backend at once; the backend needs
   no change. If you like, give the file to Claude to check and merge.

In AI Studio's preview the backend is not running, so the UI switches itself to **demo mode**
(simulated answers and one fake image) so you can judge the design.

---

```text
Build ONE self-contained file, index.html: HTML + inline CSS + inline vanilla JavaScript.
No frameworks, no build step, no external scripts, fonts or images, and NO calls to any AI
model or third-party API (no Gemini, no OpenAI). The page is the front end of a local desktop
app; all intelligence lives in an existing backend on the same origin, described below.

APP
"ECT Crack Chat" - a chat assistant for eddy-current (ECT) crack-scan images. The user types
questions or drawing commands ("Draw the heatmap of file R 1.7V 300k", "Crop the border",
"Boost the contrast", "Detect cracks"); the backend answers in streamed text and produces figures
(PNG heatmaps with crack boxes). The window has exactly two parts and nothing else:
  LEFT  = chat and the assistant's replies
  RIGHT = the image results

LAYOUT (desktop, full window, no page scroll)
- Two columns: left 40% (min 360px), right 60%. Under 900px wide, stack left above right.
- LEFT column, top to bottom:
  1. Top bar: app name with a small logo drawn in CSS or inline SVG (three concentric copper
     rings, like an eddy-current coil); a status dot (green = ready, amber pulsing = answering,
     red = backend unreachable) with a tooltip; a "New chat" button.
  2. Message list (aria-live="polite"), auto-scrolls to the newest item:
     - user messages: right-aligned copper bubbles, plain text;
     - assistant messages: left-aligned dark bubbles; render a safe markdown subset
       (paragraphs, "- " / "1." lists, **bold**, `code`); escape all HTML;
     - tool chips between messages: small pill with a status dot and a label, e.g.
       "Detecting cracks…" - pulsing while running, green with duration ("· 2.4 s") when
       done, red with "— error" on error;
     - image chips: "🖼 <title>"; clicking one shows that image on the right.
     - Empty state: heading "Ask a question or give a drawing command" and these suggestion
       chips (click = send):
       "Which scan files are there?", "Draw the heatmap of file R 1.7V 300k",
       "Auto-crop the border of file R 1.7V 300k",
       "Boost the contrast of cam_4lop_1 R 4V 10k so the cracks stand out",
       "Detect cracks in file R 1.7V 300k", "Compare R and P of cambien 1.3V 200k 145x95".
  3. Composer: auto-growing textarea (max ~6 lines), placeholder "Type a question or a drawing
     command… (Enter to send, Shift+Enter for a new line)". Enter sends, Shift+Enter adds a
     line. The "Send" button becomes "Stop" while an answer streams; pressing it aborts the
     request.
- RIGHT column, top to bottom:
  1. Top bar: title of the image on display (ellipsis if long) and an "Open original" link that
     opens the PNG in a new tab.
  2. Stage: the current image on a white rounded card with a soft shadow, fitted to the area
     (object-fit: contain). Click toggles 100% zoom with scrolling; click again to fit.
     When there is no image: a dashed placeholder "Figures drawn by the assistant appear here."
  3. Caption line (small, muted).
  4. Filmstrip: horizontal thumbnails (64px high) of every image in this chat, newest at the
     end, the displayed one outlined in copper; click to display.
- A new image arriving from the backend is displayed immediately and added to the filmstrip.

VISUAL STYLE
Dark, calm, technical. Tokens: background #12161b; panels #181d24 and #1f252e; lines #2a313b;
text #e6e9ee; muted #9aa5b1; accent copper #c8733a (hover #e08a4f); success #2fbf71; error
#ff6b6b. Radius 12px (bubbles), 999px (chips). Font: "Segoe UI", system-ui. Visible focus
rings. All visible text in English as given above.

BACKEND CONTRACT (same origin, use relative URLs, do not change)
- GET  /api/health -> {"ok":true,"model":"claude-opus-5-5","readable":101,"scans":110,"yolo_weights":true}
  Call on load; show model and "<readable>/<scans> files readable" in the status tooltip.
- POST /api/chat, JSON body {"message": string, "session_id": string|null}
  Response: a Server-Sent Events stream (Content-Type text/event-stream). Read it with
  fetch() + response.body.getReader() (EventSource cannot POST). Events are separated by a
  blank line; each has "event: <kind>" and "data: <json>". Kinds:
    session {id}                         -> remember as session_id for later requests
    delta   {text}                       -> append to the current assistant bubble
    text    {text}                       -> append a whole block (same handling as delta)
    tool    {id, name, label, status, ms?, error?}   status = "start" | "end" | "error";
            "start" ends the current assistant bubble and adds a tool chip; later events
            with the same id update that chip; text after a tool goes into a NEW bubble
    image   {id, url, title, caption}    -> show on the right + filmstrip + image chip
    done    {turns, ms, error, detail}   -> answer finished; if error, show detail in red
    error   {message}                    -> show as a red assistant bubble
- GET  /api/session/{session_id}/images -> [{id, url, title, caption}] - on load, restore
  the filmstrip of the saved session.
- POST /api/session/{session_id}/reset  -> called by "New chat", then clear the UI.
- Images: GET <url> returns a PNG.
- Keep session_id and the last 60 chat messages (role + text) in localStorage and re-render
  them on load.

DEMO MODE
If /api/health fails (for example in this preview), switch to a clearly labelled
"Demo mode": answers are simulated - stream a short English reply word by word, show a
tool chip "Drawing map" that completes after 1.5 s, and display a generated placeholder heatmap
(draw a 600x400 colour gradient with three dark ellipses on a <canvas>, use its data URL).
Demo mode must never be active when the backend answers.

DELIVERABLE
Return only the complete index.html, ready to save and open.
```
