#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_advisor.py - The AI layer of the application: sends the computed measurement
numbers plus the heatmap figure to Claude and streams back the interpretation.

INPUT  : numbers COMPUTED BY CODE (metadata, border crop, candidate boxes, statistics,
         whole-dataset sweep table) + a PNG of the figure + the user's question.
OUTPUT : an interpretation, streamed chunk by chunk into the GUI.

Key design boundary: the model never computes the numbers itself. Everything it is given
was computed by the backend over the full data; its job is interpretation and diagnosis.
This is deliberate - a language model eyeballing statistics off raw arrays is an
uncontrolled source of error, and the people reading this output make engineering
decisions from it.

INSTALL: pip install anthropic
API KEY: set the ANTHROPIC_API_KEY environment variable (recommended), or enter it in
         the app (kept in memory for the session only, never written to disk).
"""

import base64
import io
import os

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You are a non-destructive testing (NDT) specialist in eddy-current testing (ECT), helping \
an engineer analyse 2D scan images for crack detection. You are running inside an analysis \
application that has already computed every number you are given.

MANDATORY RULES:

1. Never invent or eyeball a number. Use only the values provided under "COMPUTED DATA". \
If you need a quantity that is not there, say so plainly and name the command that would \
produce it, instead of guessing.

2. The boxes under "CANDIDATE LIST" come from a statistical THRESHOLD detector (z-score \
after detrending). They are NOT ground truth and NOT a machine-learning model. Never call \
them ground truth, and never use them to compute or validate any model's accuracy - doing \
so scores a detector against itself.

3. When a result looks wrong, diagnose in this order (each step is far cheaper than the \
next):
   a. Is the background still sloped? (check the z-score map; a row detrend only removes \
      trends along Y - a background sloping along X needs 'both')
   b. Are there border artifacts left? (boxes touching the image edge are mostly scan \
      artifacts)
   c. Does the signal exist at all? CNR < 1 means the problem is in the MEASUREMENT \
      (frequency, channel, scan resolution), not the algorithm. CNR >= 3 is needed before \
      expecting stable detection.
   d. Are the threshold and merge parameters sensible?

4. An ECT crack signature is a DIPOLE (a high-amplitude lobe beside a low-amplitude one). \
Because of this, a CNR computed on signed values collapses toward zero whenever the box \
covers both lobes. If CNR_abs is much higher than the signed CNR, that is the dipole \
fingerprint - say so explicitly and read CNR_abs.

5. End every analysis with a "Limits" section stating what this data does NOT allow you \
to conclude. This is not a formality - it forces the scope of validity to be stated.

6. Prefer "not enough data to conclude" over a confident-sounding claim with no number \
behind it. The reader is making engineering decisions from your answer.

Answer in English. Be concise: lead with the conclusion, then the reasoning. Use bullet \
points for lists. No emoji."""


def make_client(api_key=None):
    """
    Build the client. With no explicit key the SDK resolves credentials in order:
    ANTHROPIC_API_KEY -> ANTHROPIC_AUTH_TOKEN -> an `ant auth login` profile.
    """
    import anthropic
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def has_credentials():
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def figure_to_png_bytes(fig, dpi=100):
    """Figure -> PNG in memory. Moderate dpi so the image does not waste tokens."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="#fcfcfb", bbox_inches="tight")
    return buf.getvalue()


def build_context(meta, params, margins=None, cropped_shape=None,
                  boxes=None, stats=None, compare_rows=None):
    """
    Assemble every number the backend computed into one structured text block.

    This is the AI layer's input. Only computed values go in, never the raw arrays -
    the model does not need them and must not re-derive statistics from them.
    """
    L = []
    L.append("### MEASUREMENT CONDITIONS")
    L.append(f"- File: {meta.get('path', '?')}")
    L.append(f"- Sensor: {meta.get('sensor', '?')}  "
             f"(R = amplitude channel, P = phase channel)")
    L.append(f"- Excitation amplitude: {meta.get('amp', '?')} V")
    L.append(f"- Frequency: {meta.get('freq', '?')} kHz")
    L.append(f"- Lift-off / step: {meta.get('lf', '?')} mm")
    L.append(f"- Scan grid: {meta.get('size_x')} x {meta.get('size_y')} px, "
             f"~{meta.get('pitch_mm', 1.0):.3f} mm/px")

    L.append("\n### PROCESSING PARAMETERS IN USE")
    L.append(f"- Filter: {params.get('filter')} (kernel {params.get('kernel')})")
    L.append(f"- Border crop: {params.get('margin')}")
    L.append(f"- Detrend: {params.get('detrend')}")
    if params.get("z_thresh") is not None:
        L.append(f"- |z| threshold: {params.get('z_thresh')}  "
                 f"| merge gap: {params.get('merge')} px "
                 f"| minimum area: {params.get('min_area')} px")
        L.append(f"- Drop edge-touching boxes: {'yes' if params.get('drop_edge') else 'no'}")

    if margins:
        L.append("\n### BORDER NOISE REMOVED (measured automatically)")
        L.append(f"- Top {margins['top']} px, bottom {margins['bottom']} px, "
                 f"left {margins['left']} px, right {margins['right']} px")
        if cropped_shape:
            L.append(f"- Size after cropping: {cropped_shape[1]} x {cropped_shape[0]} px")

    if stats:
        L.append("\n### AMPLITUDE STATISTICS (computed over all samples)")
        for k, v in stats.items():
            L.append(f"- {k}: {v}")

    if boxes is not None:
        L.append(f"\n### CRACK CANDIDATE LIST ({len(boxes)} boxes)")
        L.append("(threshold z-score detector - candidates, NOT ground truth; "
                 "coordinates are relative to the cropped image)")
        if boxes:
            L.append("| id | x0 | y0 | x1 | y1 | width | height | z score | touches edge |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for b in boxes[:80]:
                L.append(f"| {b['id']} | {b['x0']} | {b['y0']} | {b['x1']} | {b['y1']} "
                         f"| {b['x1']-b['x0']} | {b['y1']-b['y0']} | {b['score']} "
                         f"| {'yes' if b.get('touches_edge') else ''} |")
            if len(boxes) > 80:
                L.append(f"(... {len(boxes) - 80} more boxes omitted)")
        else:
            L.append("No candidate passed the threshold.")

    if compare_rows:
        L.append(f"\n### SENSOR COMPARISON TABLE ({len(compare_rows)} files)")
        cols = list(compare_rows[0].keys())
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "---|" * len(cols))
        for r in compare_rows[:40]:
            L.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        L.append("\nNOTE: if the table has dx_std but no CNR, dx_std is ONLY a relative index "
                 "(it rewards a large value scale) and must not be used to choose a sensor.")

    return "\n".join(L)


def build_sweep_context(df, ok, skipped, best, params, root=None):
    """
    Context for a WHOLE-DATASET analysis.

    Unlike build_context (a single measurement), the input here is a table of many rows,
    so it provides both the full table and the pre-computed aggregates - the model must
    not average the table itself and report the result as a measurement.
    """
    L = []
    L.append("### SCOPE")
    L.append(f"- Root directory: {root or '(unknown)'}")
    L.append(f"- Total .tdms files: {len(df)}  |  analyzed: {len(ok)}  "
             f"|  skipped: {len(skipped)}")
    L.append(f"- Processing parameters, identical for every file: filter {params.get('filter')} "
             f"k={params.get('kernel')}, border crop {params.get('margin')}, "
             f"detrend {params.get('detrend')}, |z| threshold {params.get('z_thresh')}, "
             f"merge gap {params.get('merge')}, minimum area {params.get('min_area')}")

    if len(skipped):
        L.append("\n### SKIPPED FILES (worth attention - may be data faults)")
        for _, r in skipped.iterrows():
            L.append(f"- {r['folder']}/{r['file']}: {r['status']}")

    cut = ok[ok["status"].str.contains("TRUNCATED", na=False)] if "status" in ok else []
    if len(cut):
        L.append("\n### TRUNCATED FILES (analyzable, but data is missing)")
        L.append("Recording of these files was cut short; the incomplete scan rows were "
                 "dropped before analysis, so the scanned area is smaller than declared.")
        for _, r in cut.iterrows():
            L.append(f"- {r['folder']}/{r['file']}: {r['status']}")

    L.append("\n### COLUMN MEANINGS")
    L.append("- z_max / z_top5: the largest z score and the mean of the 5 largest in the "
             "image. z is normalized by each image's own background noise (median/MAD), so "
             "**it compares fairly between sensor R (~0.01 scale) and sensor P (~30 scale)**.")
    L.append("- n_boxes / n_strong: number of candidates, and those with z >= 4.")
    L.append("- n_edge_boxes: candidates touching the image edge - mostly scan artifacts.")
    L.append("- crop_*: width of the noisy border trimmed automatically on each side (px).")
    L.append("- std_raw / dynamic_range: raw amplitude statistics. NOT comparable between "
             "R and P because the units differ (R is amplitude, P is phase).")

    if len(best):
        L.append("\n### BEST CONFIGURATION PER (folder, sensor)")
        L.append("| folder | sensor | amp (V) | freq (kHz) | z_top5 | n_strong |")
        L.append("|---|---|---|---|---|---|")
        for _, r in best.iterrows():
            L.append(f"| {r['folder']} | {r['sensor']} | {r['amp']} | {r['freq']} "
                     f"| {r['z_top5']} | {r['n_strong']} |")

    cols = ["folder", "sensor", "amp", "freq", "variant", "size_px",
            "crop_top", "crop_bottom", "crop_left", "crop_right",
            "n_boxes", "n_strong", "n_edge_boxes", "z_max", "z_top5"]
    cols = [c for c in cols if c in ok.columns]
    L.append(f"\n### FULL TABLE ({len(ok)} analyzed files, sorted by z_top5 descending)")
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "---|" * len(cols))
    tbl = ok.sort_values("z_top5", ascending=False).fillna("")
    for _, r in tbl.iterrows():
        L.append("| " + " | ".join(str(r[c]) for c in cols) + " |")

    L.append("\n### THE QUESTION THAT MATTERS")
    L.append("This is a survey sweep: the same specimen measured while varying sensor, "
             "excitation amplitude and frequency. The goal is to pick the measurement "
             "configuration with the best crack contrast, and to spot faulty measurements.")
    L.append("\nMANDATORY CAVEAT: z_top5 measures DEVIATION from each image's own background, "
             "NOT CNR against real cracks. It answers 'does this image contain standout "
             "structure', NOT 'is that structure actually a crack'. Do not write as though "
             "cracks have been confirmed.")
    return "\n".join(L)


def stream_reply(client, history, on_text, on_thinking=None, image_png=None,
                 context_text=None, question=None, max_tokens=16000):
    """
    Send one turn and stream the reply back through on_text(chunk).

    history : existing messages (the new turn and the reply are appended)
    image_png / context_text : attached only on the first turn of an analysis session
    Returns the full reply text.
    """
    content = []
    if image_png:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(image_png).decode("utf-8"),
            },
        })
    if context_text:
        content.append({"type": "text", "text": "COMPUTED DATA:\n\n" + context_text})
    content.append({"type": "text", "text": question or "Please analyse these results."})

    history.append({"role": "user", "content": content})

    parts = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive", "display": "summarized"},
        messages=history,
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    parts.append(event.delta.text)
                    on_text(event.delta.text)
                elif event.delta.type == "thinking_delta" and on_thinking:
                    on_thinking(event.delta.thinking)
        final = stream.get_final_message()

    if final.stop_reason == "refusal":
        detail = getattr(final, "stop_details", None)
        raise RuntimeError(
            "The model declined to answer this request"
            + (f" (category: {detail.category})" if detail else "") + ".")

    answer = "".join(parts)
    # Append the full content so thinking blocks survive for the next turn on the same model.
    history.append({"role": "assistant", "content": final.content})
    return answer


def friendly_error(e):
    """Turn an SDK error into a sentence that says what to do about it."""
    import anthropic
    if isinstance(e, anthropic.AuthenticationError):
        return ("Invalid API key. Check ANTHROPIC_API_KEY, or re-enter the key "
                "with the 'API key...' button.")
    if isinstance(e, anthropic.PermissionDeniedError):
        return "This API key is not allowed to call that model."
    if isinstance(e, anthropic.NotFoundError):
        return f"Model '{MODEL}' not found. Check what your account has access to."
    if isinstance(e, anthropic.RateLimitError):
        retry = e.response.headers.get("retry-after", "60") if e.response else "60"
        return f"Rate limited. Try again in about {retry} seconds."
    if isinstance(e, anthropic.BadRequestError):
        return f"Invalid request: {e.message}"
    if isinstance(e, anthropic.APIConnectionError):
        return "Could not reach the API. Check your network or proxy."
    if isinstance(e, anthropic.APIStatusError):
        if e.status_code >= 500:
            return f"Server-side error ({e.status_code}). Try again later."
        return f"API error {e.status_code}: {e.message}"
    return str(e)
