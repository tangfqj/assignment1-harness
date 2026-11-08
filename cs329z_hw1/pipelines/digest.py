from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.pipelines.priority import run_priority
from cs329z_hw1.types import Email, LMCallable

RANK = {"urgent": 0, "normal": 1}      # ignore is dropped; unknown -> normal
MAX_BODY_CHARS = 500
MAX_NORMAL_ITEMS = 15

DIGEST_PROMPT = """\
Please write a morning summary for a busy employee from the emails below.
Hard limit (aim well under 300 words); urgent items first;
each urgent item must name the company/project/site it is about;
mention only what is in the emails below; plain text.
"""

def classify_all(emails, lm, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda e: run_priority(e, lm), emails))
    return [(e, r["category"] if r["category"] != "unknown" else "normal")
            for e, r in zip(emails, results)]

def select_items(labeled):
    """Drop ignore, one item per thread, urgent first, then by date."""
    threads = {}                                   # thread_id -> (label, email)
    for e, label in labeled:
        if label == "ignore":
            continue
        cur = threads.get(e["thread_id"])
        # keep the best label; among equals keep the latest email
        if cur is None or RANK[label] < RANK[cur[0]] or (
            RANK[label] == RANK[cur[0]] and e["date"] > cur[1]["date"]):
            threads[e["thread_id"]] = (label, e)
    items = sorted(threads.values(), key=lambda t: (RANK[t[0]], t[1]["date"]))
    urgent = [t for t in items if t[0] == "urgent"]
    normal = [t for t in items if t[0] == "normal"][:MAX_NORMAL_ITEMS]
    return urgent + normal

def render(item) -> str:
    label, e = item
    body = " ".join((e.get("body") or "").split())[:MAX_BODY_CHARS]
    return (f"[{label.upper()}] From: {e.get('from') or '?'}\n"
            f"Subject: {e.get('subject') or '(no subject)'}\n{body}")

def run_daily_digest(emails: list[Email], lm: LMCallable) -> str:
    if not emails:
        return "No emails today."
    items = select_items(classify_all(emails, lm))
    if not items:
        return "Nothing needs your attention today."
    user = "\n\n---\n\n".join(render(i) for i in items)
    try:
        reply = lm([{"role": "system", "content": DIGEST_PROMPT},
                    {"role": "user", "content": user}])
    except OutputTruncated as exc:
        reply = exc.text or ""
    return reply.strip() or "No digest could be generated."