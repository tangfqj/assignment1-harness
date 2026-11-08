"""Problem 1 (priority): classify one email with a single LM call.
"""

from __future__ import annotations

import re
from datetime import datetime

from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.types import Email, LMCallable

CATEGORIES = ("urgent", "normal", "ignore")

SYSTEM_PROMPT = """\
Here is the definition of three priority labels for emails. Please classify the email I give you as one of these three labels, and explain your reasoning in one sentence.
## urgent

The recipient has to act or decide soon. Two things must both be true:

1. The email asks the recipient to do something, or reports a live problem
   that the recipient is responsible for (an outage, a failed payment, a
   customer or manager escalating an issue).
2. There is time pressure: the email states or clearly implies that the
   action is needed the same day, the next working day, or by a stated
   deadline no more than two working days away.
   A working day is Monday to Friday, counted from the email's date.

Examples:

- "The bank rejected this morning's wire. I need you to approve the
  corrected one by 1:00 PM or we pay a late penalty."
- "The customer's CFO is threatening to cancel. Can you call him before
  4:00 today?"

## normal

Legitimate work or personal mail that was written for this recipient or for
a small working group, with no deadline in the next two working days. This
covers status updates and FYIs, ordinary questions, documents sent for
review without a near deadline, meeting scheduling for later dates, replies
in an ongoing discussion, and personal notes from friends or family.

Examples:

- "Attached is the draft term sheet. Let me know if you have comments
  before our meeting the week after next."
- "Thanks for lunch yesterday. Are you still coming to the lake in
  August?"

## ignore

Mail the recipient can leave unread at no cost: advertising and spam from
vendors, newsletters and news digests, announcements sent to large employee
lists (events, training offerings, facilities notices), and automated
notifications that do not require the recipient to do anything.

Examples:

- "This week only: 40% off all standing desks. Reply REMOVE to
  unsubscribe."
- "To all employees: the annual bake sale is in the lobby on Friday from
  11:00 to 1:00."

## Deciding between two labels

- The wording does not decide the label. An email with "URGENT" in the
  subject that asks nothing of the recipient is not `urgent`. A calm email
  that says "I need your signature by 3:00 PM today" is `urgent`.
- A request with a deadline more than two working days away, or with no
  deadline, is `normal`.
- Mass mailings and advertising are `ignore` even when they use words such
  as "act now" or "important", because nothing is asked of this recipient in
  particular.
- An automated notice that asks this specific recipient to act (for
  example, an expense report waiting for their approval) is `normal`, or
  `urgent` if it states a deadline within two working days.
- A short personal or social message from someone the recipient knows is
  `normal`, not `ignore`.

Reply in exactly this format:
LABEL: <urgent|normal|ignore>
REASON: <one sentence>
"""


def _weekday(date_iso: str) -> str:
    """'2001-06-04T16:56:37Z' -> 'Monday'; '' if the date is malformed."""
    try:
        return datetime.fromisoformat(date_iso.replace("Z", "+00:00")).strftime("%A")
    except (ValueError, AttributeError):
        return ""


def build_messages(email: Email) -> list[dict]:
    """Turn one email into chat messages. Must tolerate blank subject/body,
    empty `to`, malformed addresses. Does not modify `email`."""
    date = email.get("date") or ""
    weekday = _weekday(date)
    to = ", ".join(email.get("to") or []) or "(no recipient listed)"
    user = (
        f"From: {email.get('from') or '(unknown)'}\n"
        f"To: {to}\n"
        f"Date: {date} {('(' + weekday + ')') if weekday else ''}\n"
        f"Subject: {email.get('subject') or '(no subject)'}\n\n"
        f"{email.get('body') or '(empty body)'}"
        # TODO: maybe truncate very long bodies, e.g. body[:6000]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_LABEL_LINE = re.compile(r"^\W*label\W*[:=\-]\s*\W*(urgent|normal|ignore)\b", re.I | re.M)
_REASON_LINE = re.compile(r"^\W*reason\W*[:=\-]\s*(.+)", re.I | re.M)
_ANY_LABEL = re.compile(r"\b(urgent|normal|ignore)\b", re.I)


def parse_reply(text: str) -> tuple[str, str]:
    """Return (category, reason). category is one of CATEGORIES or 'unknown'.
    Never raises; reason is never empty."""
    text = text or ""
    m = _LABEL_LINE.search(text)
    if m:
        category = m.group(1).lower()
    else:
        # Fallback: no formatted line. Accept only if exactly ONE distinct
        # label word appears; none or several -> "unknown" (don't guess).
        found = {w.lower() for w in _ANY_LABEL.findall(text)}
        category = found.pop() if len(found) == 1 else "unknown"

    r = _REASON_LINE.search(text)
    reason = (r.group(1).strip() if r else "") or " ".join(text.split())[:300]
    if not reason:
        reason = "The model gave no reason."
    return category, reason


def run_priority(email: Email, lm: LMCallable) -> dict:
    try:
        reply = lm(build_messages(email))
    except OutputTruncated as exc:  # keep whatever text was produced
        reply = exc.text or ""
    # BudgetExceeded / network errors are deliberately NOT swallowed.
    category, reason = parse_reply(reply)
    return {"category": category, "reason": reason}
