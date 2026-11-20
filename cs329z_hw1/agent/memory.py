from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import count_tokens

STORE_FILE = "memory.json"
TOP_K = 5                    # most facts one recall returns
MEMORY_FRACTION = 0.05       # share of the context budget the prompt's memory block may use

_LOCK = threading.Lock()
_STOPWORDS = frozenset(
    "a an the of to in on at for and or is are was were be do does did what which who how when where "
    "my me i you your it this that with as by from about".split()
)

MEMORY_RULES = """\
Memory: you keep notes that persist across sessions. When the user tells you something about \
themselves that will still matter in later conversations (their team or role, identifiers, preferences, \
what they mean by a term, a correction to something you assumed), call remember with one short, \
self-contained sentence, even if they did not ask you to, and then continue with their request. Do not \
store the contents of documents or emails, results you looked up, or details that matter only for the \
current request. Facts shown below come from memory: use them without searching again. If a request may \
depend on something about the user that you do not see below, call recall before you search elsewhere. \
When you use a stored fact in a search, use its exact words, not an abbreviation. If a stored fact conflicts with \
what the user says now, trust the user and remember the corrected fact. If you say something is remembered, \
make sure to write the tool call. """

def _terms(text: str) -> set[str]:
    """Content words of `text`"""
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokenize(text) if t not in _STOPWORDS}

def _line(fact: dict) -> str:
    return f"- {fact['fact']} (saved {fact.get('saved', '')[:10]})"

class MemoryStore:
    """A list of facts, oldest first, in one JSON file."""

    def __init__(self, directory):
        self.path = Path(directory) / STORE_FILE

    def facts(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return []
        return [f for f in data if isinstance(f, dict) and f.get("fact")] if isinstance(data, list) else []

    def add(self, fact: str) -> str:
        fact = " ".join(str(fact).split())
        if not fact:
            return "Nothing was saved: the fact was empty."
        with _LOCK:
            facts = self.facts()
            if any(f["fact"].lower() == fact.lower() for f in facts):
                return f"Already in memory: {fact}"
            facts.append({"fact": fact, "saved": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(facts, handle, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)  # reader never sees a half-written file
        return f"Saved to memory: {fact}"

    def search(self, query: str) -> list[dict]:
        facts = self.facts()
        docs = [_terms(f["fact"]) for f in facts]
        wanted = _terms(query)

        def idf(term: str) -> float:
            return math.log(1 + len(docs) / sum(term in d for d in docs))

        scored = [(sum(idf(t) for t in wanted & d), i) for i, d in enumerate(docs)]
        scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(key=lambda x : (-x[0], -x[1]))
        return [facts[i] for _, i in scored[:TOP_K]]

    def recall(self, query: str) -> str:
        facts = self.facts()
        if not facts:
            return "Nothing is stored in memory yet."
        hits = self.search(query)
        if hits:
            return "Stored facts that match:\n" + "\n".join(_line(f) for f in hits)
        recents = facts[::-1][:TOP_K]
        return "No stored fact matches the query. The most recent facts are:\n" + "\n".join(_line(f) for f in recents)

    def prompt_block(self, context_budget: int) -> str:
        """The newest facts that fit in MEMORY_FRACTION of the budget"""
        facts = self.facts()
        if not facts:
            return ""
        cap, used, shown = int(context_budget * MEMORY_FRACTION), 0, []
        for f in reversed(facts):
            cost = count_tokens(_line(f)) + 1
            if used + cost > cap:
                break
            shown.append(_line(f))
            used += cost
        text = "What you remember from ealier sessions with this user:\n" + "\n".join(reversed(shown))
        if len(shown) < len(facts):
            text += f"\n({len(facts) - len(shown)} older facts are not shown here; use recall to search them.)"
        return text

def make_memory_fns(memory_dir):
    store = MemoryStore(memory_dir) if memory_dir is not None else None

    def remember(fact: str) -> str:
        return store.add(fact) if store else "Memory is not available in this session."

    def recall(query: str) -> str:
        return store.recall(query) if store else "Memory is not available in this session."

    return remember, recall

