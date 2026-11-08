from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass

from cs329z_hw1.tokenizer import tokenize

K1 = 1.5
B = 0.75

@dataclass
class BM25Index:
    postings: dict[str, list[tuple[int, int]]]  # token -> list of (doc_id, term_freq)
    norm: list[float]                           # per doc: K1 * (1 - B + B * len(doc) / avgdl)
    idf: dict[str, float]                       # token -> Lucene IDF
    n_docs: int

def build_index(docs: list[str]) -> BM25Index:
    postings: dict[str, list[tuple[int, int]]] = {}
    lengths: list[int] = []

    for doc_id, text in enumerate(docs):
        tokens = tokenize(text)
        lengths.append(len(tokens))
        for term, tf in Counter(tokens).items():
            postings.setdefault(term, []).append((doc_id, tf))

    n = len(docs)
    avgdl = (sum(lengths) / n) if n else 0.0
    if avgdl == 0.0:
        avgdl = 1.0

    norm = [K1 * (1 - B + B * dl / avgdl) for dl in lengths]
    idf = {t: math.log((n - len(p) + 0.5) / (len(p) + 0.5) + 1) for t, p in postings.items()}
    return BM25Index(postings, norm, idf, n)

def search(index: BM25Index, query: str, k: int) -> list[tuple[int, float]]:
    if k <= 0:
        return []

    scores: dict[int, float] = {}
    for term in set(tokenize(query)):
        plist = index.postings.get(term)
        if plist is None:
            continue
        idf = index.idf[term]
        norm = index.norm
        for doc_id, tf in plist:
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (K1 + 1) / (tf + norm[doc_id])

    top = heapq.nsmallest(k, scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(int(d), float(s)) for d, s in top]