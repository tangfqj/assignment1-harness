from __future__ import annotations

from dataclasses import dataclass
import threading

from cs329z_hw1 import adapters, data
from cs329z_hw1.tokens import count_tokens
from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.types import DocWindow, SearchResult

SNIPPET_MAX_CHARS = 600
WINDOW_MAX_TOKENS = 1500
TOOL_WINDOW_TOKENS = 500
MAX_TOOL_K = 8
MIN_WORDS = 8

@dataclass
class DocIndex:
	docs: dict[str, dict]				# doc_id -> {"doc_id", "title", "text"}
	lines: dict[str, list[str]]			# doc_id -> text.splitlines()
	chunks: list[tuple[str, int, str]]	# (doc_id, first line number, chunk text)
	bm25: object 						# BM25 index over the chunks

def _chunks(doc_id: str, lines: list[str]) -> list[tuple[str, int, str]]:
	"""One chunk per paragraph. A heading takes the paragraph under it, and a chunk
	that is too short (a table header, a one-line item) takes the lines that follow."""
	out, i, n = [], 0, len(lines)
	while i < n:
		line = lines[i]
		if not line.strip():
			i += 1
			continue
		start, text = i + 1, line.strip()
		is_heading = line.lstrip().startswith("#")
		while True:
			j = i + 1
			while j < n and not lines[j].strip():
				j += 1
			if j >= n or lines[j].lstrip().startswith("#"):
				break                                       
			if not is_heading and len(tokenize(text)) >= MIN_WORDS:
				break                                       
			text += "\n" + lines[j].strip()
			i = j
			is_heading = False                             
		out.append((doc_id, start, text))
		i += 1
	return out

def build_doc_index(docs: list[dict]) -> DocIndex:
	by_id = {d["doc_id"]: d for d in docs}
	lines = {d["doc_id"]: d["text"].splitlines() for d in docs}
	chunks = []
	for d in docs:
		chunks.extend(_chunks(d["doc_id"], lines[d["doc_id"]]))
	indexed = [f"{by_id[doc]['title']}\n{text}" for doc, _, text in chunks]
	return DocIndex(by_id, lines, chunks, adapters.run_bm25_build(indexed))

def _snippet(text: str, query: str) -> str:
	"""At most 600 characters of the chunk, centered on the first query word if it is cut"""
	if len(text) <= SNIPPET_MAX_CHARS:
		return text
	low = text.lower()
	hits = [p for p in (low.find(t) for t in set(tokenize(query))) if p >= 0]
	start = max(0, min(hits) - 150) if hits else 0
	if start:
		start = text.rfind(" ", 0, start) + 1       # begin at a word boundary
	piece = text[start:start + SNIPPET_MAX_CHARS]
	if start + SNIPPET_MAX_CHARS < len(text):
		piece = piece.rsplit(" ", 1)[0]             # end at a word boundary
	return piece.strip()

def search_docs(index: DocIndex, query: str, k: int = 5) -> List[SearchResult]:
	if k <= 0:
		return []
	results, seen = [], set()
	for chunk_id, _score in adapters.run_bm25_search(index.bm25, query, k * 6):
		doc_id, line_no, text = index.chunks[chunk_id]
		if doc_id in seen:
			continue
		seen.add(doc_id)
		results.append({"doc_id": doc_id, "title": index.docs[doc_id]['title'],
						"snippet": _snippet(text, query), "line": line_no})
		if len(results) == k:
			break
	return results

def read_doc(index: DocIndex, doc_id: str, start_line: int = 1,
			 max_tokens: int = WINDOW_MAX_TOKENS) -> DocWindow:
	if doc_id not in index.docs:
		raise LookupError(f"No document with id {doc_id!r}.")
	lines = index.lines[doc_id]
	total = len(lines)
	if not 1 <= start_line <= total:
		raise ValueError(f"start_line must be between 1 and {total}, got {start_line}.")
	end = start_line
	while end < total and count_tokens("\n".join(lines[start_line - 1:end + 1])) <= max_tokens:
		end += 1
	return {"doc_id": doc_id, "start_line": start_line, "end_line": end,
			"total_lines": total, "text": "\n".join(lines[start_line - 1:end])}

def format_search(results: list[SearchResult]) -> str:
	if not results:
		return "No documents matched. Try different or fewer keywords."
	parts = []
	for r in results:
		snippet = " ".join(r["snippet"].split())
		parts.append(f"[{r['doc_id']}] {r['title']} (line {r.get('line', 1)})\n {snippet}")
	return "\n\n".join(parts) + "\n\nTo read more of a document, call read_doc with its doc_id and a start line."

def format_window(w: DocWindow) -> str:
	head = f"{w['doc_id']}, lines {w['start_line']}-{w['end_line']} of {w['total_lines']}:\n"
	tail = (f"\n[More: call read_doc again with start_line={w['end_line'] + 1}.]"
			if w["end_line"] < w["total_lines"] else "\n[End of document.]")
	return head + w["text"] + tail

_lock = threading.Lock()
_index: DocIndex | None = None

def get_index() -> DocIndex:
	global _index
	with _lock:
		if _index is None:
			_index = adapters.run_build_doc_index(data.load_docs())
	return _index 
