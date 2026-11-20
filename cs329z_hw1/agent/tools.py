from __future__ import annotations

import threading

from cs329z_hw1 import adapters, data
from cs329z_hw1.cardinal import CARDINAL_TOOLS, cardinal_spec
from cs329z_hw1.types import AgentConfig, ToolSpec
from cs329z_hw1.agent import docs as docs_mod
from cs329z_hw1.agent.terminal import make_run_terminal
from cs329z_hw1.agent.memory import make_memory_fns

MAX_K = 20
SNIPPET = 160
MAX_EMAIL_CHARS = 6000

class _Archive:
	"""The email archive and its BM25 index."""

	def __init__(self):
		self._lock = threading.Lock()
		self._ready = False

	def load(self):
		with self._lock:
			if not self._ready:
				emails = data.load_emails()
				self.emails = emails
				self.by_id = {e["id"]: e for e in emails}
				self.index = adapters.run_bm25_build([data.email_text(e) for e in emails])
				self._ready = True
		return self

	def search(self, query: str, k: int):
		self.load()
		hits = adapters.run_bm25_search(self.index, query, k)
		return [self.emails[doc_id] for doc_id, _ in hits]

_ARCHIVE = _Archive()	# shared by all sessions

def _one_line(e) -> str:
	body = " ".join((e.get("body") or "").split())[:SNIPPET]
	return f"{e['id']} | {e['date'][:10]} | {e.get('from') or '?'} | {e.get('subject') or '(no subject)'} | {body}"

def build_cardinal_tools(lm, config: AgentConfig) -> list[ToolSpec]:
	aux = config.aux_lm or lm

	def search_emails(query: str, k: int = 5) -> str:
		emails = _ARCHIVE.search(qeury, min(k, MAX_K))
		return "\n".join(_one_line(e) for e in emails) or "No emails matched."

	def read_email(email_id: str) -> str:
		e = _ARCHIVE.load().by_id.get(email_id)
		if e is None:
			return f"No email with id {email_id!r}."
		body = e.get("body") or ""
		cut = "\n[...email truncated...]" if len(body) > MAX_EMAIL_CHARS else ""
		return (f"id: {e['id']}\nfrom: {e.get('from')}\nto: {', '.join(e.get('to') or [])}\n"
				f"date: {e['date']}\nsubject: {e.get('subject')}\n\n{body[:MAX_EMAIL_CHARS]}{cut}")

	def email_qa(question: str) -> str:
		out = adapters.run_email_qa(question, _ARCHIVE.search, aux)
		if out["answer"] is None:
			return "No answer was found in the archive."
		return f"{out['answer']}\nSupport: {', '.join(out['support']) or '(none)'}"

	def daily_digest(date: str) -> str:
		_ARCHIVE.load()
		return adapters.run_daily_digest(data.emails_on(date), aux)

	def search_docs(query: str, k: int = 5) -> str:
		results = docs_mod.search_docs(docs_mod.get_index(), query, min(k, docs_mod.MAX_TOOL_K))
		return docs_mod.format_search(results)

	def read_doc(doc_id: str, start_line: int = 1) -> str:
		window = docs_mod.read_doc(docs_mod.get_index(), doc_id, start_line)
		return docs_mod.format_window(window)

	remember, recall = make_memory_fns(config.memory_dir)

	fns = {
		"search_emails": search_emails, "read_email": read_email,
		"email_qa": email_qa, "daily_digest": daily_digest,
		"search_docs": search_docs, "read_doc": read_doc,
		"run_terminal": make_run_terminal(config),
		"remember": remember, "recall": recall,
		"ask_user": _ask_user_fn(config),
	}
	return [cardinal_spec(name, DESCRIPTIONS[name], fns[name]) for name in CARDINAL_TOOLS]

def _ask_user_fn(config):
	def ask_user(question: str) -> str:
		if config.user is None:
			return "No user is available to answer"
		return str(config.user.ask(question))
	return ask_user

def assemble_tools(lm, tools: list[ToolSpec] | None, config: AgentConfig) -> list[ToolSpec]:
	if tools is None:
		return build_cardinal_tools(lm, config)
	result = list(tools)
	have = {t.name for t in result}
	extra = build_cardinal_tools(lm, config)
	by_name = {t.name: t for t in extra}
	if config.user is not None and "ask_user" not in have:
		result.append(by_name["ask_user"])
	if config.memory_dir is not None:
		for n in ("remember", "recall"):
			if n not in have:
				result.append(by_name[n])
	return result

DESCRIPTIONS = {
	"search_emails": (
		"Keyword search over the email archive. "
		"Returns the top matches, one per line: email_id, date, sender, subject, and a short snippet. "
		"It does NOT return full emails; call read_email with an id to read one. "
		"The search matches exact words, not meaning, so use a few distinctive terms "
		"(names, project names, places, numbers) rather than a full question. "
		"Use it to find emails or to see which email mentions something. "
		"For a question that needs facts from the emails, use email_qa instead. "
		"Arguments: qeury (search words); k (how many results, default 5, up to 20)."
	),
	"read_email": (
		"Returns one email in full: sender, recipients, data, subject, and body. "
		"Use it after search_emails when a snippet is not enough; the id looks like em-01234 and "
		"must come from a search result or another tool, never a guess. "
		"Very long bodies are cut off and marked as truncated."
	),
	"email_qa": (
		"Answers a question from the email archive by searching and reading emails itself, "
		"including questions that need facts from two or more emails. "
		"Returns a short answer and the ids of the emails it is based on, or says that no answer was found. "
		"Use it for a question like 'who leads the X project?'; use search_email when you want to browse or "
		"list emails. It is slower than a search, so do not call it for a lookup you can do with one search. "
		"Pass the full question as a sentence."
	),
	"daily_digest": (
		"Summarizes the emails of one day in about 300 words or less. "
		"Urgent items are named first; advertising and mass mailings are left out. "
		"Use it for requests like 'what happend on June 22?' or 'what needs attention that day?'. "
		"It does NOT answer questions about a specific topic; use search_emails or email_qa for that. "
		"The date must be written YYYY-MM-DD."
	),
	"search_docs": (
		"Keyword search over Cardinal Energy's internal documents: policies (expenses, PTO, parental "
		"leave, remote work, IT security), the employee handbook, benefits, org chart, product and "
		"engineering docs, and company announcements. Returns up to k documents, each with its id, "
		"title, the line where the match is, and a short snippet. A snippet is evidence. If it does not"
		"contain the exact figure, date, or name you need, call read_doc with that doc_id and the line shown, " 
		"orr search again with different words. The search matches exact words, so use a few distinctive "
		"terms. Policies are sometimes changed by a later announcement or memo (the document id "
		"contains its date), so after you find a policy, search again for newer announcements on the "
		"same topic. Use this for company questions, and search_emails for the 2001 email archive."
	),
	"read_doc": (
		"Reads part of one company document, starting at start_line. Returns the lines that fit in a window, "
		"tells you which lines you got and how many the document has, and how to continue. "
		"Use it after search_docs, with the doc_id from a result and the line shown there. "
		"To read the next part, call it again with the start_line it gives you. Read only what you need."
	),
	"run_terminal": (
		"Runs ONE shell command in a sandboxed workspace and returns its exit code, stdout and stderr. "
		"The workspace holds emails.jsonl (the whole email archive, one JSON object per line with the "
		"keys id, thread_id, reply_to, from, to, date, subject, body; dates are UTC) and docs/ (the company "
		"documents as .md files). Use it to count, filter, or aggregate over the data, which the search "
		"tools cannot do. Allowed programs include python3, grep, awk, sed, sort, uniq, wc, cut, tr, "
		"head, tail, find, ls and cat. Some listed programs may not be installed on this machine (a missing one "
		"reports 'command not found'); do not rely on jq, use python3 for JSON. Example: "
		"python3 -c \"import json; rows=[json.loads(l) for l in open('emails.jsonl')]; print(len(rows))\". "
	),
	"remember": (
		"Saves one fact to your long-term memory. It persists across sessions and is shown to you at the "
		"start of later sessions. Use it when the user tells you something durable about themselves or "
		"how they want things done: their team or role, identifiers such as a cost center, preferences, "
		"what they mean by a term, or a correction. Write one short sentence that makes sense on its own "
		"and names the user, for example: The user is on the firmware team; by \"the spec\" they mean the "
		"firmware spec. Do not store the contents of documents or emails, lookup results, or details that "
		"matter only for the current request. "
		"Argument: fact (the sentence to store)."
	),
	"recall": (
		"Searches your long-term memory for facts saved in earlier sessions or earlier in this one. "
		"It matches words, so use a few key words such as 'cost center' or 'team'. The memory shown at the "
		"start of the session can leave out older facts, so use recall when a request may depend on "
		"something about the user that you do not see there. If nothing matches, it returns the most "
		"recent facts instead. "
		"Argument: query (key words to search for)."
	),
	"ask_user": "Not implemented yet."
}
