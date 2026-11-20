from __future__ import annotations

import json
from dataclasses import dataclass, field
import re
from typing import Any

from attr import dataclass
from cs329z_hw1.agent.protocol import run_parse_response, run_render_tools
from cs329z_hw1.agent.registry import ToolRegistry
from cs329z_hw1.agent.terminal import _head_tail
from cs329z_hw1.agent.memory import MEMORY_RULES, MemoryStore
from cs329z_hw1.llm import OutputTruncated, BudgetExceeded
from cs329z_hw1.tokens import count_message_tokens, count_tokens
from cs329z_hw1.types import AgentConfig, AgentResult, ToolSpec

BASE_SYSTEM_PROMPT = """\
You are the Cardinal Energy assistant. You help employees by answering their questions, \
using the tools below when you need information you do not already have.

Work step by step: call a tool when it would help, read its result, and continue. \
When you have enough information, reply with your final answer in plain text, with no tool call. \
Base your answer on what the tools returned, and say so if you could not find something.

For any question about Cardinal Energy, its people, products, policies, or documents, or about the email \
archive, you must look the answer up with your tools before you answer. Never say that you do not have \
enough information, and do not ask the user to clarify, before you have searched: most questions can be \
answered by searching with the key words of the question. If a search finds nothing, try different words \
before you give up."""

MALFORMED_LIMIT = 3
MALFORMED_STOP = ("unknown_tool", "invalid_args")

_INTENTION = re.compile(
	r"\b(i'll|i will|let me|i'm going to|i am going to|i'm gonna|allow me to|i need to|i should|i can start|first,? i)\b",
	re.I,
)
MAX_NUDGES = 3
NUDGE = (
	"You said you would take an action but did not write a tool call. Your tools are available. "
	"Write the tool call now, in this reply, or give your final answer in plain text."
)

# context compaction
TARGET_FRACTION = 0.6		# a compaction shrinks the context to about this fraction of budget
KEEP_RECENT_RESULTS = 2 	
MIN_STUB_TOKENS = 80		# a result shorter than this is not worth replacing
SUMMARY_FRACTION = 0.125	# the summary may use this fraction of budget
PINNED_FRACTION = 0.2       # the user's earlier messages, word for word, may use this fraction

SUMMARY_SYSTEM = (
	"You maintain the memory of a conversation between a user and an AI assistant that uses tools. "
	"You are given the previous summary (it may be empty) and newer messages. Write the updated summary, "
	"which replaces the previous one. Keep, word for word: every constraint, requirement, preference, "
	"number, name, date, id and file path the user stated; the decisions made; what has been done and the "
	"results that matter; and what is still open. Leave out small talk and bulky raw data. "
	"Plain text only, no preamble, at most {words} words."
)

@dataclass
class Entry:
	"""One message of the conversation"""

	role: str 				# "user" | "assistant"
	content: str 			# the full original text
	kind: str 				# "user" | "assistant" | "result" | "note"
	tool: str = "" 		  	# for kind == "result": the tool name
	args: dict = field(default_factory=dict)
	stub: str | None = None # sent instead of `content` once the message has been shortened

	@property
	def text(self) -> str:
		return self.content if self.stub is None else self.stub

class AgentSession:
	def __init__(self, lm: Any, tools: list[ToolSpec], config: AgentConfig):
		self.lm = lm
		self.config = config
		self.aux_lm = config.aux_lm or lm
		self.registry = ToolRegistry(tools)
		self.mode = config.mode

		self.system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + run_render_tools(self.registry.schemas())
		if config.memory_dir is not None:
			self.system_prompt += "\n\n" + MEMORY_RULES
			block = MemoryStore(config.memory_dir).prompt_block(config.context_budget)
			if block:
				self.system_prompt += "\n\n" + block
		self.transcript: list[dict] = [] 	# append-only

		# what the model sees is built from:
		self.entries: list[Entry] = []
		self.covered = 0 				# entries[:covered] are replaced by summary + pinned
		self.turn_start = 0 			# index of current user message
		self.summary = ""			
		self.pinned: list[str] = []		# ealier user messages, word for word

	def set_mode(self, mode: str) -> None:
		self.mode = mode

	def send(self, user_message: str) -> AgentResult:
		self._log("user", content=user_message)
		self.turn_start = len(self.entries)
		self.entries.append(Entry("user", user_message, "user"))

		calls = 0		# model calls made in THIS send
		tokens = 0		# tokens used in THIS send
		malformed = 0	# consecutive malformed replies
		nudges = 0 	 	# nudges sent in THIS send

		while True:
			# Check limits BEFORE each model call
			if calls >= self.config.max_turns:
				return self._finish(
						"max_turns", 
						"I ran out of steps before I could finish. Ask me to continue and I will pick up from here.")
			if tokens >= self.config.token_budget:
				return self._finish(
						"token_budget",
						"I used up my budget for this request before I could finish. Ask me to continue, or narrow down the question.")

			# Ask the model
			sent = self._prepare_context()	# where compaction happens
			try:
				reply = self.lm(sent)
				truncated = False
			except OutputTruncated as exc:
				reply, truncated = exc.text, True
			calls += 1
			tokens += count_message_tokens(sent) + count_tokens(reply)

			# Truncated or unparsable -> Malformed
			parsed = None if truncated else run_parse_response(reply)
			if truncated or parsed.error:
				message = (
					"Your reply was cut off because it was too long. Write a shorter reply: "
					"if you are calling a tool, write only the call."
					if truncated else parsed.error
				)
				self._log("assistant", content=reply)
				self._log("error", content=message)
				if reply.strip():
					self.entries.append(Entry("assistant", reply, "assistant"))
				self.entries.append(Entry("user", message, "note"))
				malformed += 1
				if malformed >= MALFORMED_LIMIT:
					return self._finish(
							"malformed",
							"I could not produce a valid tool call after several tries, so I stopped. "
							"Please rephrase your request or try again.")
				continue

			if parsed.tool_call is None and not parsed.text.strip():
				message = ("Your reply was empty. Write the tool call you need now, "
						   "or give your final answer in plain text.")
				self._log("assistant", content=reply)
				self._log("error", content=message)
				self.entries.append(Entry("user", message, "note"))
				malformed += 1
				if malformed >= MALFORMED_LIMIT:
					return self._finish(
						"malformed",
						"I could not produce a valid tool call after several tries, so I stopped. "
						"Please rephrase your request or try again.")
				continue
			
			self._log("assistant", content=reply)
			self.entries.append(Entry("assistant", reply, "assistant"))
			malformed = 0 if parsed.tool_call is None else malformed

			# No call
			nudges = 0
			if parsed.tool_call is None:
				if nudges < MAX_NUDGES and self._announces_action(parsed.text):
					nudges += 1
					self._log("nudge", content=NUDGE)
					self.entries.append(Entry("user", NUDGE, "note"))
					continue
				return self._finish("done", parsed.text or "I do not have an answer to give.")

			# A tool call
			name, args = parsed.tool_call.name, parsed.tool_call.args
			self._log("tool_call", name=name, args=args)

			result = self.registry.execute(name, args)
			shown = self._cap_result(result.content)
			self._log("tool_result", name=name, status=result.status, content=shown)
			self.entries.append(Entry("user", self._format_result(name, result.status, shown),
										"result", tool=name, args=args))

			malformed = malformed + 1 if result.status in MALFORMED_STOP else 0
			if malformed >= MALFORMED_LIMIT:
				return self._finish(
						"malformed",
						"I could not produce a valid tool call after several tries, so I stopped. "
						"Please rephrase your request or try again.")


	def _view(self) -> list[dict]:
		view = [{"role": "system", "content": self.system_prompt}]
		block = self._context_block()
		if block:
			view.append({"role": "user", "content": block})
		view.extend({"role": e.role, "content": e.text} for e in self.entries[self.covered:])
		return view

	def _size(self) -> int:
		return count_message_tokens(self._view())

	def _prepare_context(self) -> list[dict]:
		budget = self.config.context_budget
		if self._size() <= budget:
			return self._view()
		target = int(budget * TARGET_FRACTION)

		n_stubbed = self._stub_results(target)
		summarized = False
		if self._size() > budget:
			summarized = self._summarize_older()
		guarded = False
		if self._size() > budget:
			self._force_fit()
			guarded = True
		self._log("compaction", content=(
			f"stubbed {n_stubbed} tool results"
			+ ("; summarized earlier turns" if summarized else "")
			+ ("; truncated items to fit" if guarded else "")))
		return self._view()

	def _stubbable(self, i: int) -> bool:
		e = self.entries[i]
		return e.kind == "result" and e.stub is None

	def _stub_for(self, e: Entry) -> str:
		args = json.dumps(e.args, ensure_ascii=False)
		if len(args) > 80:
			args = args[:77] + "..."
		saved = sorted(set(re.findall(r"\.tool_outputs/[\w.\-]+", e.content)))
		where = (f" The full output is saved in {', '.join(saved)}." if saved
				 else " Call the tool again if you need it.")
		return f"[Result of {e.tool}({args}) removed to save space; it had {count_tokens(e.content):,} tokens.{where}]"

	def _shrink_block(self) -> None:
		if len(self.summary) > 160:
			self.summary = self.summary[: len(self.summary) // 2]
		elif len(self.pinned) > 2:
			del self.pinned[1]
		else:
			self.summary = ""
			self.pinned = [m[: max(40, len(m) // 2)] for m in self.pinned]

	def _cap_result(self, content: str) -> str:
		"""No single tool result may use more than a third of the context budget."""
		cap = max(400, self.config.context_budget // 3)
		shown, _ = _head_tail(content, cap)
		return shown
	
	def _stub_results(self, target: int) -> int:
		"""Replace old tool results by short stubs, oldest first, until the context is down to `target`."""
		older = [i for i in range(self.covered, self.turn_start) if self._stubbable(i)]

	def _summarize_older(self) -> bool:
		"""Fold every message before the current request into the summary.
		The current user message and what follows it stay as they are."""
		cut = self.turn_start
		if cut <= self.covered:
			return False
		chunk = self.entries[self.covered:cut]
		for e in chunk:
			if e.kind == "user":
				self._pin(e.content)
		cap_tokens = max(120, int(self.config.context_budget * SUMMARY_FRACTION))
		self.summary = self._fold(self.summary, [self._line_for_summary(e) for e in chunk], cap_tokens)
		self.covered = cut
		return True

	def _pin(self, text: str) -> None:
		cap = int(self.config.context_budget * PINNED_FRACTION)
		self.pinned.append(" ".join(text.split())[:cap])
		while len(self.pinned) > 2 and sum(count_tokens(m) + 1 for m in self.pinned) > cap:
			del self.pinned[1]

	@staticmethod
	def _line_for_summary(e: Entry) -> str:
		if e.kind == "user":
			return "User: " + e.content[:1500]
		if e.kind == "assistant":
			return "Assistant: " + e.content[:600]
		if e.kind == "result":
			return f"Tool result ({e.tool}): " + e.content[:800]
		return "Note: " + e.content[:300]

	def _fold(self, previous: str, lines: list[str], cap_tokens: int) -> str:
		"""One aux_lm call: merge `lines` into `previous`. Falls back to a plain extract."""
		prompt = [
			{"role": "system", "content": SUMMARY_SYSTEM.format(words=int(cap_tokens * 0.7))},
			{"role": "user", "content": f"Previous summary:\n{previous or '(none yet)'}\n\n"
										"New messages to fold in:\n" + "\n".join(lines)},
		]
		try:
			out = self.aux_lm(prompt)
		except OutputTruncated as exc:
			out = exc.text or ""
		except BudgetExceeded:
			raise
		except Exception:
			out = ""
		out = (out or "").strip()
		if not out:
			kept = [l[:150] for l in lines if l.startswith("User:")]
			out = ((previous + "\n") if previous else "") + "\n".join(kept)
		return out[: cap_tokens * 4]
	
	def _context_block(self) -> str:
		if not self.summary and not self.pinned:
			return ""
		parts = ["[Earlier parts of this conversation were compacted to save space.]"]
		if self.summary:
			parts.append("Summary of the earlier conversation:\n" + self.summary)
		if self.pinned:
			parts.append("The user's earlier messages, word for word (oldest first):\n"
						+ "\n".join(f"- {m}" for m in self.pinned))
		return "\n\n".join(parts)

	def _force_fit(self) -> None:
		"""Last resort: halve the largest item (never the current user message) until it fits."""
		budget = self.config.context_budget
		for _ in range(300):
			if self._size() <= budget:
				return
			best, best_size = None, 40     # items under 40 tokens are not worth shrinking
			for i in range(self.covered, len(self.entries)):
				if i == self.turn_start:
					continue
				n = count_tokens(self.entries[i].text)
				if n > best_size:
					best, best_size = i, n
			block_size = count_tokens(self._context_block())
			if block_size > best_size:
				self._shrink_block()
			elif best is not None:
				e = self.entries[best]
				e.stub, _ = _head_tail(e.text, best_size // 2)
			else:
				return

	@staticmethod
	def _announces_action(text: str) -> bool:
		text = (text or "").strip().replace("\u2019", "'").replace("\u2018", "'")   # curly apostrophes
		return 0 < len(text) <= 400 and bool(_INTENTION.search(text))

	@staticmethod
	def _format_result(name: str, status: str, content: str) -> str:
		return f"Tool result from {name} (status: {status}): \n{content}"

	def _log(self, type_: str, **fields) -> None:
		self.transcript.append({"type": type_, **fields})

	def _finish(self, status: str, text: str) -> AgentResult:
		text = text.strip() or "The agent finished with no result"
		return AgentResult(text=text, status=status, transcript=[dict(e) for e in self.transcript])
