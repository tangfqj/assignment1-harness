from __future__ import annotations

import re
from typing import Any

from cs329z_hw1.agent.protocol import run_parse_response, run_render_tools
from cs329z_hw1.agent.registry import ToolRegistry
from cs329z_hw1.llm import OutputTruncated
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

class AgentSession:
	def __init__(self, lm: Any, tools: list[ToolSpec], config: AgentConfig):
		self.lm = lm
		self.config = config
		self.aux_lm = config.aux_lm or lm
		self.registry = ToolRegistry(tools)
		self.mode = config.mode

		self.system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + run_render_tools(self.registry.schemas())
		self.messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
		self.transcript: list[dict] = [] 	# append-only

	def set_mode(self, mode: str) -> None:
		self.mode = mode

	def send(self, user_message: str) -> AgentResult:
		self._log("user", content=user_message)
		self.messages.append({"role": "user", "content": user_message})

		calls = 0		# model calls made in THIS send
		tokens = 0		# tokens used in THIS send
		malformed = 0	# consecutive malformed replies
		nudged = False 	# at most one nudge per send

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
			sent = list(self.messages)
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
					self.messages.append({"role": "assistant", "content": reply})
				self.messages.append({"role": "user", "content": message})
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
				self.messages.append({"role": "user", "content": message})
				malformed += 1
				if malformed >= MALFORMED_LIMIT:
					return self._finish(
						"malformed",
						"I could not produce a valid tool call after several tries, so I stopped. "
						"Please rephrase your request or try again.")
				continue
			
			self._log("assistant", content=reply)
			self.messages.append({"role": "assistant", "content": reply})
			malformed = 0 if parsed.tool_call is None else malformed

			# No call
			nudges = 0
			if parsed.tool_call is None:
				if nudges < MAX_NUDGES and self._announces_action(parsed.text):
					nudges += 1
					self._log("nudge", content=NUDGE)
					self.messages.append({"role": "user", "content": NUDGE})
					continue
				return self._finish("done", parsed.text or "I do not have an answer to give.")

			# A tool call
			name, args = parsed.tool_call.name, parsed.tool_call.args
			self._log("tool_call", name=name, args=args)

			result = self.registry.execute(name, args)
			self._log("tool_result", name=name, status=result.status, content=result.content)
			self.messages.append({"role": "user", "content": self._format_result(name, result)})

			malformed = malformed + 1 if result.status in MALFORMED_STOP else 0
			if malformed >= MALFORMED_LIMIT:
				return self._finish(
						"malformed",
						"I could not produce a valid tool call after several tries, so I stopped. "
						"Please rephrase your request or try again.")

	@staticmethod
	def _announces_action(text: str) -> bool:
		text = (text or "").strip().replace("\u2019", "'").replace("\u2018", "'")   # curly apostrophes
		return 0 < len(text) <= 400 and bool(_INTENTION.search(text))

	def _format_result(self, name: str, result) -> str:
		return f"Tool result form {name} (status: {result.status}): \n{result.content}"

	def _log(self, type_: str, **fields) -> None:
		self.transcript.append({"type": type_, **fields})

	def _finish(self, status: str, text: str) -> AgentResult:
		text = text.strip() or "The agent finished with no result"
		return AgentResult(text=text, status=status, transcript=[dict(e) for e in self.transcript])
