from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Callable

from cs329z_hw1 import sandbox
from cs329z_hw1.types import AgentConfig

OUTPUT_DIR = ".tool_outputs"      # inside the workspace, so later commands can read it
MAX_TOKENS_PER_RESULT = 1500
RESERVE_TOKENS = 150              # headers, markers, and the "saved to" note
HEAD_FRACTION = 0.6

class TerminalFailure(RuntimeError):
	"""Raised so that the registry reports status tool_error"""

def _head_tail(text: str, max_tokens: int) -> tuple[str, bool]:
	max_chars = max(max_tokens, 1) * 4
	if len(text) <= max_chars:
		return text, False

	head_n = int(max_chars * HEAD_FRACTION)
	tail_n = max_chars - head_n
	head, tail = text[:head_n], text[-tail_n:]
	cut = head.rfind("\n")
	if cut > head_n * 0.5:
		head = head[:cut + 1]
	nl = tail.find("\n")
	if 0 <= nl < tail_n * 0.5:
		tail = tail[nl + 1:]
	omitted_chars = len(text) - len(head) - len(tail)
	omitted_lines = max(text.count("\n") - head.count("\n") - tail.count("\n"), 0)
	marker = f"\n[... {omitted_lines:,} lines ({omitted_chars:,} characters) omitted ...]\n"
	return head + marker + tail, True

def _count_lines(text: str) -> int:
	return text.count("\n") + (1 if text and not text.endswith("\n") else 0)

def make_run_terminal(config: AgentConfig) -> Callable[[str], str]:
	session_id = uuid.uuid4().hex[:6]
	counter = 0
	lock = threading.Lock()
	cap = max(300, min(MAX_TOKENS_PER_RESULT, config.context_budget // 6))
	usable = cap - RESERVE_TOKENS
	out_budget, err_budget = int(usable * 0.8), usable - int(usable * 0.8)

	def save(name: str, text: str) -> str:
		ws = (sandbox.make_workspace(config.workspace) if config.workspace is not None
			  else sandbox.default_workspace())
		folder = Path(ws) / OUTPUT_DIR
		folder.mkdir(exist_ok=True)
		(folder / name).write_text(text, encoding="utf-8")
		return f"{OUTPUT_DIR}/{name}"                  # relative: commands run in the workspace

	def run_terminal_tool(cmd: str) -> str:
		nonlocal counter
		result = sandbox.run_terminal(cmd, workspace=config.workspace, timeout=config.terminal_timeout)

		if result.rejected:
			raise TerminalFailure(
				f"The command was rejected and did not run: {result.reason}\n"
				"Use only the allowed programs, no $(...) or backticks, and no here-documents; "
				"write multi-line code as python3 -c '...'.")

		out, out_cut = _head_tail(result.stdout, out_budget)
		err, err_cut = _head_tail(result.stderr, err_budget)

		if result.timed_out:
			status = f"TIMED OUT: {result.reason}"
		else:
			status = f"exit code: {result.exit_code}"
		parts = [status, "stdout:", out if out.strip() else "(empty)",
				 "stderr:", err if err.strip() else "(empty)"]

		if out_cut or err_cut:
			with lock:
				counter += 1
				n = counter
			notes = []
			if out_cut:
				path = save(f"cmd-{session_id}-{n}.stdout.txt", result.stdout)
				notes.append(f"the full stdout ({_count_lines(result.stdout):,} lines, "
							 f"{len(result.stdout):,} characters) is saved in {path}")
			if err_cut:
				path = save(f"cmd-{session_id}-{n}.stderr.txt", result.stderr)
				notes.append(f"the full stderr is saved in {path}")
			parts.append("[Output truncated: the start and the end are shown above; " + "; ".join(notes)
						 + ". Read the rest from that file with grep, head, tail, sed -n 'A,Bp' or "
						   "python3 instead of printing it again.]")

		hint = ""
		if result.exit_code == 127 or "command not found" in result.stderr:
			hint = ("\n[Hint: that program is not installed in this sandbox. Use python3 instead "
					"(for example python3 -c '...' for JSON or text processing).]")

		shown = "\n".join(parts) + hint
		if result.timed_out or result.exit_code != 0:
			raise TerminalFailure(shown)               # -> status tool_error, content includes `shown`
		return shown

	return run_terminal_tool