from __future__ import annotations

from jsonschema import Draft202012Validator

from cs329z_hw1.types import ToolResult, ToolSpec

class ToolRegistry:
	def __init__(self, tools: list[ToolSpec]):
		self._tools: dict[str, ToolSpec] = {}
		self._order: list[str] = []
		for t in tools:
			if t.name not in self._tools:
				self._order.append(t.name)
			self._tools[t.name] = t

	def schemas(self) -> list[dict]:
		return [
			{"name": n, "description": self._tools[n].description,
			 "parameters": self._tools[n].parameters}
			for n in self._order 
		]

	def validate(self, name: str, args: dict) -> ToolResult:
		try:
			tool = self._tools.get(name)
			if tool is None:
				known = ", ".join(self._order) or "(none)"
				return ToolResult("unknown_tool", f"No tool named {name!r}. Available tools: {known}.")
			if not isinstance(args, dict):
				return ToolResult("invalid_args", f"Argument for {name!r} must be a JSON object.")
			errors = sorted(Draft202012Validator(tool.parameters).iter_errors(args),
                            key=lambda e: list(e.path))
			if errors:
				msgs = "; ".join(
					(f"{'/'.join(map(str, e.path))}: " if e.path else "") + e.message
                    for e in errors[:5])
				return ToolResult("invalid_args", f"Invalid arguments for {name!r}: {msgs}")
			return None
		except Exception as exc:
			return ToolResult("invalid_args", f"Could not validate arguments for {name!r}: {exc}")

	def execute(self, name: str, args: dict) -> ToolResult:
		bad = self.validate(name, args)
		if bad is not None:
			return bad
		try:
			return ToolResult("ok", str(self._tools[name].fn(**args)))
		except Exception as exc:
			return ToolResult("tool_error", f"Tool {name!r} failed: {type(exc).__name__}: {exc}")
				