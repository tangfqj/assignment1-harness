from __future__ import annotations

import json

from cs329z_hw1.types import ParsedResponse, ToolCall

OPEN = "<tool_call>"
CLOSE = "</tool_call>"

def _escape(s: str) -> str:
    """Make sure OPEN/CLOSE can never appear literally inside the JSON body."""
    return s.replace("<", "\\u003c")

def run_format_tool_call(text: str, call: ToolCall) -> str:
    body = _escape(json.dumps({"name": call.name, "args": call.args}))
    return (text.strip() + "\n\n" if text.strip() else "") + f"{OPEN}{body}{CLOSE}"

def run_parse_response(text: str) -> ParsedResponse:
    try:
        text = text if isinstance(text, str) else ""
        n_open = text.count(OPEN)
        if n_open == 0:
            return ParsedResponse(text=text.strip())
        if n_open > 1:
            return ParsedResponse(
                text=text.strip(), 
                error="Your reply contained more than one tool call. Write exactly one call per reply.")
        start = text.index(OPEN)
        end = text.find(CLOSE, start + len(OPEN))
        if end == -1:
            return ParsedResponse(
                text=text[:start].strip(), 
                error=f"Your tool call was cut off (no closing {CLOSE}). Write the whole call again.")
        body = text[start + len(OPEN):end]
        remaining = (text[:start] + text[end + len(CLOSE):]).strip()
        try:
            obj = json.loads(body)
        except ValueError:
            return ParsedResponse(
                text=remaining, 
                error=f"The text inside {OPEN}...{CLOSE} is not a valid JSON. "
                    'It must be {"name": "...", "args": {...}}.')
        if not isinstance(obj, dict):
            return ParsedResponse(
                    text=remaining, 
                    error='The tool call must be a JSON object with "name" and "args".')
        name, args = obj.get("name"), obj.get("args", {})
        if not isinstance(name, str) or not name.strip():
            return ParsedResponse(text=remaining, error='The tool call needs a "name" that is a non-empty string.')
        if not isinstance(args, dict):
            return ParsedResponse(text=remaining, error='"args" must be a JSON object.')
        return ParsedResponse(text=remaining, tool_call=ToolCall(name, args))
    except Exception as exc:
        return ParsedResponse(text="", error=f"Could not parse the reply: {exc}")

def _example_call(schema: dict) -> str:
    props = (schema.get("parameters") or {}).get("properties", {}) or {}
    required = (schema.get("parameters") or {}).get("required", []) or []
    placeholder = {"string": "example", "integer": 3, "number": 1.5, "boolean": True}
    args = {p: placeholder.get((props.get(p) or {}).get("type"), "example") for p in required}
    return run_format_tool_call("I'll look that up.", ToolCall(schema["name"], args))

def _param_lines(parameters: dict) -> list[str]:
    props = (parameters or {}).get("properties", {}) or {}
    required = set((parameters or {}).get("required", []) or [])
    lines = []
    for pname, spec in props.items():
        spec = spec if isinstance(spec, dict) else {}
        bits = [str(spec.get("type", "any")), "required" if pname in required else "optional"]
        if "enum" in spec:
            bits.append("one of " + ", ".join(json.dumps(v) for v in spec["enum"]))
        if "default" in spec:
            bits.append(f"default {json.dumps(spec['default'])}")
        if "minimum" in spec:
            bits.append(f"minimum {spec['minimum']}")
        if "pattern" in spec:
            bits.append(f"format {spec['pattern']}")
        desc = f": {spec['description']}" if spec.get("description") else ""
        lines.append(f"  - {pname} ({', '.join(bits)}){desc}")
    return lines or ["  (no arguments)"]

def run_render_tools(schemas: list[dict]) -> str:
    if not schemas:
        return "You have no tools. Answer in plain text."
    lines = [
        "You can use the tools below. To call a tool, write one tool call in exactly this format:",
        "",
        f'{OPEN}{{"name": "<tool name>", "args": {{"<argument>": <value>}}}}{CLOSE}',
        "",
        "Rules:",
        "- Write at most ONE tool call per reply, and put it at the END of the reply. "
        "After the call, stop writing: do not guess the result.",
        "- The call must be valid JSON. Use the exact tool and argument names listed below, "
        "and give every required argument. Strings go in quotes, numbers and true/false do not.",
        "- The result of your call will be sent to you in the next message. "
        "Then you can call another tool or give your final answer.",
        f"- Give your final answer as plain text with no {OPEN}. "
        "Do not call a tool if you can answer without one.",
        "",
        "Tools:",
        "",
    ]

    for s in schemas:
        lines.append(f"### {s['name']}")
        lines.append(s.get("description", ""))
        lines.append("Arguments:")
        lines.extend(_param_lines(s.get("parameters", {})))
        lines.append("")
    lines += ["Example of a reply that calls a tool:", "", _example_call(schemas[0])]
    return "\n".join(lines)