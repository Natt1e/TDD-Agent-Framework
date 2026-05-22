"""Parse TDD tool calls and compile them into command-based actions."""

from __future__ import annotations

import base64
import json
import shlex
from pathlib import PurePosixPath

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError

MAX_READ_LINES = 200


def _tool_schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TDD_TOOLS = [
    _tool_schema(
        "search_command",
        "Execute raw Linux search commands from a specified path (find, grep, xargs, or combinations via pipe '|').",
        {
            "command": {
                "type": "string", 
                "description": (
                    "The exact raw search command to execute (e.g., 'find . -name \"*.py\" | xargs grep -n \"import\" | head -n 50'). "
                    "CRITICAL RULES: "
                    "1. You can ONLY use the following allowed commands: find, grep, egrep, fgrep, xargs, head, tail, cat, less, wc, sort, uniq, awk, cut. "
                    "2. ONLY '|' is allowed for combining commands."
                    "3. Other shell operators like ';', '&&', '||', '<', '>', '<<', '>>', and subshells like '$()' or backticks are STRICTLY PROHIBITED. "
                    "Violating these rules will cause the execution to be rejected."
                )
            },
            "path": {
                "type": "string", 
                "description": (
                    "The path from which to execute the search command. "
                    "If not set, defaults to the root of the codebase ('/testbed')."
                )
            }
        },
        ["command"],
    ),
    # _tool_schema(
    #     "grep",
    #     "Search files recursively for a pattern.",
    #     {
    #         "pattern": {
    #             "type": "string", 
    #             "description": "The pattern to search for. If your pattern uses ANY regular expression syntax (like .*, ^, $), you MUST set 'regex' to true."
    #         },
    #         "regex": {
    #             "type": "boolean",
    #             "description": "If true, treats the pattern as a Regular Expression. If false, treats it as a literal string. Defaults to false."
    #         },
    #         "path": {
    #             "type": "string", 
    #             "description": "The directory path to search in (e.g., 'src/'). Defaults to '.'(/testbed)."
    #         },
    #         "include": {
    #             "type": "string", 
    #             "description": "Optional glob to limit searched files. Do NOT use directory paths here. Defaults to '*.py'."
    #         },
    #         "max_results": {
    #             "type": "integer",
    #             "description": "Maximum number of matching lines to return. Defaults to 100.",
    #         },
    #         "ignore_case": {
    #             "type": "boolean",
    #             "description": "If true, ignores case distinctions. Defaults to false."
    #         }
    #     },
    #     ["pattern"],
    # ),
    _tool_schema(
        "read_files",
        "Read a file or a selected line range from a file.",
        {
            "file_path": {"type": "string", "description": "Path to the file to read."},
            "start_line": {"type": "integer", "description": "Optional 1-based starting line."},
            "end_line": {"type": "integer", "description": "Optional inclusive ending line."},
        },
        ["file_path"],
    ),
    _tool_schema(
        "inspect_structure",
        "Inspect a Python file and return its classes, method signatures, and global variables with line numbers.",
        {
            "file_path": {"type": "string", "description": "Path to the Python file to inspect."},
        },
        ["file_path"],
    ),
    _tool_schema(
        "list_files",
        "List files and directories in a specified directory.",
        {
            "directory": {
                "type": "string", 
                "description": "The directory path to list."
            },
            "recursive": {
                "type": "boolean", 
                "description": "Whether to list files recursively (including subdirectories)."
            },
            "max_results": {
                "type": "integer", 
                "description": "Maximum number of results to return. Defaults to 100."
            },
            "show_hidden": {
                "type": "boolean", 
                "description": "Whether to show hidden files and directories (starting with '.'). Always excludes .git and .venv."
            },
        },
        ["directory"],
    ),
    _tool_schema(
        "run_tests",
        "Run the generated test suite (`test_by_agent.py`). This tool takes NO ARGUMENTS. DO NOT pass any file paths, flags, or parameters.",
        {
        },
        [],
    ),
    _tool_schema(
        "submit_implementation",
        "Submit your implementation.",
        {
            "submission": {
                "type": "string",
                "description": "The raw implementation code to submit.",
            }
        },
        ["submission"],
    ),
    _tool_schema(
        "submit_tests",
        "Submit your tests.",
        {
            "submission": {
                "type": "string",
                "description": "The complete test code to submit.",
            }
        },
        ["submission"],
    ),
    _tool_schema(
        "finish",
        "Finish the task.",
        {
        },
        [],
    ),
]


def _format_error(error: str, format_error_template: str) -> None:
    raise FormatError(
        {
            "role": "user",
            "content": Template(
                format_error_template, 
                undefined=StrictUndefined
            ).render(error=error, actions=[]),
            "extra": {"interrupt_type": "FormatError"},
        }
    )


def _require_object(args: object, tool_name: str) -> dict:
    if not isinstance(args, dict):
        raise ValueError(f"Arguments for '{tool_name}' must be a JSON object.")
    return args


def _require_string(args: dict, key: str, tool_name: str, *, allow_empty: bool = True) -> str:
    value = args.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Tool '{tool_name}' requires string argument '{key}'.")
    if not allow_empty and not value:
        raise ValueError(f"Tool '{tool_name}' requires non-empty argument '{key}'.")
    return value


def _optional_string(args: dict, key: str, tool_name: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Tool '{tool_name}' expects '{key}' to be a string when provided.")
    return value


def _optional_int(args: dict, key: str, tool_name: str, *, default: int | None = None) -> int | None:
    value = args.get(key, default)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"Tool '{tool_name}' expects '{key}' to be an integer when provided.")
    return value


def _optional_bool(args: dict, key: str, tool_name: str, *, default: bool = False) -> bool:
    value = args.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    if not isinstance(value, bool):
        raise ValueError(
            f"Tool '{tool_name}' expects '{key}' to be a boolean when provided."
        )
    return value




def _python_command(script: str) -> str:
    return "python - <<'PY'\n" + script + "\nPY"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _compile_create_file(args: dict) -> str:
    path = _require_string(args, "path", "create_file", allow_empty=False)
    file_text = _require_string(args, "file_text", "create_file")
    qpath = shlex.quote(path)
    qparent = shlex.quote(str(PurePosixPath(path).parent))
    encoded = shlex.quote(_b64(file_text))
    return (
        f"if [ -e {qpath} ]; then printf '%s\\n' {shlex.quote(f'File already exists at: {path}')}; exit 1; fi; "
        f"mkdir -p {qparent} && printf %s {encoded} | base64 -d > {qpath}"
    )


def _compile_append_string(args: dict) -> str:
    path = _require_string(args, "path", "append_string", allow_empty=False)
    new_str = _require_string(args, "new_str", "append_string")
    qpath = shlex.quote(path)
    encoded = shlex.quote(_b64(new_str))
    return (
        f"if [ ! -f {qpath} ]; then printf '%s\\n' {shlex.quote(f'File not found: {path}')}; exit 1; fi; "
        f"printf %s {encoded} | base64 -d >> {qpath}"
    )

import re
def _compile_search_command(args: dict) -> str:
    command = _require_string(args, "command", "search_command", allow_empty=False)
    path = args.get("path", "").strip()
    # ---------------------------------------------------------
    # 第一道防线：无视空格的正则黑名单 (拦截注入核心)
    # 拦截: ;  &  <  >  `  $(  )
    # 允许: |  (管道符作为唯一合法的操作符)
    # ---------------------------------------------------------
    forbidden_pattern = re.compile(r'(;|;|&|<|>|`|\$\(|\))')
    if forbidden_pattern.search(command):
        raise ValueError("Security Error: Dangerous characters detected (e.g., ;, &, >, <, subshells). Only '|' is allowed.")

    # ---------------------------------------------------------
    # 强制在管道符两边加上空格，确保 shlex.split 能正确拆分出 '|'
    # ---------------------------------------------------------
    safe_command = command.replace('|', ' | ')

    try:
        tokens = shlex.split(safe_command)
    except ValueError as e:
        raise ValueError(f"Command syntax error: {e}")

    if not tokens:
        raise ValueError("Tool 'search_command' requires a non-empty command string.")

    # ---------------------------------------------------------
    # 第三道防线：校验白名单 (确保只有基础搜索命令)
    # ---------------------------------------------------------
    allowed_cmds = {
        "find", "grep", "egrep", "fgrep", "xargs",
        "head", "tail", "cat", "less", 
        "wc", "sort", "uniq",
        "awk", "cut" 
    }
    # 提取所有要执行的主命令 (第一个命令，以及所有 '|' 后面的命令)
    cmds_to_execute = [tokens[0]]
    for i, tok in enumerate(tokens):
        if tok == "|":
            if i + 1 >= len(tokens) or tokens[i + 1] == "|":
                raise ValueError("Invalid pipe placement.")
            cmds_to_execute.append(tokens[i + 1])

    for cmd in cmds_to_execute:
        if cmd not in allowed_cmds:
            raise ValueError(f"Security Error: Command '{cmd}' is strictly forbidden. Allowed commands are: {', '.join(allowed_cmds)}.")
    if path:
        command = f"cd {shlex.quote(path)} && {command}"
    return command

def _compile_grep(args: dict) -> str:
    pattern = _require_string(args, "pattern", "grep", allow_empty=False)
    include = _optional_string(args, "include", "grep")
    if not include:
        include = "*.py"
    search_path = _optional_string(args, "path", "grep")
    max_results = _optional_int(args, "max_results", "grep", default=100)

    use_regex = _optional_bool(args, "regex", "grep", default=False)
    ignore_case = _optional_bool(args, "ignore_case", "grep", default=False)

    if not search_path:
        search_path = "."
        
    if max_results is None or max_results <= 0:
        raise ValueError("Tool 'grep' expects 'max_results' to be a positive integer.")
    
    flags = "-"
    flags += "E" if use_regex else "F"
    flags += "i" if ignore_case else ""
    flags += "RIn"

    command_parts = [
        "grep",
        flags, 
        "--binary-files=without-match",
        "--exclude-dir=.git",
        "--exclude-dir=__pycache__",
    ]
    
    if include:
        include = include.split('/')[-1]
        command_parts.append(f"--include={include}")
        
    command_parts.extend(["-e", pattern, search_path])
    
    grep_command = shlex.join(command_parts)
    return (
        "tmp=$(mktemp); "
        f"{grep_command} >\"$tmp\" 2>&1; status=$?; "
        'if [ "$status" -eq 0 ]; then head -n '
        + str(max_results)
        + ' "$tmp"; '
        'elif [ "$status" -eq 1 ]; then printf "%s\\n" "No matches found."; '
        'else cat "$tmp"; rm -f "$tmp"; exit "$status"; fi; '
        'rm -f "$tmp"'
    )




def _compile_read_files(args: dict) -> str:
    file_path = _require_string(args, "file_path", "read_files", allow_empty=False)
    start_line = _optional_int(args, "start_line", "read_files")
    end_line = _optional_int(args, "end_line", "read_files")
    script = f"""
from pathlib import Path

path_str = {file_path!r}
path = Path(path_str)
start_line = {start_line!r}
end_line = {end_line!r}
max_lines = {MAX_READ_LINES}

if not path.exists():
    print(f"The requested file {{path_str}} is not found.")
    raise SystemExit(1)
if path.is_dir():
    print(f"The requested path {{path_str}} is a directory.")
    raise SystemExit(1)
if start_line is not None and start_line < 1:
    print("start_line must be >= 1.")
    raise SystemExit(1)
if end_line is not None and end_line < 1:
    print("end_line must be >= 1.")
    raise SystemExit(1)
if start_line is not None and end_line is not None and end_line < start_line:
    print("end_line must be greater than or equal to start_line.")
    raise SystemExit(1)

text = path.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()
line_count = len(lines)

if start_line is None and end_line is None:
    start_idx = 0
    actual_end = min(line_count, max_lines)
else:
    if start_line is not None and start_line > line_count and line_count != 0:
        print(f"The requested start line {{start_line}} is greater than the number of lines in the file {{line_count}}.")
        raise SystemExit(1)
    if start_line is not None and start_line > 1 and line_count == 0:
        print(f"The requested start line {{start_line}} is greater than the number of lines in the file 0.")
        raise SystemExit(1)
    start_idx = (start_line - 1) if start_line is not None else 0
    requested_end = end_line if end_line is not None else line_count
    actual_end = min(requested_end, start_idx + max_lines, line_count)

selected = lines[start_idx:actual_end]
line_start = start_idx + 1 if line_count or start_idx == 0 else start_idx
line_info = f" lines {{line_start}}-{{actual_end}}" if selected else ""
print(f"```{{path_str}}{{line_info}}")
for lineno, line in enumerate(selected, start=start_idx + 1):
    print(f"{{lineno:>6}}\t{{line}}")
if not selected and line_count == 0:
    print("<empty file>")
if end_line is None and start_line is None and line_count > max_lines:
    print(f"... (truncated at {{max_lines}} lines)")
elif end_line is not None and actual_end < end_line:
    print(f"... (truncated at {{max_lines}} lines)")
elif start_line is not None and end_line is None and actual_end < line_count:
    print(f"... (truncated at {{max_lines}} lines)")
print("```")
""".strip()
    return _python_command(script)




def _compile_list_files(args: dict) -> str:
    directory = _require_string(args, "directory", "list_files")
    recursive = _optional_bool(args, "recursive", "list_files", default=False)
    max_results = _optional_int(args, "max_results", "list_files", default=100)
    show_hidden = _optional_bool(args, "show_hidden", "list_files", default=False)
    if max_results is None or max_results <= 0:
        raise ValueError("Tool 'list_files' expects 'max_results' to be a positive integer.")

    script = f"""
import os
from pathlib import Path

directory = {directory!r}
recursive = {recursive!r}
max_results = {max_results!r}
show_hidden = {show_hidden!r}
ignored_dirs = {{".git", ".venv"}}

target_str = directory or "."
target = Path(target_str)

if not target.exists():
    print(f"Directory not found: {{target_str}}")
    raise SystemExit(1)
if not target.is_dir():
    print(f"Path is not a directory: {{target_str}}")
    raise SystemExit(1)

directories = []
files = []

if recursive:
    for root, dirnames, filenames in os.walk(target, topdown=True):
        dirnames[:] = [
            d for d in dirnames
            if d not in ignored_dirs and (show_hidden or not d.startswith("."))
        ]
        root_path = Path(root)
        rel_root = root_path.relative_to(target)

        for dirname in dirnames:
            rel = (rel_root / dirname).as_posix() if rel_root != Path(".") else dirname
            directories.append(rel)

        for filename in filenames:
            if not show_hidden and filename.startswith("."):
                continue
            rel = (rel_root / filename).as_posix() if rel_root != Path(".") else filename
            files.append(rel)
else:
    for child in target.iterdir():
        name = child.name
        if child.is_dir():
            if name in ignored_dirs:
                continue
            if not show_hidden and name.startswith("."):
                continue
            directories.append(name)
        else:
            if not show_hidden and name.startswith("."):
                continue
            files.append(name)

directories.sort()
files.sort()

total_found = len(directories) + len(files)
limited = total_found > max_results
if limited:
    remaining = max_results
    if len(directories) > remaining:
        directories = directories[:remaining]
        files = []
    else:
        files = files[: remaining - len(directories)]

recursive_label = " (recursive)" if recursive else ""
hidden_label = " (including hidden)" if show_hidden else ""
print(f"Contents of '{{directory or '(root)'}}'{{recursive_label}}{{hidden_label}}")

if not directories and not files:
    print("<empty directory>")
else:
    if directories:
        print("Directories:")
        for item in directories:
            print(f"  [D] {{item}}")
    if files:
        print("Files:")
        for item in files:
            print(f"  [F] {{item}}")

if limited:
    print(f"Note: results limited to {{max_results}} items. Total found: {{total_found}}.")
""".strip()
    return _python_command(script)




def _compile_finish(args: dict) -> str:
    return (
        "printf '%s\\n' COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    )


_COMPILERS = {
    "append_string": _compile_append_string,
    "create_file": _compile_create_file,
    "grep": _compile_grep,
    "read_files": _compile_read_files,
    "list_files": _compile_list_files,
    "finish": _compile_finish,
    "search_command": _compile_search_command,
}


def parse_tdd_toolcall_actions(tool_calls: list, *, format_error_template: str) -> list[dict]:

    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    actions = []
    for tool_call in tool_calls:
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except Exception as e:
            _format_error(f"Error parsing tool call arguments: {e}.", format_error_template)
        tool_name = tool_call.function.name
        compiler = _COMPILERS.get(tool_name)
        if compiler is None:
            if tool_name in {"run_tests", "submit_implementation", "submit_tests", "inspect_structure"} :
                actions.append(
                    {
                        "tool_name": tool_name,
                        "arguments": args,
                        "command": None,
                        "tool_call_id": tool_call.id,
                    }
                )
            else :
                _format_error(f"Unknown tool '{tool_name}'.", format_error_template)
        else :
            try:
                args = _require_object(args, tool_name)
                command = compiler(args)
            except ValueError as e:
                _format_error(str(e), format_error_template)
            actions.append(
                {
                    "tool_name": tool_name,
                    "arguments": args,
                    "command": command,
                    "tool_call_id": tool_call.id,
                }
            )
    return actions
