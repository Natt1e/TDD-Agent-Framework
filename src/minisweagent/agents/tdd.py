"""TDD agent implementation."""

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import LimitsExceeded
import logging
from minisweagent import Environment, Model, __version__
from pydantic import BaseModel
from pathlib import Path
from minisweagent.exceptions import FormatError, InterruptAgentFlow
import json

class AgentConfig(BaseModel):

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    output_path: Path | None = None
    """Save the trajectory to this path."""
    execution_limit: int = 5
    """Maximum number of code executions the agent can perform."""
    execution_success_template: str
    """Template for the user message after a successful code execution."""
    execution_failure_template: str
    """Template for the user message after a failed code execution."""
    execution_limit_template: str
    """Template for the user message when the execution limit is exceeded."""
    format_error_limit: int = 10
    tests_submit_limit: str = "off"
    record_every_submit_implementation: str = "off"


import ast
import re
from textwrap import dedent
import textwrap

def get_function_body(
    generated_function : str,
    function_name : str
) :
    generated_function = textwrap.dedent(generated_function).rstrip("\n")
    tree = ast.parse(generated_function)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                return ast.unparse(node.body)

def try_ast_parse_function_body(
    generated_code: str,
    function_name : str
) -> str:
    generated_function = textwrap.dedent(generated_code).rstrip("\n")
    try :
        tree = ast.parse(generated_function)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    return ast.unparse(node.body)
        return None
    except :
        return None
    
def code_equal(a: str, b: str) -> bool:
    def normalize_code(s: str) -> str:
        s = dedent(s)              
        s = re.sub(r'\s+', '', s)  
        return s
    return normalize_code(a) == normalize_code(b)

def get_replaced_file_content(
    original_code: str,
    file_path : str, 
    line_number : int, 
    function_name : str,
    generated_function : str,
    ground_truth: str
) -> str:
    """
    Replace a function in a Python file at the specified line number with a new function.
    
    Args:
        file_path: Python file path
        line_number: Line number of the function definition (1-based)
        new_function: New function object or function source code string
    """
    read_result = original_code
    if read_result.get("returncode", 1) != 0:
        raise RuntimeError(
            "Failed to read target function file in docker container: "
            f"{read_result.get('output', '')}"
        )

    original_content = read_result.get("output", "")
    tree = ast.parse(original_content)
    # Find the target function definition
    target_func = None
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
            if node.name == function_name:
                if node.lineno == line_number:
                    target_func = node
                else :
                    end_line = node.end_lineno
                    got_from_file = original_content.splitlines()[line_number : end_line]                   
                    if code_equal('\n'.join(got_from_file), ground_truth):
                        target_func = node
    if not target_func:
        raise ValueError(f"{file_path} : Line {line_number} : not find the function definition {function_name}")
    
    
    lines = generated_function.splitlines()

    if (try_ast_parse_function_body(generated_function, function_name)) :
        function_body = get_function_body(generated_function, function_name) # no ident
        function_body = textwrap.dedent(function_body).rstrip("\n")
    else :
        # generated_function is only the function body
        function_lines = generated_function.splitlines()
        """
        delete the redundant code
        such as:
        ```
                ...some code...
                return xxx
            def redundant_part() :
                return xxx
        ```
        """
        for line in function_lines :
            if line.strip() :
                stripped_line = line.strip()
                correct_indent = line[:len(line) - len(stripped_line)]
                break
        target_function = []
        for line in function_lines :
            if line.strip():
                stripped_line = line.strip()
                indent = line[:len(line) - len(stripped_line)]
                if indent < correct_indent :
                    break
                else :
                    target_function.append(line)
            
        function_body = textwrap.dedent('\n'.join(target_function)).rstrip("\n")

    body_col = min(n.col_offset for n in target_func.body if hasattr(n, "col_offset"))
    indent_str = " " * body_col
    function_body_indented = textwrap.indent(function_body, indent_str, lambda line: line.strip() != "")

    # replace the original function body
    lines = original_content.splitlines()
    line_number = line_number - 1  # convert to 0-based index
    new_lines = (
        lines[:line_number + 1] + 
        [function_body_indented] + 
        lines[target_func.end_lineno:]
    )
    updated_content = '\n'.join(new_lines)
    ast.parse(updated_content)
    return updated_content



class TDDAgent(DefaultAgent):
    
    def __init__(
        self, 
        model: Model, 
        env: Environment, 
        instance: dict,
        original_code: str,
        *, 
        config_class: type = AgentConfig, 
        **kwargs
    ):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_executions = 0
        self.format_error_count = 0
        self.instance = instance
        self.target_path = str(Path(*instance['metadata']['fpath_tuple'][1:-1]) / Path("test_by_agent.py"))
        self.original_code = original_code
        self.submit_test = ""
        self.submit_code = ""
        self.submit_test_num = 0
        self.artifacts = {
            'code': [],
            'test': []
        }
        
        
    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(
                role="system", 
                content=self._render_template(self.config.system_template)
            ),
            self.model.format_message(
                role="user", 
                content=self._render_template(self.config.instance_template)
            ),
        )
        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                if isinstance(e, FormatError):
                    self.format_error_count += 1
                self.add_messages(*e.messages)
                if self.format_error_count > self.config.format_error_limit:
                    self.add_messages(
                        {
                            "role": "exit",
                            "content": "FormatError limit exceeded.",
                            "extra": {"exit_status": "FormatErrorLimit", "submission": ""},
                        }
                    )
            except Exception as e:
                self.handle_uncaught_exception(e)
                return self.artifacts
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        if len(self.artifacts['code']) == 0 :
            self.artifacts['code'] = [self.submit_code]
        if len(self.artifacts['test']) == 0 :
            self.artifacts['test'] = [self.submit_test]
        return self.artifacts
        
    def execute_pytest(self, results, message, action) :
        if self.n_executions >= self.config.execution_limit:
            self._add_tool_observation(
                results,
                action,
                {
                    "returncode": -1,
                    "output": "run_tests skipped: execution limit reached.",
                    "exception_info": "execution limit reached",
                },
            )
            results.extend(
                self.add_messages(
                    {
                        "role": "user",
                        "content": self._render_template(self.config.execution_limit_template),
                        "extra": {},
                    }
                )
            )
            results.extend(
                self.add_messages(
                    {
                        "role": "exit",
                        "content": "Execution limit reached.",
                        "extra": {},
                    }
                )
            )
            return

        if not (self.submit_test and self.submit_code):
            self._add_tool_observation(
                results,
                action,
                {
                    "returncode": -1,
                    "output": "run_tests skipped: missing submission(s). Please submit both the implementation and the test cases before running run_tests.",
                    "exception_info": "missing submissions",
                },
            )
            return

        output = self.env.execute(
            {
                "tool_name": action["tool_name"],
                "arguments": {},
                "command": f"python -m pytest -q --no-header --tb=short --disable-warnings --maxfail=2 {self.target_path}",
                "tool_call_id": action["tool_call_id"],
            }
        )
        self._add_tool_observation(results, action, output)
        self.n_executions += 1
        if output.get('returncode', 0) != 0 :
            results.extend(
                self.add_messages(
                    {
                        "role": "user",
                        "content": self._render_template(self.config.execution_failure_template),
                        "extra": {},
                    }
                )
            )
        else :
            results.extend(
                self.add_messages(
                    {
                        "role": "user",
                        "content": self._render_template(self.config.execution_success_template),
                        "extra": {},
                    }
                )
            )
        self.artifacts['code'].append(self.submit_code)
        self.artifacts['test'].append(self.submit_test)

    def execute_submit(self, results, message, action) :
        if action.get('tool_name', '') == 'submit_implementation' :
            try :
                new_file_content = get_replaced_file_content(
                    self.original_code,
                    self.target_path,
                    self.instance['metadata']['lineno'],
                    self.instance['metadata']['function_name'],
                    action.get("arguments", {}).get("submission", "EMPTY"),
                    self.instance['metadata']['ground_truth'],
                )
                self.env.write_file(
                    file_path=str(Path(*self.instance['metadata']['fpath_tuple'][1:])), 
                    content=new_file_content,
                    cwd="/testbed"
                )
                results.extend(
                    self.add_messages(
                        {
                            "role": "tool",
                            "content": "Submission received.",
                            "tool_call_id": action.get("tool_call_id"),
                            "extra": {},
                        }
                    )
                )
                self.submit_code = action.get("arguments", {}).get("submission", "EMPTY")
                if self.config.record_every_submit_implementation == "on" :
                    self.artifacts['code'].append(self.submit_code)
            except Exception as e :
                results.extend(
                    self.add_messages(
                        {
                            "role": "tool",
                            "content": "Submission failed. Please submit ONLY the raw implementation of the target function. DO NOT include any imports, test cases, or contextual code.",
                            "tool_call_id": action.get("tool_call_id"),
                            "extra": {},
                        }
                    )
                )
        else :
            if self.config.tests_submit_limit == "on" :
                if self.submit_test_num >= 1 :
                    results.extend(
                        self.add_messages(
                            {
                                "role": "tool",
                                "content": f"Submission failed. You are not allowed to submit tests again. Please call `submit_implementation` and `run_tests` to refine your code based on the execution results.",
                                "tool_call_id": action.get("tool_call_id"),
                                "extra": {},
                            }
                        )
                    )
                    return
            self.env.write_file(
                file_path=self.target_path,
                content=action.get("arguments", {}).get("submission", "EMPTY"),
                cwd="/testbed"
            )
            if self.submit_test_num == 0 :
                self.submit_test_num += 1
                results.extend(
                    self.add_messages(
                        {
                            "role": "tool",
                            "content": f"Submission received. Before you call `run_tests`, make sure your implementation is submitted.",
                            "tool_call_id": action.get("tool_call_id"),
                            "extra": {},
                        }
                    )
                )
            else :
                results.extend(
                    self.add_messages(
                        {
                            "role": "tool",
                            "content": f"Submission received.",
                            "tool_call_id": action.get("tool_call_id"),
                            "extra": {},
                        }
                    )
                )                
            self.submit_test = action.get("arguments", {}).get("submission", "EMPTY")

    def execute_inspect_structure(self, results, message, action):
        file_path = action.get("arguments", {}).get("file_path", "")
        if not file_path:
            results.extend(
                self.add_messages(
                    {
                        "role": "tool",
                        "content": "Please provide the file path to inspect the structure.",
                        "tool_call_id": action.get("tool_call_id"),
                        "extra": {},
                    }
                )
            )
        else:
            try:
                content = self.env.read_file(file_path)
                import ast
                
                try:
                    tree = ast.parse(content)
                except SyntaxError as syntax_err:
                    raise Exception(f"SyntaxError while parsing: {syntax_err}")

                lines = content.splitlines()
                structure = []

                def get_declaration(node):
                    start_idx = node.lineno - 1
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        for i in range(start_idx, len(lines)):
                            stripped = lines[i].lstrip()
                            if stripped.startswith(("class ", "def ", "async def ")):
                                return i + 1, lines[i].rstrip()
                    return node.lineno, lines[start_idx].rstrip()

        
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        structure.append(get_declaration(node))
                        for sub_node in node.body:
                            if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                structure.append(get_declaration(sub_node))
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        structure.append(get_declaration(node))
                    elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        structure.append(get_declaration(node))


                structure.sort(key=lambda x: x[0])
                
                formatted_lines = [f"{line_num}\t{text}" for line_num, text in structure]
                structure_text = "\n".join(formatted_lines)
                
                if not structure_text:
                    structure_text = "No classes, functions, or global variables found."

                results.extend(
                    self.add_messages(
                        {
                            "role": "tool",
                            "content": structure_text,
                            "tool_call_id": action.get("tool_call_id"),
                            "extra": {},
                        }
                    )
                )
                
            except Exception as e:
                results.extend(
                    self.add_messages(
                        {
                            "role": "tool",
                            "content": str(e),
                            "tool_call_id": action.get("tool_call_id"),
                            "extra": {},
                        }
                    )
                )

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        results = []
        for action in message.get("extra", {}).get("actions", []) :
            
            if action.get('tool_name', '') == 'run_tests' :
                self.execute_pytest(results, message, action)
                continue
            if action.get('tool_name', '') in ["submit_implementation", "submit_tests"] :
                self.execute_submit(results, message, action)
                continue
            if action.get('tool_name', '') == 'inspect_structure' :
                self.execute_inspect_structure(results, message, action)
                continue
            # if action.get('tool_name', '') == 'note' :
            #     self._add_tool_observation(
            #         results,
            #         action,
            #         {
            #             "returncode": 0,
            #             "output": "Please continue by calling tools.",
            #             "exception_info": ""
            #         },
            #     )
            #     continue
            output = self.env.execute(action)
            results.extend(
                self.add_messages(
                    *self._format_single_action_observation(action, output)
                )
            )
            
        return results

    def _format_single_action_observation(self, action: dict, output: dict) -> list[dict]:
        message = {"extra": {"actions": [action]}}
        return self.model.format_observation_messages(message, [output], self.get_template_vars())

    def _add_tool_observation(self, results: list[dict], action: dict, output: dict) -> None:
        results.extend(
            self.add_messages(
                *self._format_single_action_observation(action, output)
            ))
