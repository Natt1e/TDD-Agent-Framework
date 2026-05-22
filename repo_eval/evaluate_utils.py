import ast
import base64
import re
import shlex
from textwrap import dedent
import textwrap
from minisweagent.environments.docker import DockerEnvironment

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

def replace_function_in_file(
    environment: DockerEnvironment,
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
    read_result = environment.execute(
        {"command": f"cat {shlex.quote(file_path)}"},
        cwd="/testbed",
    )
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
    encoded_content = base64.b64encode(updated_content.encode("utf-8")).decode("ascii")
    command = (
        f"printf %s {shlex.quote(encoded_content)} | base64 -d > "
        f"{shlex.quote(file_path)}"
    )
    write_result = environment.execute({"command": command}, cwd="/testbed")
    if write_result.get("returncode", 1) != 0:
        raise RuntimeError(
            "Failed to replace function in docker container: "
            f"{write_result.get('output', '')}"
        )
    try :
        ast.parse(updated_content)
        return True
    except :
        return False

