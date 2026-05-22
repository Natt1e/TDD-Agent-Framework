import ast
import base64
import json
from pathlib import Path
import shlex
import os
import yaml
import litellm
from minisweagent.environments.docker import DockerEnvironment

def extract_final_submission(traj_path: str, model_cfg: dict = None) -> str:
    with open(traj_path, 'r', encoding='utf-8') as f:
        traj_data = json.load(f)
    
    if traj_data["info"]["exit_status"] == "Submitted":
        
        messages = traj_data.get("messages", [])
        cleaned_messages = []
        for msg in messages:
            clean_msg = {"role": msg.get("role")}
            if "content" in msg and msg["content"] is not None:
                clean_msg["content"] = msg["content"]
            if "tool_calls" in msg and msg["tool_calls"]:
                clean_msg["tool_calls"] = []
                for tc in msg["tool_calls"]:
                    clean_tc = {
                        "id": tc.get("id"),
                        "type": tc.get("type", "function"),
                        "function": {
                            "name": tc.get("function", {}).get("name"),
                            "arguments": tc.get("function", {}).get("arguments")
                        }
                    }
                    clean_msg["tool_calls"].append(clean_tc)
            if "tool_call_id" in msg:
                clean_msg["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                clean_msg["name"] = msg["name"]
            cleaned_messages.append(clean_msg)

        system_prompt = "Your task is to extract the final implementation code from the following conversation trajectory. Only output the code block containing the final implementation. Do not output anything else. Output as follows:\n```python\n<only the final implementation code>\n```"
        user_prompt = f"Trajectory:\n{json.dumps(cleaned_messages, indent=2, ensure_ascii=False)}"
        
        if model_cfg is None:
            config_file = Path("mini-swe-agent-repo-eval/config/repoeval_mini_swe_agent.yaml")
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f.read())
                model_cfg = cfg.get("model", {})
            else:
                model_cfg = {}
                
        model_name = model_cfg.get("model_name", "openai/Qwen3-Coder-30B-A3B-Instruct")
        model_kwargs = model_cfg.get("model_kwargs", {})
        api_base = model_kwargs.get("base_url") or model_kwargs.get("api_base")
        api_key = model_kwargs.get("api_key")
        
        try:
            response = litellm.completion(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                api_base=api_base,
                api_key=api_key,
                temperature=0.0
            )
            content = response.choices[0].message.content
            return content.strip()
        except Exception as e:
            print(f"Failed to extract final submission: {e}")
            return ""

def token_count(traj: str) -> dict:
    with open(traj, 'r', encoding='utf-8') as f:
        traj = json.load(f)
    messages = traj.get("messages", [])
    completion_tokens = 0
    prompt_tokens = 0
    cached_tokens = 0
    for message in messages:
        if "extra" in message and "response" in message["extra"] and "usage" in message["extra"]["response"]:
            usage = message['extra']['response']['usage']
            completion_tokens += usage.get("completion_tokens", 0)
            prompt_tokens += usage.get("prompt_tokens", 0)
            details = usage.get('prompt_tokens_details')
            if isinstance(details, dict):
                cached_tokens += details.get("cached_tokens", 0)

    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens
    }

def write_jsonl(file_path, data):
    """Write a list of JSON objects to a JSONL file."""
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def format_instruction(instance: dict) -> str:
    function_name = instance['metadata']['function_name']
    function_path = str(Path("/testbed") / Path(*instance['metadata']['fpath_tuple'][1:]))
    line_no = instance['metadata']['real_lineno']
    context = instance['prompt']
    import_path = instance['metadata']['import_path']
    return f"""
## Target Function Information

Function file path: {function_path}
Function name: {function_name}
Function definition line number: {line_no}
Import path for the target function : {import_path}
Code context in the same file :
{context}
"""


def load_repoeval_dataset(dataset_path: str):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    return dataset


def get_repoeval_predict_env(
    instance: dict,
    run_args=["--rm", "--memory=4g", "--cpus=2.0"],  
) :
    repo_name = instance['metadata']['fpath_tuple'][0].lower()
    docker_image = f"natt1e/{repo_name}:v0"
    environment = DockerEnvironment(
        image=docker_image.strip(),
        cwd="/testbed",
        run_args=run_args,
        env={
            "HF_ENDPOINT": "https://hf-mirror.com",
            "MPLBACKEND": "Agg",
            "HF_HUB_OFFLINE" : "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1"
        },
    )
    full_path = Path("/testbed") / Path(*instance['metadata']['fpath_tuple'][1:])
    function_name = instance['metadata']['function_name']
    real_lineno = instance['metadata']['real_lineno']
    truncate_line_number = instance['metadata']['lineno']
    original_code = environment.execute(
        {"command": f"cat {shlex.quote(str(full_path))}"},
        cwd="/testbed",
    )
    mask_target_function(
        environment,
        str(full_path),
        function_name,
        real_lineno,
        truncate_line_number,
    )
    return environment, original_code

def get_repoeval_env(
    instance: dict,
    run_args=["--rm", "--memory=4g", "--cpus=2.0"],
    mask_function=True,
    version="v0"
):
    repo_name = instance['metadata']['fpath_tuple'][0].lower()
    docker_image = f"natt1e/{repo_name}:{version}"
    print(docker_image)
    environment = DockerEnvironment(
        image=docker_image.strip(),
        cwd="/testbed",
        run_args=run_args,
        env={
            "HF_ENDPOINT": "https://hf-mirror.com",
            "MPLBACKEND": "Agg",
            "HF_HUB_OFFLINE" : "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1"
        },
    )
    full_path = Path("/testbed") / Path(*instance['metadata']['fpath_tuple'][1:])
    function_name = instance['metadata']['function_name']
    real_lineno = instance['metadata']['real_lineno']
    truncate_line_number = instance['metadata']['lineno']
    if mask_function:
        mask_target_function(
            environment,
            str(full_path),
            function_name,
            real_lineno,
            truncate_line_number,
        )
    return environment


def mask_target_function(
    environment: DockerEnvironment,
    original_file_path: str,
    function_name: str,
    real_line_number: int,
    truncate_line_number: int,
) -> None:
    read_result = environment.execute(
        {"command": f"cat {shlex.quote(original_file_path)}"},
        cwd="/testbed",
    )
    if read_result.get("returncode", 1) != 0:
        raise RuntimeError(
            "Failed to read target function file in docker container: "
            f"{read_result.get('output', '')}"
        )

    original_content = read_result.get("output", "")
    tree = ast.parse(original_content)
    indent_str = ""
    end_line = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name and node.lineno == real_line_number:
                end_line = node.end_lineno
                body_col = min(n.col_offset for n in node.body if hasattr(n, "col_offset"))
                indent_str = " " * body_col
                break

    if end_line is None:
        raise ValueError(
            f"Function not found: {function_name} at line {real_line_number} in {original_file_path}"
        )

    mask_statement = [
        f"{indent_str}## To be implemented",
        f"{indent_str}...",
    ]
    original_lines = original_content.splitlines()
    masked_content = (
        original_lines[:truncate_line_number] +
        mask_statement +
        original_lines[end_line:]
    )
    encoded_content = base64.b64encode("\n".join(masked_content).encode("utf-8")).decode("ascii")
    command = (
        f"printf %s {shlex.quote(encoded_content)} | base64 -d > "
        f"{shlex.quote(original_file_path)}"
    )
    result = environment.execute({"command": command}, cwd="/testbed")
    if result.get("returncode", 1) != 0:
        raise RuntimeError(
            "Failed to mask target function in docker container: "
            f"{result.get('output', '')}"
        )
        
