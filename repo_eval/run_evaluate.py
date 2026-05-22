import argparse
import json
import os
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

from tqdm import tqdm

from evaluate_utils import replace_function_in_file
from utils import get_repoeval_env


def read_jsonl(file_path: str) -> list[dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(file_path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_implementation_code(content: str) -> str:
    """Extract implementation code from fenced markdown blocks.

    Supports both ```python ... ``` and ``` ... ``` forms.
    If no fenced code block exists, returns the original input string.
    """
    if not isinstance(content, str):
        return content

    pattern = re.compile(r"```\s*([a-zA-Z0-9_+-]*)\s*\n([\s\S]*?)\n```", re.MULTILINE)
    matches = list(pattern.finditer(content))
    if not matches:
        return content

    for match in matches:
        language = (match.group(1) or "").strip().lower()
        if language in {"python", "py"}:
            return match.group(2)

    return matches[0].group(2)


def run_pytest_in_env(env, command: str, timeout: int = 600) -> dict:
    result = env.execute(
        {"command": command},
        cwd="/testbed",
        timeout=timeout,
    )
    return {
        "status": "success" if result.get("returncode", 1) == 0 else "failed",
        "returncode": result.get("returncode", 1),
        "stdout": result.get("output", ""),
    }


def save_result(
    output_path,
    repo,
    result_id,
    status,
    id_info,
):
    if 'detail' not in id_info:
        id_info['detail'] = {}
    if 'score' not in id_info:
        id_info['score'] = {}

    result_key = str(result_id)
    if repo not in id_info['detail']:
        id_info['detail'][repo] = {}
    if result_key not in id_info['detail'][repo]:
        id_info['detail'][repo][result_key] = []

    if status == 'success' :
        id_info['detail'][repo][result_key].append(True)
    else :
        id_info['detail'][repo][result_key].append(False)

    repo_detail = id_info['detail'][repo]
    max_submit_count = max((len(v) for v in repo_detail.values()), default=0)
    repo_score = []
    for submit_idx in range(max_submit_count):
        success_count = 0
        for attempts in repo_detail.values():
            if not attempts:
                continue
            idx = min(submit_idx, len(attempts) - 1)
            if attempts[idx]:
                success_count += 1
        repo_score.append(success_count)
    id_info['score'][repo] = repo_score
    
    write_json(output_path, id_info)


def process_one_item(
    repo,
    result,
    args,
    eval_lock,
    id_info_results,
):
    status = 'failed'
    instance_id = result.get('id', 'unknown_id')
    if not result.get('predict', None):
        print(f'[WARN] No predictions')
        return
    last_execution_result = None
    for i, predict_code in enumerate(result['predict']):
        if last_execution_result is not None:
            if predict_code == result['predict'][i-1]:
                with eval_lock:
                    save_result(
                        args.output_path,
                        repo,
                        instance_id,
                        last_execution_result,
                        id_info_results,
                    )
                    continue
        completed_code = extract_implementation_code(predict_code)
        env = None
        if completed_code == '' :
            status = 'failed'
        else :
            run_command = "pytest -x -q --disable-warnings"
            try:

                env = get_repoeval_env(
                    result, 
                    run_args=["--rm", f"--memory={args.memory}g", f"--cpus={args.cpus}"],
                    mask_function=False,
                )

                # 2. Replace file in docker
                container_file_path = str(Path('/testbed') / Path(*result['metadata']['fpath_tuple'][1:]))
                replace_function_in_file(
                    env,
                    container_file_path,
                    result['metadata']['lineno'],
                    result['metadata']['function_name'],
                    completed_code,
                    result['metadata']['ground_truth'],
                )

                # 3. Run pytest in docker
                status = run_pytest_in_env(
                    env,
                    run_command,
                    timeout=args.timeout
                )['status']
                last_execution_result = status
            except Exception as e:
                status = 'failed'
                print(f"[ERROR] id={instance_id}: {e}")
            finally:
                if env is not None:
                    env.cleanup()

        # 4. Collect result
        with eval_lock:
            save_result(
                args.output_path,
                repo,
                instance_id,
                status,
                id_info_results,
            )

def main(args):
    
    id_info_results = {
        'total_success': [],
        'total_rate': [],
        'detail': {},
        'score': {},
    }
    
    results = read_jsonl(args.predict_path)
    random.seed(args.shuffle_seed)
    random.shuffle(results)

    
    for result in results:
        if not isinstance(result.get('predict'), list):
            result['predict'] = [result['predict']]

    eval_lock = threading.Lock()     
    progress_lock = threading.Lock()    

    total_tasks = len(results)
    pbar = tqdm(total=total_tasks, desc='Evaluating', ncols=100)


    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = []
        for item in results:
            repo = item['metadata']['fpath_tuple'][0]
            futures.append(
                ex.submit(
                    process_one_item, 
                    repo,
                    item,
                    args,
                    eval_lock,
                    id_info_results,
                )
            )

        for fut in as_completed(futures):
            _ = fut.result()
            with progress_lock:
                pbar.update(1)

    max_submit_count_global = max(
        (len(scores) for scores in id_info_results['score'].values()), 
        default=0
    )
    
    for index in range(max_submit_count_global):
        total = 0
        for repo, scores in id_info_results['score'].items():
            if not scores:
                continue
            idx = min(index, len(scores) - 1) 
            total += scores[idx]
        
        id_info_results['total_success'].append(total)
        id_info_results['total_rate'].append(total / len(results))
    write_json(args.output_path, id_info_results)
    pbar.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path', type=str, default='predict/tdd_agents_gpt-4o-mini.jsonl')
    parser.add_argument('--output_path', type=str, default='results/repo_results.json')
    parser.add_argument('--max_workers', type=int, default=5)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cpus', type=float, default=4.0)
    parser.add_argument('--memory', type=int, default=12)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--shuffle_seed', type=int, default=42)
    
    args = parser.parse_args()

    test_repos = [
        'leopard-ai_betty',
        'CarperAI_trlx',
        'lucidrains_imagen-pytorch',
        'deepmind_tracr',
        'google_lightweight_mmm',
        'amazon-science_patchcore-inspection',
        'facebookresearch_omnivore',
        'maxhumber_redframes'
    ]
    main(args)
