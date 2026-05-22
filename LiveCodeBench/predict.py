import json
import yaml
import re
import argparse
import time
from dataclasses import dataclass
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
from functools import partial
from datasets import load_dataset
from datetime import datetime


def format_prompt(task):
    return f"""{task.question_content}

```python
{task.starter_code}
```
"""

@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: str
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: str
    public_test_cases: list
    private_test_cases: list
    metadata: dict


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_code(text):
    if text is None:
        return ""

    python_matches = re.findall(r'```python(.*?)```', text, re.DOTALL)
    if python_matches:
        return python_matches[-1].strip()

    generic_matches = re.findall(r'```(.*?)```', text, re.DOTALL)
    if generic_matches:
        return generic_matches[-1].strip()
    
    return text.strip()


def split_model_kwargs(model_kwargs):
    kwargs = {}
    extra_body = {}
    standard_keys = {
        'temperature', 'top_p', 'max_tokens', 'stop',
        'presence_penalty', 'frequency_penalty', 'stream', 'seed',
        'reasoning_effort', 'verbosity'
    }
    for k, v in (model_kwargs or {}).items():
        if k in standard_keys:
            kwargs[k] = v
        elif k != 'n':
            extra_body[k] = v
    if extra_body:
        kwargs['extra_body'] = extra_body
    return kwargs


def chat_completion(client, model_name, messages, model_kwargs, max_retries=5, timeout=120):
    kwargs = split_model_kwargs(model_kwargs)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                timeout=timeout,
                **kwargs
            )

            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n[Error] API call failed after {max_retries} attempts: {e}")
                raise
            wait_time = min(2 ** attempt, 60)
            print(f"\n[Warning] API call failed: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)


def chat_completion_n(client, model_name, messages, model_kwargs, num_samples, inner_workers=4, max_retries=5, timeout=120):
    if num_samples == 1:
        return [chat_completion(client, model_name, messages, model_kwargs, max_retries=max_retries, timeout=timeout)]

    def _single_call(_):
        try:
            return chat_completion(client, model_name, messages, model_kwargs, max_retries=max_retries, timeout=timeout)
        except Exception:
            return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_samples, inner_workers)) as executor:
        futures = [executor.submit(_single_call, i) for i in range(num_samples)]
        for future in futures:
            results.append(future.result())
    return results


def render_template(template, mapping):
    rendered = template
    for key, value in mapping.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


def build_samples_for_task(client, model_cfg, prompt_cfg, task, mode, num_samples, inner_workers, max_retries, timeout):
    # Copy once per task to avoid mutating shared config across worker threads.
    model_kwargs = dict(model_cfg.get('model_kwargs', {}))
    model_name = model_cfg['model_name']

    if mode == 'one_shot':
        messages = [{'role': 'system', 'content': prompt_cfg['one_shot_system_prompt']}]
        for ex in prompt_cfg.get('one_shot_examples', []):
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({'role': 'user', 'content': format_prompt(task)})
        texts = chat_completion_n(client, model_name, messages, model_kwargs, num_samples, inner_workers=inner_workers, max_retries=max_retries, timeout=timeout)
        return [
            {
                'intermediate': '',
                'code': t
            } 
            for t in texts
        ]

    if mode == 'cot':
        messages = [{'role': 'system', 'content': prompt_cfg['cot_system_prompt']}]
        messages.append({'role': 'user', 'content': format_prompt(task)})
        texts = chat_completion_n(client, model_name, messages, model_kwargs, num_samples, inner_workers=inner_workers, max_retries=max_retries, timeout=timeout)
        return [
            {
                'intermediate': '',
                'code': t
            } 
            for t in texts
        ]

    if mode == 'icot':
        messages = [{'role': 'system', 'content': prompt_cfg['icot_system_prompt']}]
        for ex in prompt_cfg.get('icot_examples', []):
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append(
            {
                'role': 'user', 
                'content': render_template(
                    prompt_cfg['icot_user'], 
                    {'prompt': format_prompt(task)}
                )
            }
        )
        
        breakdowns = chat_completion_n(
            client, 
            model_name, 
            messages, 
            model_kwargs, 
            num_samples, 
            inner_workers=inner_workers, 
            max_retries=max_retries, 
            timeout=timeout
        )

        def process_icot(breakdown):
            breakdown = (breakdown or '').strip()
            try:
                combined_icot_prompt = format_prompt(task) + f"\n\n{breakdown}\n"

                messages = [{'role': 'system', 'content': prompt_cfg['icot_code_system_prompt']}]
                for ex in prompt_cfg.get('icot_code_examples', []):
                    messages.append({"role": "user", "content": ex["user"]})
                    messages.append({"role": "assistant", "content": ex["assistant"]})
                messages.append(
                    {
                        'role': 'user', 
                        'content': render_template(
                            prompt_cfg['icot_code_user'], 
                            {'combined_icot_prompt': combined_icot_prompt}
                        )
                    }
                )
                codegen_kwargs = dict(model_kwargs)
                codegen_kwargs['temperature'] = 0.0
                completion_text = chat_completion(
                    client, 
                    model_name, 
                    messages, 
                    codegen_kwargs, 
                    max_retries=max_retries, 
                    timeout=timeout
                )
                return {
                    'intermediate': breakdown,
                    'code': completion_text
                }
            except Exception as e:
                return {'intermediate': '', 'code': ''}

        with concurrent.futures.ThreadPoolExecutor(max_workers=inner_workers) as executor:
            return list(executor.map(process_icot, breakdowns))

    if mode == 'self_plan':
        messages = [{'role': 'system', 'content': prompt_cfg['plan_system_prompt']}]
        for ex in prompt_cfg.get('plan_examples', []):
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({'role': 'user', 'content': format_prompt(task)})
        plans = chat_completion_n(
            client, 
            model_name, 
            messages, 
            model_kwargs, 
            num_samples, 
            inner_workers=inner_workers, 
            max_retries=max_retries, 
            timeout=timeout
        )

        def process_self_plan(plan):
            plan = (plan or '').strip()
            try:
                code_messages = [{'role': 'system', 'content': prompt_cfg['plan_code_system']}]
                for ex in prompt_cfg.get('plan_code_examples', []):
                    code_messages.append({"role": "user", "content": ex["user"]})
                    code_messages.append({"role": "assistant", "content": ex["assistant"]})
                code_messages.append({
                    'role': 'user', 
                    'content': f"{format_prompt(task)}\n\n{plan}\n"
                })
                codegen_kwargs = dict(model_kwargs)
                codegen_kwargs['temperature'] = 0.0
                completion_text = chat_completion(
                    client, 
                    model_name, 
                    code_messages, 
                    codegen_kwargs, 
                    max_retries=max_retries, 
                    timeout=timeout
                )
                return {
                    'intermediate': plan,
                    'code': completion_text
                }
            except Exception as e:
                return {
                    'intermediate': plan,
                    'code': ''
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=inner_workers) as executor:
            return list(executor.map(process_self_plan, plans))

    if mode == 'tdd':
        messages = [{'role': 'system', 'content': prompt_cfg['tdd_system']}]
        messages.append({
            'role': 'user', 
            'content': render_template(
                    prompt_cfg['tdd_user'], 
                    {
                        'prompt': task.question_content,
                        'starter_code': task.starter_code,
                    })
        })
        texts = chat_completion_n(client, model_name, messages, model_kwargs, num_samples, inner_workers=inner_workers, max_retries=max_retries, timeout=timeout)
        return [
            {
                'intermediate': '',
                'code': t
            } 
            for t in texts
        ]
        
    if mode == 'tdd_ablation':
        messages = [{'role': 'system', 'content': prompt_cfg['tdd_ablation_system']}]
        messages.append({
            'role': 'user', 
            'content': render_template(
                    prompt_cfg['tdd_ablation_user'], 
                    {
                        'prompt': task.question_content,
                        'starter_code': task.starter_code,
                    })
        })
        texts = chat_completion_n(client, model_name, messages, model_kwargs, num_samples, inner_workers=inner_workers, max_retries=max_retries, timeout=timeout)
        return [
            {
                'intermediate': '',
                'code': t
            } 
            for t in texts
        ]
        
        
    if mode == 'scot':
        messages = [{'role': 'system', 'content': prompt_cfg['scot_system_prompt']}]
        for ex in prompt_cfg.get('scot_examples', []):
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({
            'role': 'user', 
            'content': render_template(
                prompt_cfg['scot_user'],
                {'prompt': format_prompt(task)}
            )
        })
        plans = chat_completion_n(
            client, 
            model_name, 
            messages, 
            model_kwargs, 
            num_samples, 
            inner_workers=inner_workers,
            max_retries=max_retries, 
            timeout=timeout
        )

        def process_scot(plan):
            plan = (plan or '').strip()
            try:
                code_messages = [{'role': 'system', 'content': prompt_cfg['scot_code_system']}]
                code_messages.append({
                    'role': 'user', 
                    'content': render_template(
                        prompt_cfg['scot_code_user'],
                        {'combined_prompt': format_prompt(task) + '\n\n' + plan}
                    )
                })
                codegen_kwargs = dict(model_kwargs)
                codegen_kwargs['temperature'] = 0.0
                completion_text = chat_completion(client, model_name, code_messages, codegen_kwargs, max_retries=max_retries, timeout=timeout)
                return {
                    'intermediate': plan,
                    'code': completion_text
                }
            except Exception as e:
                return {
                    'intermediate': plan,
                    'code': ''
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=inner_workers) as executor:
            return list(executor.map(process_scot, plans))

    raise ValueError(f"Unsupported mode: {mode}")


def process_single_task_wrapper(task, client, model_cfg, prompt_cfg, mode, num_samples, inner_workers, max_retries, timeout):
    res = {
        'question_id': task.question_id,
        'mode': mode,
        'num_samples': num_samples,
        'raw_response': [],
        'code_list': []
    }
    try:
        samples = build_samples_for_task(client, model_cfg, prompt_cfg, task, mode, num_samples, inner_workers, max_retries, timeout)
        if len(samples) != num_samples:
            if len(samples) < num_samples:
                samples = samples + ([{'code': 'Error', 'intermediate': 'Error'}] * (num_samples - len(samples)))
            else:
                samples = samples[:num_samples]
        res['raw_response'] = samples
        res['code_list'] = [extract_code(s['code']) for s in samples]
    except Exception as e:
        samples = []
        for _ in range(num_samples):
            sample = {'code': 'Error', 'intermediate': 'Error'}
            samples.append(sample)
        res['code_list'] = [extract_code(s['code']) for s in samples]
    return res


def main():
    parser = argparse.ArgumentParser(description='Generate predictions via OpenAI website API')
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to config yaml')
    parser.add_argument('--model_config', type=str, default='qwen3.yaml', help='Path to model config yaml')
    parser.add_argument('--mode', type=str, default=None, help='Generation strategy')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to generate per task')
    parser.add_argument('--output_path', type=str, default=None, help='Override output path')
    parser.add_argument('--max_workers', type=int, default=12, help='Number of concurrent task workers')
    parser.add_argument('--inner_workers', type=int, default=1, help='Number of concurrent worker threads for per-task sampling')
    parser.add_argument('--max_retries', type=int, default=5, help='Maximum number of retries for each API call')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout in seconds for each API request')
    parser.add_argument('--start', type=int, default=0, help='The start to predict')
    parser.add_argument('--end', type=int, default=None, help='The end to predict')
    args = parser.parse_args()

    config = load_config(args.config_path)
    model_cfg = load_config(args.model_config)['model']
    prompt_cfg = config['prompt']
    run_cfg = config.get('run', {})
    mode = args.mode or run_cfg.get('mode', 'one_shot')
    num_samples = args.num_samples if args.num_samples is not None else run_cfg.get('num_samples', 1)

    if num_samples < 1:
        raise ValueError('num_samples must be at least 1')

    client = OpenAI(
        base_url=model_cfg.get('base_url'),
        api_key=model_cfg.get('api_key', 'EMPTY')
    )

    dataset_name = config['data']['dataset']
    release = config['data']['release']
    platform = config['data']['platform']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    try:
        tasks = load_dataset(dataset_name, version_tag=release)
    except ValueError as e:
        msg = str(e)
        if "Couldn't find cache for" in msg:
            tasks = load_dataset(dataset_name, release)
        else:
            raise

    tasks = tasks['test']
    tasks = [CodeGenerationProblem(**p) for p in tasks]  # type: ignore
    if start_date:
        p_start_date = datetime.strptime(start_date, '%Y-%m-%d')
        tasks = [e for e in tasks if p_start_date <= datetime.strptime(str(e.contest_date)[:10], '%Y-%m-%d')]
    if end_date:
        p_end_date = datetime.strptime(end_date, '%Y-%m-%d')
        tasks = [e for e in tasks if datetime.strptime(str(e.contest_date)[:10], '%Y-%m-%d') <= p_end_date]
    if platform:
        tasks = [e for e in tasks if e.platform == platform]

    print(f"Loaded {len(tasks)} tasks. Generating {num_samples} samples per task in '{mode}' mode...")


    tasks.sort(key=lambda task: task.question_id)
    tasks = tasks[args.start : args.end]
        
    output_path = args.output_path or config['data']['output_path']
    print(f"Using ThreadPoolExecutor with {args.max_workers} outer task workers and {args.inner_workers} inner workers for sampling.")
    print(f"Each API request: timeout={args.timeout}s, max_retries={args.max_retries}.")

    task_processor = partial(
        process_single_task_wrapper,
        client=client,
        model_cfg=model_cfg,
        prompt_cfg=prompt_cfg,
        mode=mode,
        num_samples=num_samples,
        inner_workers=args.inner_workers,
        max_retries=args.max_retries,
        timeout=args.timeout
    )

    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = executor.map(task_processor, tasks)
        for res in tqdm(results, total=len(tasks), desc='Predicting'):
            all_results.append(res)

    all_results.sort(key=lambda x: str(x.get('question_id', '')))

    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(all_results, f_out, ensure_ascii=False, indent=2)
        f_out.write('\n')

    print(f"Predictions saved to {output_path}")


if __name__ == '__main__':
    main()
