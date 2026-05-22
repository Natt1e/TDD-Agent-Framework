from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any
import time

import typer

from minisweagent.utils.log import logger
from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel

import yaml

from utils import (
    get_repoeval_env,
    load_repoeval_dataset,
    format_instruction,
    write_jsonl,
    token_count,
    extract_final_submission,
)
import tqdm


DEFAULT_CONFIG_FILE = ""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def _run_single_instance(
    instance: dict[str, Any],
    model_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    traj_output_dir: Path,
    docker_memory: str,
    docker_cpus: float,
) -> dict[str, Any]:
    instance_id = instance["id"]
    env = None
    start_time = time.perf_counter()
    try :
        env = get_repoeval_env(
            instance,
            run_args=["--rm", f"--memory={docker_memory}", f"--cpus={docker_cpus}"]
        )
        agent = DefaultAgent(
            LitellmModel(**model_cfg),
            env,
            **agent_cfg,
            output_path=str(traj_output_dir / f"{instance_id}.json"),
        )
        result = agent.run(format_instruction(instance))
        token_stats = token_count(str(traj_output_dir / f"{instance_id}.json"))
        submission = result["submission"]
        if not submission:
            submission = extract_final_submission(str(traj_output_dir / f"{instance_id}.json"), model_cfg)
        
        runtime_seconds = time.perf_counter() - start_time
        output_instance = dict(instance)
        output_instance["predict"] = [submission]
        output_instance["runtime_seconds"] = runtime_seconds
        output_instance['completion_tokens'] = token_stats.get('completion_tokens', -1)
        output_instance['prompt_tokens'] = token_stats.get('prompt_tokens', -1)
        output_instance['cached_tokens'] = token_stats.get('cached_tokens', 0)
        return output_instance
    except Exception as e:
        runtime_seconds = time.perf_counter() - start_time
        output_instance = dict(instance)
        output_instance["predict"] = ['']
        output_instance["runtime_seconds"] = runtime_seconds
        output_instance['completion_tokens'] = -1
        output_instance['prompt_tokens'] = -1
        output_instance['cached_tokens'] = 0
        return output_instance
    finally:
        if env is not None:
            env.cleanup()



@app.command()
def main(
    dataset_path: str = typer.Option("/mnt/data/projects/yuhy/tdd_repogen/dataset/repo_eval/function_2k_washed.jsonl", "-d", "--dataset", help="Path to the RepoEval dataset file", rich_help_panel="Basic"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", rich_help_panel="Basic"),
    traj_output: str = typer.Option(".", "-o", "--traj-output", help="Directory to save trajectory json files", rich_help_panel="Basic"),
    concurrency: int = typer.Option(5, "-j", "--concurrency", min=1, help="Number of worker threads", rich_help_panel="Basic"),
    predict_output: str = typer.Option(".", "-p", "--predict-output", help="Directory to save prediction json files", rich_help_panel="Basic"),
    docker_memory: str = typer.Option("4g", "--docker-memory", help="Docker memory limit passed to run args, e.g. 4g", rich_help_panel="Basic"),
    docker_cpus: float = typer.Option(2.0, "--docker-cpus", min=0.1, help="Docker CPU cores passed to run args", rich_help_panel="Basic"),
    
) -> None:
    logger.info(f"Building agent config from specs: {config_spec}")
    config_file = Path(config_spec[0])
    cfg = yaml.safe_load(config_file.read_text())
    model_cfg = cfg["model"]
    agent_cfg = cfg["agent"]

    traj_output_dir = Path(traj_output)
    traj_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_repoeval_dataset(dataset_path)

    worker = partial(
        _run_single_instance,
        model_cfg=model_cfg,
        agent_cfg=agent_cfg,
        traj_output_dir=traj_output_dir,
        docker_memory=docker_memory,
        docker_cpus=docker_cpus,
    )
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(worker, dataset),
                total=len(dataset),
            )
        )

    write_jsonl(predict_output, results)

if __name__ == "__main__":
    app()
