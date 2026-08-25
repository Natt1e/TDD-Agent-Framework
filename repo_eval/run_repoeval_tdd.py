"""Run RepoEval with the multi-tool TDD agent."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any
import json
import time

import typer
import yaml
import tqdm

from minisweagent.utils.log import logger
from minisweagent.agents.tdd import TDDAgent
from minisweagent.models.tdd_litellm_model import TDDLitellmModel

from utils import (
    get_repoeval_predict_env,
    load_repoeval_dataset,
    format_instruction,
    token_count,
)
import litellm

litellm.suppress_debug_info = True

DEFAULT_CONFIG_FILE = "config/repoeval_tdd_multitool.yaml"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def _parse_instance_ids(id_specs: list[str]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for spec in id_specs:
        for instance_id in spec.split(","):
            instance_id = instance_id.strip()
            if instance_id and instance_id not in seen:
                ids.append(instance_id)
                seen.add(instance_id)
    return ids


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
    try:
        env, original_code = get_repoeval_predict_env(
            instance,
            run_args=["--rm", f"--memory={docker_memory}", f"--cpus={docker_cpus}"],
        )
        agent = TDDAgent(
            TDDLitellmModel(**model_cfg),
            env,
            instance,
            original_code,
            **agent_cfg,
            output_path=str(traj_output_dir / f"{instance_id}.json"),
        )
        artifacts = agent.run(format_instruction(instance))
        token_stats = token_count(str(traj_output_dir / f"{instance_id}.json"))

        runtime_seconds = time.perf_counter() - start_time
        output_instance = dict(instance)
        output_instance["predict"] = artifacts.get("code", [])
        output_instance["test"] = artifacts.get("test", [])
        output_instance["runtime_seconds"] = runtime_seconds
        output_instance["completion_tokens"] = token_stats.get("completion_tokens", -1)
        output_instance["prompt_tokens"] = token_stats.get("prompt_tokens", -1)
        output_instance["cached_tokens"] = token_stats.get("cached_tokens", 0)
        return output_instance
    except Exception as e:
        print(e)
        runtime_seconds = time.perf_counter() - start_time
        output_instance = dict(instance)
        output_instance["predict"] = []
        output_instance["test"] = []
        output_instance["runtime_seconds"] = runtime_seconds
        output_instance["completion_tokens"] = -1
        output_instance["prompt_tokens"] = -1
        output_instance["cached_tokens"] = 0
        return output_instance
    finally:
        if env is not None:
            env.cleanup()


@app.command()
def main(
    dataset_path: str = typer.Option(
        "function_2k_washed.jsonl",
        "-d",
        "--dataset",
        help="Path to the RepoEval dataset file",
        rich_help_panel="Basic",
    ),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", rich_help_panel="Basic"),
    traj_output: str = typer.Option(
        ".",
        "-o",
        "--traj-output",
        help="Directory to save trajectory json files",
        rich_help_panel="Basic",
    ),
    concurrency: int = typer.Option(5, "-j", "--concurrency", min=1, help="Number of worker threads", rich_help_panel="Basic"),
    predict_output: str = typer.Option(
        ".",
        "-p",
        "--predict-output",
        help="Directory to save prediction json files",
        rich_help_panel="Basic",
    ),
    docker_memory: str = typer.Option("4g", "--docker-memory", help="Docker memory limit passed to run args, e.g. 4g", rich_help_panel="Basic"),
    docker_cpus: float = typer.Option(1.0, "--docker-cpus", min=0.1, help="Docker CPU cores passed to run args", rich_help_panel="Basic"),
    debug: int = typer.Option(-1, "--debug", help="debug mode", rich_help_panel="Basic"),
    resume: bool = typer.Option(False, "--resume", help="Whether to resume from existing predict output file", rich_help_panel="Basic"),
    ids: list[str] = typer.Option(
        [],
        "-i",
        "--id",
        "--ids",
        help=(
            "Run only the specified RepoEval ids. Can be repeated or comma-separated, "
            "e.g. -i 1 -i 2 or --ids 1,2."
        ),
        rich_help_panel="Data selection",
    ),
) -> None:
    logger.info(f"Building agent config from specs: {config_spec}")
    config_file = Path(config_spec[0])
    cfg = yaml.safe_load(config_file.read_text())
    model_cfg = cfg["model"]
    agent_cfg = cfg["agent"]

    traj_output_dir = Path(traj_output)
    traj_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_repoeval_dataset(dataset_path)

    requested_ids = _parse_instance_ids(ids)
    if requested_ids:
        requested_id_set = set(requested_ids)
        dataset = [instance for instance in dataset if str(instance["id"]) in requested_id_set]
        matched_ids = {str(instance["id"]) for instance in dataset}
        missing_ids = [instance_id for instance_id in requested_ids if instance_id not in matched_ids]
        if missing_ids:
            raise typer.BadParameter(f"Requested ids not found in dataset: {', '.join(missing_ids)}")
        logger.info(f"Filtered dataset to {len(dataset)} requested ids")

    if resume:
        existing_ids = []
        if Path(predict_output).is_file():
            with open(predict_output, "r", encoding="utf-8") as file:
                for line in file:
                    existing_ids.append(str(json.loads(line)["id"]))
            dataset = [instance for instance in dataset if str(instance["id"]) not in existing_ids]
    if debug > 0:
        dataset = dataset[:debug]

    worker = partial(
        _run_single_instance,
        model_cfg=model_cfg,
        agent_cfg=agent_cfg,
        traj_output_dir=traj_output_dir,
        docker_memory=docker_memory,
        docker_cpus=docker_cpus,
    )
    output_path = Path(predict_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        open_mode = "a" if resume else "w"
        with open(output_path, open_mode, encoding="utf-8") as handle:
            futures = [executor.submit(worker, instance) for instance in dataset]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                handle.write(json.dumps(result) + "\n")
                handle.flush()


if __name__ == "__main__":
    app()
