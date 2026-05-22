## TDD-Agent Setup

This project is build on [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)

```bash
pip install -e .
```

## How to run

### Prepare docker environments

`RepoEval` contains 8 code repositories, we have prepared the docker images for every repository.

This script will autonomously pull all 8 images and run `pytest` in each container to check the environment.

```bash
python pull_and_check_docker.py
```

### Config your own configuration

Config your api_key in config/*.yaml

These yaml files contains the prompt for each methods:

```bash
repo_eval/
└── config/
    ├── tdd_agent.yaml : the config for TDD-Agent
    ├── mini_swe_agent.yaml: the config for mini-swe-agent
    └── ablation/
        ├── tdd_agent_reflect.yaml: the config for Reflect-Variant 
        ├── tdd_agent_single_track.yaml: the config for Single-Track-Variant 
        ├── tdd_agent_vanilla.yaml: the config for Vanilla-Variant
```

### Run predict, baseline, ablation and evalutaion

#### Run TDD-Agent:

After add your output_path to `sh/run_tdd_agent.sh`, run:

```bash
bash sh/run_tdd_agent.sh
```

#### Run mini-swe-agent:

After add your output_path to `sh/run_mini_swe_agent.sh`, run:

```bash
bash sh/run_mini_swe_agent.sh
```

#### Run ablation variants:

After add your output_path to `sh/run_ablation.sh`, run

```bash
bash sh/run_ablation.sh
```

#### Run evalution:

After add your output_path to `sh/run_tdd_agent.sh`, run:

```bash
bash sh/run_evaluation.sh
```

## TDD-prompt

All prompts used are in LiveCodeBench/config/model.yaml

mode can be selected from :

['one_shot', 'cot', 'icot', 'self_plan', 'scot', 'tdd', 'tdd_ablation']

Run predict using:
```bash
python "LiveCodeBench/predict.py" \
    --config_path LiveCodeBench/config/config.yaml \
    --model_config LiveCodeBench/config/model.yaml \
    --mode tdd \
    --max_workers 5 \
    --inner_workers 5 \
    --max_retries 10 \
    --timeout 600 \
    --num_samples 10 \
    --output_path your-predict-path 
```