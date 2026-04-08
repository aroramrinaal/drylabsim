---
title: DryLabSim
sdk: docker
pinned: true
app_port: 8000
base_path: /demo
tags:
  - openenv
  - RL Environment
  - bioinformatics
  - computational biology
  - biology
  - scientific planning
colorFrom: green
colorTo: green
short_description: LLM agents plan noisy biological experiment pipelines
---

# DryLabSim

DryLabSim is an OpenEnv-compatible reinforcement learning environment for biological experiment planning. Instead of solving a toy puzzle, an agent must plan a realistic dry-lab and wet-lab pipeline one step at a time under partial observability, noisy outputs, budget constraints, time pressure, and scientific validity rules.

The core challenge is not just "get the right answer." The agent must:
- choose scientifically valid next steps
- deal with incomplete and noisy intermediate results
- spend budget and time carefully
- avoid unsupported causal claims
- synthesize a final conclusion that matches hidden biological ground truth

<img width="1710" height="883" alt="Screenshot 2026-04-08 at 10 52 13 PM" src="https://github.com/user-attachments/assets/a8b571e4-d341-4d79-8aca-e9f35f285488" />


The environment is graded deterministically with programmatic biology and pipeline scoring. There is no LLM judge in the final score loop.

## Task Difficulties

DryLabSim currently ships with four benchmark tasks:

| task | difficulty | tissue | focus |
|---|---|---|---|
| `cardiac_disease_de` | easy | heart | recover differential expression and validate markers in dilated cardiomyopathy |
| `hematopoiesis_trajectory` | medium | bone marrow | reconstruct branching hematopoietic trajectories and regulatory drivers |
| `perturbation_immune` | hard | synovial fluid | explain how JAK inhibition shifts immune cell state in rheumatoid arthritis |
| `venetoclax_resistance_multiclone` | expert | bone marrow | disentangle parallel AML resistance mechanisms across multiple post-treatment subclones |

Why these tasks are interesting:
- they cover differential expression, trajectory inference, perturbation analysis, and multiclone resistance reasoning
- each task has hidden biological truth plus hidden technical noise
- the expert task includes adversarial structure, including a distractor clone that should not be overclaimed as resistance

## Why This Environment Is Strong for a Hackathon

- **Real planning problem**: the agent chooses structured scientific actions instead of generating free-form text only.
- **Partial observability**: hidden biology, hidden failure modes, and noisy outputs make planning non-trivial.
- **Deterministic grading**: `server/grader/grade.py` computes a reproducible final score from pipeline quality, biology recovery, and efficiency.
- **Scientifically grounded tasks**: scenarios encode true DE genes, pathways, regulatory programs, trajectory structure, and causal mechanisms.
- **Human-demo ready**: the project exposes a custom browser demo at `/demo` in addition to the OpenEnv server routes.

## Quick Start

The simplest local workflow is:

```bash
uv sync --extra dev
uv run --project . server --host 0.0.0.0 --port 8000
```

Then open:
- `http://localhost:8000/demo` for the custom demo UI
- `http://localhost:8000/docs` for the FastAPI docs
- `http://localhost:8000/reset`, `http://localhost:8000/step`, and `http://localhost:8000/state` for the core environment API

You can also connect from Python:

```python
from drylabsim import BioExperimentEnv, ExperimentAction

with BioExperimentEnv(base_url="http://localhost:8000") as env:
    result = env.reset(task_name="easy")
    print(result.observation.task.problem_statement)

    result = env.step(
        ExperimentAction(
            action_type="collect_sample",
            parameters={"n_samples": 6},
            justification="Start the pipeline by collecting material for downstream profiling.",
            confidence=0.8,
        )
    )

    print(result.observation.latest_output.summary)
    print(result.reward)
```

## Running Locally

If you want to inspect the environment without Docker:

```bash
uv sync --extra dev
uv run --project . server --host 0.0.0.0 --port 8000
```

If you prefer `uvicorn` directly:

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

The root route redirects to `/demo`, which makes the repo easier to show to judges immediately.

## Project Commands

These are the project-specific commands currently used in this repo:

```bash
# OpenEnv validation
openenv validate --verbose

# Run tests
uv run --project . python -m pytest tests -q

# Docker build
docker build -t drylabsim:latest -f server/Dockerfile .

# Docker run
docker run --rm -p 8000:8000 drylabsim:latest
```

## Docker

Build the image from the project root:

```bash
docker build -t drylabsim:latest -f server/Dockerfile .
```

Run it locally:

```bash
docker run --rm -p 8000:8000 drylabsim:latest
```

After startup, visit `http://localhost:8000/demo`.

## Deploying to Hugging Face Spaces

This repo is structured as an OpenEnv environment and includes [`openenv.yaml`](https://github.com/aroramrinaal/drylabsim/blob/main/openenv.yaml), so deployment is straightforward:

```bash
openenv push
```

Useful variants:

```bash
openenv push --namespace my-org --private
openenv push --repo-id my-org/drylabsim
```

## Environment Details

### Action

`ExperimentAction` is a structured planning step with:
- `action_type`: the next scientific action to take
- `method`: optional named tool or protocol such as `Seurat` or `CellRanger`
- `parameters`: action-specific details such as comparisons, perturbation targets, or claims
- `justification`: optional scientific rationale
- `confidence`: agent calibration in `[0, 1]`

Representative action types include:
- wet-lab: `collect_sample`, `prepare_library`, `sequence_cells`, `validate_marker`
- computational: `run_qc`, `filter_data`, `normalize_data`, `cluster_cells`, `differential_expression`, `trajectory_analysis`, `pathway_enrichment`, `regulatory_network_inference`, `marker_selection`
- meta: `design_followup_experiment`, `request_subagent_review`, `synthesize_conclusion`

### Observation

`ExperimentObservation` exposes the visible state only:
- `task`: task specification, modality, tissue, budget, time limit, and success criteria
- `pipeline_history`: previous successful and failed steps
- `resource_usage`: budget, time, samples, and compute consumed so far
- `latest_output` and `all_outputs`: noisy simulated outputs from prior actions
- `discovered_markers` and `candidate_mechanisms`: evidence accumulated so far
- `rule_violations`: hard or soft scientific/procedural violations
- `conclusions`: structured claims submitted by the agent

The true latent biology is hidden from the agent and kept inside the simulator.

### Reward and Grading

DryLabSim has two scoring layers:

1. **Dense per-step reward**
   The environment gives step-wise reward for validity, ordering, information gain, efficiency, novelty, and penalties.

2. **Deterministic terminal grading**
   [`server/grader/grade.py`](https://github.com/aroramrinaal/drylabsim/blob/main/server/grader/grade.py) combines:
   - pipeline completeness: `0.30`
   - biology recovery: `0.55`
   - efficiency: `0.15`

Important grading properties:
- final grading is deterministic and reproducible
- the expert AML task applies extra caps if the agent misses parallel resistant clones, skips integration or validation, or overclaims a distractor clone
- conclusions are rewarded for calibration, not just confidence

## How an Episode Works

At a high level:

1. `reset()` selects a scenario and seeds the simulator.
2. The agent receives a partial observation with task metadata and visible history.
3. The agent submits one structured action.
4. The rule engine checks prerequisites, redundancy, resource limits, causal validity, and tool compatibility.
5. The transition engine updates hidden state and generates a noisy intermediate output.
6. The reward computer scores the step.
7. The environment returns the next observation.
8. The episode ends when the agent synthesizes a conclusion, exhausts resources, or hits the step limit.

This creates a POMDP where the agent has to plan, adapt, and calibrate instead of following a perfectly observed pipeline.

## Inference Entrypoint

[`inference.py`](https://github.com/aroramrinaal/drylabsim/blob/main/inference.py) is the hackathon inference entrypoint. It:
- connects to the environment over HTTP
- drives the environment step-by-step
- prints machine-parseable `[START]`, `[STEP]`, and `[END]` lines
- supports `easy`, `medium`, `hard`, and `expert` task aliases

Environment variables used by the baseline include:
- `API_BASE_URL`
- `HF_TOKEN` or `API_KEY`
- `MODEL_NAME`


- `ENV_URL`



## Baseline Scores



Baseline scores from running `inference.py` with different models:



### Qwen/Qwen2.5-7B-Instruct

| task | difficulty | score |
|---|---|---|
| cardiac_disease_de | easy | 0.719 |
| hematopoiesis_trajectory | medium | 0.569 |
| perturbation_immune | hard | 0.363 |
| venetoclax_resistance_multiclone | expert | 0.150 |



### Qwen/Qwen2.5-72B-Instruct

| task | difficulty | score |
|---|---|---|
| cardiac_disease_de | easy | 0.654 |
| hematopoiesis_trajectory | medium | 0.487 |
| perturbation_immune | hard | 0.478 |
| venetoclax_resistance_multiclone | expert | 0.200 |



### openai/gpt-oss-120b

| task | difficulty | score |
|---|---|---|
| cardiac_disease_de | easy | 0.713 |
| hematopoiesis_trajectory | medium | 0.497 |
| perturbation_immune | hard | 0.395 |
| venetoclax_resistance_multiclone | expert | 0.250 |



These scores were obtained by averaging over single runs; multiple runs may vary due to stochasticity in the environment and model generation.





## Project Structure

```text
drylabsim/
├── README.md
├── __init__.py                  # Package exports
├── client.py                    # OpenEnv client for the environment
├── inference.py                 # Hackathon inference entrypoint
├── models.py                    # Action, observation, task, tool, and claim schemas
├── openenv.yaml                 # OpenEnv manifest and task registration
├── pyproject.toml               # Package metadata and dependencies
├── tests/                       # API, environment, rewards, simulator, and grader tests
├── server/
│   ├── app.py                   # FastAPI app and session-backed HTTP routes
│   ├── demo_ui.py               # Demo UI wiring
│   ├── Dockerfile               # Container image
│   ├── drylabsim_environment.py # Main environment orchestration
│   ├── biology/                 # Biology utilities and gene index support
│   ├── grader/                  # Deterministic terminal grading
│   ├── rewards/                 # Dense reward computation and breakdowns
│   ├── rules/                   # Validity, prerequisite, redundancy, and resource checks
│   ├── simulator/               # Latent state, transitions, noise, and output generation
│   ├── tasks/                   # Scenario generation and benchmark task library
│   └── demo/                    # Browser-based demo assets
└── .context/
    └── commands.md              # Project command reference
```

## Core Scientific Components

- [`server/tasks/`](https://github.com/aroramrinaal/drylabsim/tree/main/server/tasks) defines the benchmark scenarios and optional domain randomization.
- [`server/simulator/`](https://github.com/aroramrinaal/drylabsim/tree/main/server/simulator) holds the hidden biological state, stochastic noise model, transition logic, and output synthesis.
- [`server/rules/engine.py`](https://github.com/aroramrinaal/drylabsim/blob/main/server/rules/engine.py) enforces scientific sequencing and resource validity.
- [`server/rewards/`](https://github.com/aroramrinaal/drylabsim/tree/main/server/rewards) computes dense reward components.
- [`server/grader/`](https://github.com/aroramrinaal/drylabsim/tree/main/server/grader) scores final episodes deterministically for benchmark evaluation.

## Why It Matters

DryLabSim is meant to benchmark a harder class of agent behavior than one-shot QA or static tool use. It evaluates whether an agent can plan a scientific investigation under uncertainty, gather evidence in the right order, use resources wisely, and end with a conclusion that is both informative and calibrated.

That makes it a good fit for:
- RL on long-horizon scientific planning
- evaluating LLM agents in realistic bioinformatics workflows
- benchmarking structured reasoning under partial observability
- comparing planning policies, tool strategies, and conclusion calibration
