---
title: DryLabSim Server
sdk: docker
pinned: false
app_port: 8000
base_path: /demo
tags:
  - openenv
  - RL Environment
  - bioinformatics
  - computational biology
---

# DryLabSim

This repository implements an OpenEnv-compatible reinforcement learning environment for planning biological experiment pipelines. The agent does not directly see the true biological state. Instead, it proposes one structured experiment or analysis step at a time, receives a noisy simulated output, and is rewarded for valid, informative, efficient, well-calibrated plans.

The environment is designed as a partially observable Markov decision process (POMDP) with:

- hidden ground-truth biology
- hidden technical noise and failure conditions
- visible task metadata, resource usage, step history, and intermediate outputs
- dense step-wise reward plus terminal reward for conclusion quality

## How it works

At a high level, each episode looks like this:

1. `reset()` picks a biological scenario and seeds the simulator.
2. The agent receives an `ExperimentObservation` describing the task and current visible state.
3. The agent submits an `ExperimentAction` such as `collect_sample`, `run_qc`, or `differential_expression`.
4. The rule engine checks whether the action is valid at this point in the pipeline.
5. The transition engine updates hidden state, spends resources, and asks the output generator to simulate the result.
6. The reward computer scores the step for validity, ordering, information gain, efficiency, novelty, and penalties.
7. The environment returns a new observation with updated history, outputs, discoveries, violations, and reward.
8. The episode ends when the agent synthesizes a conclusion, exhausts resources, or reaches the step limit.

### `server/tasks/`

This is where episodes come from.

- `scenarios.py` defines a curated library of four biological scenarios as `Scenario` dataclass objects, each bundling a `TaskSpec`, a `LatentBiologicalState`, a `TechnicalState`, hidden failure conditions, and tags
- `generator.py` turns a scenario into a `(TaskSpec, FullLatentState)` pair via `TaskGenerator.generate()`; optional domain randomisation perturbs budget (±30%), time (±20%), technical noise, batch effects, cell proportions, and effect sizes

The four scenarios are:

| Name | Difficulty | Tissue | Problem | Budget | Time |
|---|---|---|---|---|---|
| `cardiac_disease_de` | easy | heart | Differential expression between healthy and dilated cardiomyopathy cardiomyocytes | $80 K | 120 days |
| `hematopoiesis_trajectory` | medium | bone marrow | Infer HSC → mature lineage trajectory with three branches | $100 K | 150 days |
| `perturbation_immune` | hard | synovial fluid | JAK inhibitor effect on T-cell states in rheumatoid arthritis | $120 K | 180 days |
| `venetoclax_resistance_multiclone` | expert | bone marrow | Resolve parallel venetoclax resistance mechanisms across AML subclones | $150 K | 210 days |

Each scenario carries paper references with DOIs, true DE genes with log2FC values, true pathway activities, true regulatory networks, and ground-truth causal mechanisms used for terminal reward calibration.

### `server/simulator/`

This is the simulator itself.

- `latent_state.py` defines `FullLatentState`, the root aggregate of all hidden state. Key sub-structures are `LatentBiologicalState` (true DE genes, pathways, gene programs, trajectory, regulatory network, markers, causal mechanisms), `TechnicalState` (dropout, doublets, ambient RNA, sample quality), `ExperimentProgress` (18 boolean milestone flags plus counts), and `ResourceState` (internal budget and time tracking with exhaustion properties)
- `noise.py` centralises stochasticity in `NoiseModel`. All randomness flows through a single seeded `numpy.Generator`. Methods include `add_expression_noise`, `sample_effect_sizes`, `sample_p_values`, `generate_false_positives`, `generate_false_negatives`, `quality_degradation`, `sample_qc_metric`, `sample_cluster_count`, `shuffle_ranking`, and `coin_flip`
- `output_generator.py` turns an action plus hidden state into a realistic `IntermediateOutput`. Every action type has a dedicated handler conditioned on the latent state; noise is then injected — dropout in expression data, false positives and false negatives in DE and marker results, over/under-clustering, and pathway contamination
- `transition.py` applies action costs from `ACTION_COSTS`, updates progress flags, calls the output generator, degrades quality on soft violations, propagates discovered DE genes and cluster names back into latent state, and decides whether the episode is done

The output generator does not simply echo the action. It conditions outputs on the hidden state, then injects realistic noise.

### `server/rules/engine.py`

The rule engine enforces scientific and procedural constraints before each action is applied.

- hard violations block the action entirely
- soft violations allow the action, but reduce output quality and add reward penalties

The five rule families are:

1. **Prerequisites (HARD)** — each computational step requires the appropriate upstream milestone flag. For example: `normalize_data` requires `data_filtered`, `differential_expression` requires `data_normalized`, `validate_marker` requires `markers_discovered`
2. **Resource constraints (HARD)** — budget or time exhausted, or action cost exceeding remaining budget, all block the action
3. **Redundancy (HARD)** — repeating an already-completed step such as `run_qc` or `normalize_data` is blocked
4. **Causal validity (HARD/SOFT)** — synthesizing conclusions without prior DE, clustering, marker, or mechanism evidence is blocked; unsupported causal claims and pathway enrichment before DE are soft warnings
5. **Tool compatibility (HARD)** — checks that the requested action is compatible with available tools and modalities

### `server/rewards/reward.py`

Rewards are decomposed rather than being a single opaque number.

Per-step reward formula:

```
R_t = r_validity + r_ordering + r_info_gain + r_efficiency + r_novelty + r_penalty + γ[φ(s_{t+1}) − φ(s_t)]
```

| Component | Weight | Description |
|---|---|---|
| `validity` | 0.3 | `1.0` if output succeeded, `−1.0` if hard violation |
| `ordering` | 0.2 | `1.0` if natural next step, `0.3` otherwise |
| `info_gain` | 0.4 | `quality_score × (1 − uncertainty)` |
| `efficiency` | 0.3 | `max(0, 1 − 5 × budget_fraction_used)` |
| `novelty` | +0.1 | Bonus when no soft violations |
| `penalty` | −0.15/violation | Per soft violation |
| `shaping` | γ = 0.99 | Potential-based over 12 progress milestones |

Terminal reward adds:

| Component | Weight | Description |
|---|---|---|
| Pipeline completeness | 3.0 | Fraction of 7 core milestones completed |
| Calibration | 4.0 | How well conclusions match hidden markers and mechanisms |
| Budget + time efficiency | 1.0 | Average fraction of budget and time remaining |
| Overconfidence penalty | −0.5/claim | For high-confidence claims (`> 0.8`) that are wrong |

This makes the environment easier to debug, benchmark, and train against.

### `server/drylabsim_environment.py`

This is the orchestration layer that ties everything together.

On `reset()` it:

- seeds the noise model
- generates a task and latent state via `TaskGenerator`
- clears history, outputs, discoveries, conclusions, and cumulative reward

On `step()` it:

- checks rules
- calls the transition engine
- computes reward
- appends a `PipelineStepRecord`
- updates discovered markers and candidate mechanisms
- stores conclusion claims if the action is `synthesize_conclusion`
- builds the next `ExperimentObservation`

This file is the best place to read if you want the end-to-end control flow.

## What actually happens on one step

Here is the concrete order of operations for `env.step(action)`:

1. Increment the step counter.
2. Copy the previous latent state for reward comparison.
3. Run rule checks and split violations into hard vs soft.
4. If there is a hard violation, return a failure report without applying the action.
5. Otherwise deduct budget and time based on `ACTION_COSTS`.
6. Update latent progress flags like `samples_collected`, `qc_performed`, or `de_performed`.
7. Generate a structured simulated output for the chosen action.
8. If there were soft violations, degrade output quality (×0.5) and attach warnings.
9. Propagate artifacts back into latent state, such as discovered DE genes or cluster names.
10. Compute decomposed reward from state transition plus output quality.
11. If the episode is ending, compute terminal reward from completeness and conclusion calibration.
12. Return an observation that exposes the visible summary but not the hidden truth.

## Action costs

Each action deducts from the episode's budget and time. Computational steps also accrue compute hours.

| Action | Budget | Time (days) |
|---|---|---|
| `sequence_cells` | $15,000 | 5 |
| `prepare_library` | $8,000 | 3 |
| `collect_sample` | $5,000 | 7 |
| `validate_marker` | $5,000 | 14 |
| `culture_cells` | $3,000 | 14 |
| `perturb_gene` | $2,000 | 3 |
| `perturb_compound` | $1,000 | 2 |
| `select_cohort` | $500 | 1 |
| `run_qc` | $100 | 0.5 |
| `integrate_batches` | $300 | 1 |
| `regulatory_network_inference` | $200 | 1 |
| `cluster_cells` | $150 | 0.5 |
| `differential_expression`, `trajectory_analysis`, `pathway_enrichment` | $100–200 | 0.5–1 |
| `filter_data`, `normalize_data`, `marker_selection` | $50–100 | 0.25–0.5 |
| `synthesize_conclusion`, `design_followup_experiment`, `request_subagent_review` | $0 | 0.25–0.5 |

## Why this is useful

This environment is trying to model a realistic scientific planning loop rather than a toy decision problem:

- actions have prerequisites
- outputs are noisy and imperfect
- budget and time matter
- not every correct-looking answer is well supported
- final conclusions are scored against hidden ground truth

That makes it suitable for:

- agent planning benchmarks
- RL experiments on long-horizon scientific reasoning
- literature-grounded evaluation
- comparing structured policies against LLM-driven planners
