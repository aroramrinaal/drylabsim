"""
Baseline inference script for the DryLabSim RL environment.
===================================
MANDATORY env vars before submitting:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    ENV_URL        The base URL of the deployed environment server.

Defaults:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
    ENV_URL      = os.getenv("ENV_URL",      "http://localhost:8000")

STDOUT FORMAT (mandatory — automated grader parses these lines):
    [START] task=<task_name> env=drylabsim model=<model_name>
    [STEP]  step=<n> action=<action_type> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import json
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000").rstrip("/")

ENV_NAME         = "drylabsim"
MAX_STEPS        = 28          # env hard-limit is 30, leave 2 steps as buffer
TEMPERATURE      = 0.2
MAX_TOKENS       = 1024
SUCCESS_THRESHOLD = 0.35       # score >= this counts as success for [END]

# maps task_id (as registered in openenv.yaml) to scenario_name
TASKS: Dict[str, str] = {
    "task_easy":   "cardiac_disease_de",
    "task_hard":   "perturbation_immune",
    "task_expert": "venetoclax_resistance_multiclone",
}

# ordered pipeline for each difficulty level
# the agent LLM decides the exact action at each step, but we give it a
# recommended next action derived from what's been completed so far
_PIPELINE_EASY = [
    "collect_sample",
    "prepare_library",
    "sequence_cells",
    "run_qc",
    "filter_data",
    "normalize_data",
    "cluster_cells",
    "differential_expression",
    "pathway_enrichment",
    "marker_selection",
    "validate_marker",
    "synthesize_conclusion",
]

_PIPELINE_HARD = [
    "collect_sample",
    "prepare_library",
    "sequence_cells",
    "run_qc",
    "filter_data",
    "normalize_data",
    "cluster_cells",
    "differential_expression",
    "pathway_enrichment",
    "regulatory_network_inference",
    "marker_selection",
    "validate_marker",
    "synthesize_conclusion",
]

_PIPELINE_EXPERT = [
    "collect_sample",
    "prepare_library",
    "sequence_cells",
    "run_qc",
    "filter_data",
    "normalize_data",
    "integrate_batches",       # critical for expert: resolves minor clone
    "cluster_cells",
    "differential_expression",
    "pathway_enrichment",
    "trajectory_analysis",     # expert requires this before conclusion
    "regulatory_network_inference",  # expert requires this before conclusion
    "marker_selection",
    "validate_marker",
    "synthesize_conclusion",
]

TASK_PIPELINES: Dict[str, List[str]] = {
    "task_easy":   _PIPELINE_EASY,
    "task_hard":   _PIPELINE_HARD,
    "task_expert": _PIPELINE_EXPERT,
}

# milestone flags we can read from pipeline_history to infer completeness
_MILESTONE_FLAGS = [
    "collect_sample",
    "sequence_cells",
    "run_qc",
    "filter_data",
    "normalize_data",
    "differential_expression",
    "synthesize_conclusion",
]

_OPTIONAL_MILESTONE_FLAGS = [
    "cluster_cells",
    "pathway_enrichment",
    "marker_selection",
    "validate_marker",
    "trajectory_analysis",
    "regulatory_network_inference",
]

# ---------------------------------------------------------------------------
# stdout logging helpers (exact format the automated grader expects)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_oneline = action.replace("\n", " ").replace("\r", "")
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_oneline} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(scenario_name: str) -> dict:
    """POST /reset with the scenario name, return full response."""
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"scenario_name": scenario_name},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(session_id: str, action: dict) -> dict:
    """POST /step with the structured action, return full response."""
    resp = requests.post(
        f"{ENV_URL}/step",
        json={
            "session_id": session_id,
            "action": action,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def env_state(session_id: str) -> dict:
    """GET /state for the current session."""
    resp = requests.get(
        f"{ENV_URL}/state",
        params={"session_id": session_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# score approximation from the final observation
# (grade_episode needs hidden latent; we reconstruct what we can from obs)
# ---------------------------------------------------------------------------

def _infer_completeness_from_history(pipeline_history: List[dict]) -> float:
    """
    Reconstruct pipeline completeness from the pipeline_history list in obs.
    Mirrors the scoring logic in server/grader/pipeline.py.
    """
    completed = {step["action_type"] for step in pipeline_history if step.get("success")}

    core_hits = sum(1 for m in _MILESTONE_FLAGS if m in completed)
    core_score = core_hits / len(_MILESTONE_FLAGS)

    optional_hits = sum(1 for m in _OPTIONAL_MILESTONE_FLAGS if m in completed)
    optional_score = optional_hits / len(_OPTIONAL_MILESTONE_FLAGS)

    return 0.7 * core_score + 0.3 * optional_score


def _infer_biology_score(obs: dict) -> float:
    """
    Rough biology proxy from observable fields only.
    We cannot call gene_index.py without the hidden ground truth, so we use
    a heuristic that is monotonically correlated with true biology score:
      - were markers discovered? (+0.40 weight)
      - were mechanisms identified? (+0.35 weight)
      - was a structured conclusion submitted with pathways? (+0.25 weight)
    Each sub-score is proportional to how many non-empty items were produced.
    """
    discovered_markers = obs.get("discovered_markers", [])
    candidate_mechanisms = obs.get("candidate_mechanisms", [])
    conclusions = obs.get("conclusions", [])

    # marker sub-score: credit for each distinct marker found, cap at 1
    marker_score = min(1.0, len(discovered_markers) / max(len(discovered_markers), 4))
    if not discovered_markers:
        marker_score = 0.0

    # mechanism sub-score
    mech_score = min(1.0, len(candidate_mechanisms) / 2.0) if candidate_mechanisms else 0.0

    # conclusion sub-score: full credit if a structured conclusion was filed
    conclusion_score = 0.0
    if conclusions:
        # check if at least one conclusion has structured fields (top_markers, causal_mechanisms)
        for c in conclusions:
            top_markers = c.get("top_markers", [])
            causal_mechs = c.get("causal_mechanisms", [])
            if top_markers or causal_mechs:
                conclusion_score = 1.0
                break
        if conclusion_score == 0.0:
            conclusion_score = 0.5  # unstructured conclusion still gets partial credit

    bio = 0.40 * marker_score + 0.35 * mech_score + 0.25 * conclusion_score
    return min(1.0, bio)


def grade_from_obs(obs: dict) -> float:
    """
    Approximate the grade_episode score purely from observable fields.
    Mirrors weights in server/grader/grade.py:
      pipeline completeness: 0.30
      biology score:         0.55
      efficiency:            0.15
    Returns a value clamped to [0.01, 0.99].
    """
    pipeline_history = obs.get("pipeline_history", [])
    resource_usage = obs.get("resource_usage", {})

    completeness = _infer_completeness_from_history(pipeline_history)

    biology = _infer_biology_score(obs)

    budget_remaining = resource_usage.get("budget_remaining", 0.0)
    budget_used = resource_usage.get("budget_used", 1.0)
    budget_total = budget_remaining + budget_used
    time_remaining = resource_usage.get("time_remaining_days", 0.0)
    time_used = resource_usage.get("time_used_days", 1.0)
    time_total = time_remaining + time_used

    budget_eff = budget_remaining / max(budget_total, 1.0)
    time_eff = time_remaining / max(time_total, 1.0)
    efficiency = 0.5 * budget_eff + 0.5 * time_eff

    raw = 0.30 * completeness + 0.55 * biology + 0.15 * efficiency
    return max(0.01, min(0.99, raw))

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert computational biologist operating a dry-lab single-cell RNA
    sequencing pipeline. You will be given the current state of a biological
    experiment (task description, completed steps, available tools, resource usage)
    and a recommended next action.

    Your job is to output a single JSON object representing the next ExperimentAction.

    The JSON must have this shape:
    {
      "action_type": "<one of the action types listed below>",
      "method": "<optional: specific tool/method name, e.g. scanpy.tl.leiden>",
      "parameters": { <optional: key-value pairs relevant to the action> },
      "confidence": <float 0.0-1.0>
    }

    Valid action_type values:
      collect_sample, select_cohort, prepare_library, culture_cells,
      perturb_gene, perturb_compound, sequence_cells, run_qc, filter_data,
      normalize_data, integrate_batches, cluster_cells, differential_expression,
      trajectory_analysis, pathway_enrichment, regulatory_network_inference,
      marker_selection, validate_marker, design_followup,
      request_subagent_review, synthesize_conclusion

    For synthesize_conclusion, your parameters must include:
    {
      "claims": [
        {
          "claim": "<one sentence summary of finding>",
          "claim_type": "causal",
          "confidence": <float>,
          "top_markers": ["GENE1", "GENE2"],
          "causal_mechanisms": ["mechanism description"],
          "predicted_pathways": {"pathway_name": <activity 0.0-1.0>},
          "mechanism_confidence": {"mechanism description": <float>}
        }
      ]
    }

    Output ONLY the JSON object — no explanation, no markdown fences.
""").strip()


def build_prompt(obs: dict, recommended_action: str, step: int) -> str:
    """Build the user prompt from current observation state."""
    task = obs.get("task", {})
    problem = task.get("problem_statement", "unknown task")
    conditions = task.get("conditions", [])
    tissue = task.get("tissue", "unknown")
    budget_info = obs.get("resource_usage", {})
    discovered_markers = obs.get("discovered_markers", [])
    candidate_mechanisms = obs.get("candidate_mechanisms", [])
    latest_output = obs.get("latest_output", {})
    violations = obs.get("rule_violations", [])

    # summarize recent pipeline history (last 5 steps)
    history = obs.get("pipeline_history", [])
    recent_history = history[-5:] if len(history) > 5 else history
    history_lines = []
    for h in recent_history:
        status = "SUCCESS" if h.get("success") else "FAILED"
        history_lines.append(
            f"  step {h['step_index']}: {h['action_type']} [{status}] — {h.get('output_summary', '')}"
        )
    history_str = "\n".join(history_lines) if history_lines else "  none yet"

    latest_summary = ""
    if latest_output:
        latest_summary = (
            f"Latest output type: {latest_output.get('output_type', 'none')}\n"
            f"Latest summary: {latest_output.get('summary', '')}\n"
        )
        data_keys = list((latest_output.get("data") or {}).keys())
        if data_keys:
            latest_summary += f"Data fields available: {data_keys}\n"

    violations_str = "\n".join(f"  WARNING: {v}" for v in violations) if violations else "  none"

    return textwrap.dedent(f"""
        Step {step} — Biological Experiment Planning

        Task: {problem}
        Tissue: {tissue}
        Conditions: {conditions}

        Resource usage:
          budget used:     ${budget_info.get('budget_used', 0):,.0f}
          budget remaining: ${budget_info.get('budget_remaining', 100000):,.0f}
          time used:       {budget_info.get('time_used_days', 0):.0f} days
          time remaining:  {budget_info.get('time_remaining_days', 180):.0f} days

        Recent pipeline history:
        {history_str}

        {latest_summary}
        Discovered markers so far: {discovered_markers}
        Candidate mechanisms so far: {candidate_mechanisms}

        Rule violations from last action:
        {violations_str}

        RECOMMENDED NEXT ACTION: {recommended_action}

        Output a single JSON ExperimentAction to execute this step.
        If recommended action is synthesize_conclusion, include structured claims
        with top_markers and causal_mechanisms fields populated from what was discovered.
    """).strip()


def call_llm(client: OpenAI, obs: dict, recommended_action: str, step: int) -> dict:
    """
    Ask the LLM what action to take. Returns a parsed action dict.
    Falls back to a hardcoded action for the recommended step on any error.
    """
    user_prompt = build_prompt(obs, recommended_action, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # strip markdown fences if model wrapped the json
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        parsed = json.loads(raw)
        if "action_type" not in parsed:
            raise ValueError("missing action_type in response")
        return parsed

    except Exception as exc:
        print(f"[DEBUG] llm call failed at step {step}: {exc}", flush=True)
        return _fallback_action(recommended_action, obs)


def _fallback_action(recommended_action: str, obs: dict) -> dict:
    """
    Hardcoded fallback action dict for each action type.
    Used when the LLM fails or returns unparseable output.
    """
    discovered_markers = obs.get("discovered_markers", [])
    candidate_mechanisms = obs.get("candidate_mechanisms", [])
    task = obs.get("task", {})

    base: Dict[str, Any] = {
        "action_type": recommended_action,
        "method": None,
        "parameters": {},
        "confidence": 0.7,
    }

    if recommended_action == "collect_sample":
        base["parameters"] = {"n_samples": 6}
        base["method"] = "10x_chromium"

    elif recommended_action == "prepare_library":
        base["method"] = "10x_chromium"

    elif recommended_action == "sequence_cells":
        base["method"] = "NovaSeq"

    elif recommended_action == "normalize_data":
        base["method"] = "scanpy.pp.normalize_total"

    elif recommended_action == "integrate_batches":
        base["method"] = "harmony"

    elif recommended_action == "cluster_cells":
        base["method"] = "scanpy.tl.leiden"
        base["parameters"] = {"resolution": 0.5}

    elif recommended_action == "differential_expression":
        conditions = task.get("conditions", ["disease", "healthy"])
        comparison = f"{conditions[-1]}_vs_{conditions[0]}" if len(conditions) >= 2 else "disease_vs_healthy"
        base["method"] = "scanpy.tl.rank_genes_groups"
        base["parameters"] = {"comparison": comparison}

    elif recommended_action == "pathway_enrichment":
        base["method"] = "gseapy.prerank"

    elif recommended_action == "trajectory_analysis":
        base["method"] = "monocle3"

    elif recommended_action == "regulatory_network_inference":
        base["method"] = "SCENIC"

    elif recommended_action == "marker_selection":
        base["method"] = "scanpy.tl.rank_genes_groups"

    elif recommended_action == "validate_marker":
        marker = discovered_markers[0] if discovered_markers else "NPPA"
        base["parameters"] = {"marker": marker, "assay": "qPCR"}

    elif recommended_action == "synthesize_conclusion":
        problem = task.get("problem_statement", "")
        top_markers = discovered_markers[:4] if discovered_markers else ["NPPA"]
        mechanisms = candidate_mechanisms[:2] if candidate_mechanisms else ["pathway dysregulation"]
        base["parameters"] = {
            "claims": [
                {
                    "claim": f"Computational analysis identified key molecular drivers: {', '.join(top_markers[:2])}",
                    "claim_type": "causal",
                    "confidence": 0.75,
                    "top_markers": top_markers,
                    "causal_mechanisms": mechanisms,
                    "predicted_pathways": {},
                    "mechanism_confidence": {m: 0.70 for m in mechanisms},
                }
            ]
        }

    return base

# ---------------------------------------------------------------------------
# pipeline driver: determines recommended next action from obs state
# ---------------------------------------------------------------------------

def _get_completed_actions(pipeline_history: List[dict]) -> set:
    return {step["action_type"] for step in pipeline_history if step.get("success")}


def recommend_next_action(pipeline: List[str], pipeline_history: List[dict]) -> str:
    """
    Walk the recommended pipeline list and return the first action not yet
    successfully completed. Falls back to synthesize_conclusion at the end.
    """
    completed = _get_completed_actions(pipeline_history)
    for action in pipeline:
        if action not in completed:
            return action
    return "synthesize_conclusion"

# ---------------------------------------------------------------------------
# single episode runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str, scenario_name: str) -> Tuple[float, List[float]]:
    """
    Run one full episode for the given scenario.
    Returns (final_score, list_of_step_rewards).
    """
    log_start(task=task_id, model=MODEL_NAME)

    pipeline = TASK_PIPELINES.get(task_id, _PIPELINE_HARD)
    rewards: List[float] = []
    step_count = 0
    final_score = 0.0
    success = False
    last_obs = {}

    try:
        # --- reset ---
        reset_resp = env_reset(scenario_name)
        obs = reset_resp.get("observation") or reset_resp
        session_id = (obs.get("metadata") or {}).get("episode_id", "")
        if not session_id:
            # try top-level session_id key (some openenv versions surface it here)
            session_id = reset_resp.get("session_id", "default")
        done = reset_resp.get("done", False)
        last_obs = obs

        print(
            f"[DEBUG] task={task_id} scenario={scenario_name} session_id={session_id}",
            flush=True,
        )

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            pipeline_history = obs.get("pipeline_history", [])
            recommended = recommend_next_action(pipeline, pipeline_history)

            action_dict = call_llm(client, obs, recommended, step)

            try:
                step_resp = env_step(session_id, action_dict)
            except Exception as http_exc:
                print(f"[DEBUG] step http error: {http_exc}", flush=True)
                rewards.append(0.0)
                log_step(step=step, action=str(action_dict.get("action_type", "unknown")),
                         reward=0.0, done=False, error=str(http_exc))
                continue

            obs = step_resp.get("observation") or step_resp
            reward = float(step_resp.get("reward") or 0.0)
            done = step_resp.get("done", False)
            last_obs = obs

            step_count = step
            rewards.append(reward)

            violations = obs.get("rule_violations", [])
            error_msg = "; ".join(violations) if violations else None

            log_step(
                step=step,
                action=action_dict.get("action_type", "unknown"),
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

    except Exception as exc:
        print(f"[DEBUG] episode error for task={task_id}: {exc}", flush=True)
        if not rewards:
            rewards = [0.0]

    # compute the final score from the last observation we have
    if last_obs:
        final_score = grade_from_obs(last_obs)
    else:
        final_score = 0.01

    # clamp per hackathon requirement
    final_score = max(0.01, min(0.99, final_score))

    success = final_score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=step_count, score=final_score, rewards=rewards)

    return final_score, rewards

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[DEBUG] env={ENV_URL} model={MODEL_NAME}", flush=True)

    results = {}
    for task_id, scenario_name in TASKS.items():
        score, rewards = run_task(client, task_id, scenario_name)
        results[task_id] = {"score": score, "rewards": rewards}

    # summary table
    print("\n" + "=" * 60, flush=True)
    print(f"{'task':<15} {'score':>8}  rewards", flush=True)
    print("-" * 60, flush=True)
    for task_id, data in results.items():
        rewards_str = ", ".join(f"{r:.2f}" for r in data["rewards"])
        print(f"{task_id:<15} {data['score']:>8.3f}  [{rewards_str}]", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
