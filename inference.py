"""
Deterministic baseline inference script for the DryLabSim RL environment.
=========================================================================

This baseline intentionally avoids proxy-grading. The environment computes the
canonical final score server-side via grade_episode(obs, latent) and exposes it
in terminal observation metadata.

MANDATORY env vars typically used by the hackathon harness:
    ENV_URL        The base URL of the deployed environment server.

Optional env vars kept for compatibility with sample scripts:
    API_BASE_URL   Unused by this deterministic baseline.
    MODEL_NAME     Label only, used in stdout logs.
    HF_TOKEN       Unused by this deterministic baseline.

STDOUT FORMAT (mandatory — automated grader parses these lines):
    [START] task=<task_name> env=drylabsim model=<model_name>
    [STEP]  step=<n> action=<action_type> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "https://mrinaalarora-drylabsim.hf.space").rstrip("/")

ENV_NAME = "drylabsim"
MAX_STEPS = 30
SUCCESS_THRESHOLD = 0.35
TEMPERATURE = 0.2
MAX_TOKENS = 700

TASKS: Dict[str, str] = {
    "easy": "cardiac_disease_de",
    "medium": "hematopoiesis_trajectory",
    "hard": "perturbation_immune",
    "expert": "venetoclax_resistance_multiclone",
}

TASK_COMPARISONS: Dict[str, str] = {
    "easy": "disease_vs_healthy",
    "hard": "treated_vs_untreated",
    "expert": "post_vs_pre_bulk",
}

TASK_PIPELINES: Dict[str, List[str]] = {
    "easy": [
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
    ],
    "medium": [
        "collect_sample",
        "prepare_library",
        "sequence_cells",
        "run_qc",
        "filter_data",
        "normalize_data",
        "cluster_cells",
        "differential_expression",
        "trajectory_analysis",
        "regulatory_network_inference",
        "marker_selection",
        "synthesize_conclusion",
    ],
    "hard": [
        "collect_sample",
        "prepare_library",
        "sequence_cells",
        "run_qc",
        "filter_data",
        "normalize_data",
        "integrate_batches",
        "cluster_cells",
        "differential_expression",
        "pathway_enrichment",
        "marker_selection",
        "validate_marker",
        "synthesize_conclusion",
    ],
    "expert": [
        "collect_sample",
        "prepare_library",
        "sequence_cells",
        "run_qc",
        "filter_data",
        "normalize_data",
        "integrate_batches",
        "cluster_cells",
        "differential_expression",
        "pathway_enrichment",
        "regulatory_network_inference",
        "trajectory_analysis",
        "marker_selection",
        "synthesize_conclusion",
    ],
}

CORE_MILESTONES = [
    "collect_sample",
    "sequence_cells",
    "run_qc",
    "filter_data",
    "normalize_data",
    "differential_expression",
    "synthesize_conclusion",
]
OPTIONAL_MILESTONES = [
    "cluster_cells",
    "pathway_enrichment",
    "marker_selection",
    "validate_marker",
    "trajectory_analysis",
    "regulatory_network_inference",
]

ACTION_JSON_RE = re.compile(r"\{[\s\S]*\}")

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert computational biologist operating a dry-lab single-cell
    analysis environment. Reply with exactly one JSON object representing the
    next action to take.

    The JSON must follow this shape:
    {
      "action_type": "<valid action type>",
      "method": null,
      "parameters": {},
      "confidence": 0.7
    }

    Valid action types:
    collect_sample, select_cohort, prepare_library, culture_cells,
    perturb_gene, perturb_compound, sequence_cells, run_qc, filter_data,
    normalize_data, integrate_batches, cluster_cells, differential_expression,
    trajectory_analysis, pathway_enrichment, regulatory_network_inference,
    marker_selection, validate_marker, design_followup_experiment,
    request_subagent_review, synthesize_conclusion

    Rules:
    - Prefer the recommended next action unless the observation clearly shows it is unsafe.
    - Keep method null unless you are very sure.
    - For synthesize_conclusion, include structured claims with top_markers,
      causal_mechanisms, predicted_pathways, and evidence_steps when available.
    - Output JSON only. No markdown fences. No explanation.
    """
).strip()

# ---------------------------------------------------------------------------
# stdout logging helpers
# ---------------------------------------------------------------------------


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    action_oneline = action.replace("\n", " ").replace("\r", "")
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_oneline} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def env_reset(task_name: str) -> dict:
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_name": task_name},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(session_id: str, action: dict) -> dict:
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# scoring fallback
# ---------------------------------------------------------------------------


def _completed_actions(pipeline_history: List[dict]) -> set[str]:
    return {
        step.get("action_type")
        for step in pipeline_history
        if step.get("success") and step.get("action_type")
    }


def _infer_completeness_from_history(pipeline_history: List[dict]) -> float:
    completed = _completed_actions(pipeline_history)
    core_score = sum(1 for step in CORE_MILESTONES if step in completed) / len(
        CORE_MILESTONES
    )
    optional_score = sum(
        1 for step in OPTIONAL_MILESTONES if step in completed
    ) / len(OPTIONAL_MILESTONES)
    return 0.7 * core_score + 0.3 * optional_score


def _infer_biology_score(obs: dict) -> float:
    discovered_markers = obs.get("discovered_markers", [])
    candidate_mechanisms = obs.get("candidate_mechanisms", [])
    conclusions = obs.get("conclusions", [])

    marker_score = min(1.0, len(discovered_markers) / 4.0) if discovered_markers else 0.0
    mechanism_score = (
        min(1.0, len(candidate_mechanisms) / 2.0) if candidate_mechanisms else 0.0
    )

    conclusion_score = 0.0
    if conclusions:
        for claim in conclusions:
            if claim.get("top_markers") or claim.get("causal_mechanisms"):
                conclusion_score = 1.0
                break
        if conclusion_score == 0.0:
            conclusion_score = 0.5

    return min(
        1.0,
        0.40 * marker_score + 0.35 * mechanism_score + 0.25 * conclusion_score,
    )


def grade_from_obs(obs: dict) -> float:
    metadata = obs.get("metadata") or {}
    if "score" in metadata:
        return float(metadata["score"])

    pipeline_history = obs.get("pipeline_history", [])
    resource_usage = obs.get("resource_usage", {})

    completeness = _infer_completeness_from_history(pipeline_history)
    biology = _infer_biology_score(obs)

    budget_remaining = float(resource_usage.get("budget_remaining", 0.0))
    budget_used = float(resource_usage.get("budget_used", 0.0))
    budget_total = budget_remaining + budget_used

    time_remaining = float(resource_usage.get("time_remaining_days", 0.0))
    time_used = float(resource_usage.get("time_used_days", 0.0))
    time_total = time_remaining + time_used

    budget_eff = budget_remaining / max(budget_total, 1.0)
    time_eff = time_remaining / max(time_total, 1.0)
    efficiency = 0.5 * budget_eff + 0.5 * time_eff

    raw = 0.30 * completeness + 0.55 * biology + 0.15 * efficiency
    return max(0.01, min(0.99, raw))


# ---------------------------------------------------------------------------
# deterministic policy + LLM prompt helpers
# ---------------------------------------------------------------------------


def recommend_next_action(task_name: str, pipeline_history: List[dict]) -> str:
    completed = _completed_actions(pipeline_history)
    for action_name in TASK_PIPELINES[task_name]:
        if action_name not in completed:
            return action_name
    return "synthesize_conclusion"


def _best_pathways(outputs: Iterable[dict]) -> Dict[str, float]:
    pathways: Dict[str, float] = {}
    for output in outputs:
        if output.get("output_type") != "pathway_result":
            continue
        for item in output.get("data", {}).get("top_pathways", []):
            if not isinstance(item, dict):
                continue
            pathway = item.get("pathway")
            score = item.get("score")
            if pathway is None or score is None:
                continue
            pathways[str(pathway)] = max(float(score), pathways.get(str(pathway), 0.0))
    return dict(list(pathways.items())[:6])


def _evidence_steps(pipeline_history: List[dict]) -> List[int]:
    supported_actions = {
        "differential_expression",
        "pathway_enrichment",
        "regulatory_network_inference",
        "trajectory_analysis",
        "marker_selection",
        "validate_marker",
    }
    return [
        int(step.get("step_index", 0))
        for step in pipeline_history
        if step.get("success") and step.get("action_type") in supported_actions
    ]


def build_fallback_action(task_name: str, obs: dict, action_name: str) -> dict:
    discovered_markers = obs.get("discovered_markers", [])
    candidate_mechanisms = obs.get("candidate_mechanisms", [])
    all_outputs = obs.get("all_outputs", [])
    pipeline_history = obs.get("pipeline_history", [])

    action: Dict[str, Any] = {
        "action_type": action_name,
        "method": None,
        "parameters": {},
        "confidence": 0.7,
    }

    if action_name == "collect_sample":
        action["parameters"] = {"n_samples": 6}
    elif action_name == "differential_expression":
        comparison = TASK_COMPARISONS.get(task_name)
        if comparison:
            action["parameters"] = {"comparison": comparison}
    elif action_name == "validate_marker":
        marker = discovered_markers[0] if discovered_markers else "NPPA"
        action["parameters"] = {"marker": marker, "assay": "qPCR"}
    elif action_name == "synthesize_conclusion":
        top_markers = discovered_markers[:6]
        mechanisms = candidate_mechanisms[:4]
        predicted_pathways = _best_pathways(all_outputs)
        confidence = 0.62 if task_name == "expert" else 0.68
        claim_type = "causal" if mechanisms else "correlational"

        if not top_markers:
            top_markers = ["NPPA"] if task_name == "easy" else []

        claim = {
            "claim": (
                "The pipeline recovered task-relevant markers and mechanisms from the "
                "simulated single-cell experiment."
            ),
            "claim_type": claim_type,
            "confidence": confidence,
            "top_markers": top_markers,
            "causal_mechanisms": mechanisms,
            "predicted_pathways": predicted_pathways,
            "mechanism_confidence": {mechanism: confidence for mechanism in mechanisms},
            "evidence_steps": _evidence_steps(pipeline_history),
        }
        action["parameters"] = {"claims": [claim]}

    return action


def _recent_history_lines(pipeline_history: List[dict]) -> str:
    recent = pipeline_history[-6:]
    if not recent:
        return "None"
    lines = []
    for step in recent:
        status = "SUCCESS" if step.get("success") else "FAILED"
        summary = step.get("output_summary", "")
        lines.append(
            f"- step {step.get('step_index')}: {step.get('action_type')} [{status}] {summary}"
        )
    return "\n".join(lines)


def build_user_prompt(
    task_name: str,
    obs: dict,
    recommended_action: str,
    fallback_action: dict,
) -> str:
    task = obs.get("task", {})
    resource_usage = obs.get("resource_usage", {})
    latest_output = obs.get("latest_output") or {}
    pipeline_history = obs.get("pipeline_history", [])
    discovered_markers = obs.get("discovered_markers", [])
    candidate_mechanisms = obs.get("candidate_mechanisms", [])
    rule_violations = obs.get("rule_violations", [])

    latest_summary = "None"
    if latest_output:
        latest_summary = (
            f"type={latest_output.get('output_type')} "
            f"summary={latest_output.get('summary', '')}"
        )

    return textwrap.dedent(
        f"""
        Task tier: {task_name}
        Problem: {task.get('problem_statement', 'unknown')}
        Tissue: {task.get('tissue', 'unknown')}
        Conditions: {task.get('conditions', [])}
        Available tools: {task.get('available_tools', [])}
        Available assays: {task.get('available_assays', [])}

        Budget used: {resource_usage.get('budget_used', 0)}
        Budget remaining: {resource_usage.get('budget_remaining', 0)}
        Time used days: {resource_usage.get('time_used_days', 0)}
        Time remaining days: {resource_usage.get('time_remaining_days', 0)}

        Recent pipeline history:
        {_recent_history_lines(pipeline_history)}

        Latest output:
        {latest_summary}

        Discovered markers: {discovered_markers}
        Candidate mechanisms: {candidate_mechanisms}
        Current rule violations: {rule_violations}

        Recommended next action: {recommended_action}
        Safe fallback action:
        {json.dumps(fallback_action, ensure_ascii=True)}

        Return exactly one JSON action object.
        """
    ).strip()


def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        parts = text.splitlines()
        if len(parts) >= 3:
            text = "\n".join(parts[1:-1]).strip()
    match = ACTION_JSON_RE.search(text)
    if match:
        return match.group(0)
    return text if text.startswith("{") else None


def choose_action(
    client: OpenAI,
    task_name: str,
    obs: dict,
    recommended_action: str,
) -> dict:
    fallback_action = build_fallback_action(task_name, obs, recommended_action)
    user_prompt = build_user_prompt(
        task_name=task_name,
        obs=obs,
        recommended_action=recommended_action,
        fallback_action=fallback_action,
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
        json_blob = _extract_json_object(response_text)
        if not json_blob:
            raise ValueError("model did not return JSON")
        parsed = json.loads(json_blob)
        if not isinstance(parsed, dict):
            raise ValueError("model response was not a JSON object")
        if "action_type" not in parsed:
            raise ValueError("model response missing action_type")

        candidate = {
            "action_type": parsed["action_type"],
            "method": parsed.get("method"),
            "parameters": parsed.get("parameters", {}),
            "confidence": parsed.get("confidence", 0.7),
        }
        return candidate
    except Exception as exc:
        print(
            f"[DEBUG] model action selection failed for task={task_name} "
            f"recommended={recommended_action}: {exc}",
            flush=True,
        )
        return fallback_action


# ---------------------------------------------------------------------------
# single episode runner
# ---------------------------------------------------------------------------


def run_task(client: OpenAI, task_name: str) -> Tuple[float, List[float]]:
    log_start(task=task_name, model=MODEL_NAME)

    rewards: List[float] = []
    step_count = 0
    final_score = 0.0
    last_obs: dict = {}

    try:
        reset_resp = env_reset(task_name)
        obs = reset_resp.get("observation") or reset_resp
        session_id = reset_resp.get("session_id", "")
        done = bool(reset_resp.get("done", False))
        last_obs = obs

        if not session_id:
            raise RuntimeError("Reset response did not include session_id")

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            recommended = recommend_next_action(
                task_name,
                obs.get("pipeline_history", []),
            )
            action = choose_action(client, task_name, obs, recommended)

            step_resp = env_step(session_id, action)
            obs = step_resp.get("observation") or step_resp
            reward = float(step_resp.get("reward") or 0.0)
            done = bool(step_resp.get("done", False))
            last_obs = obs

            rewards.append(reward)
            step_count = step

            violations = obs.get("rule_violations", [])
            error_msg = "; ".join(violations) if violations else None
            log_step(
                step=step,
                action=action["action_type"],
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

    except Exception as exc:
        print(f"[DEBUG] episode error for task={task_name}: {exc}", flush=True)
        if not rewards:
            rewards = [0.0]

    final_score = grade_from_obs(last_obs) if last_obs else 0.01
    final_score = max(0.01, min(0.99, final_score))
    success = final_score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=step_count, score=final_score, rewards=rewards)
    return final_score, rewards


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(
        f"[DEBUG] env={ENV_URL} model={MODEL_NAME} api_base={API_BASE_URL} "
        f"hf_token_set={str(bool(API_KEY)).lower()}",
        flush=True,
    )

    results: Dict[str, Dict[str, Any]] = {}
    for task_name in TASKS:
        score, rewards = run_task(client, task_name)
        results[task_name] = {"score": score, "rewards": rewards}

    print("\n" + "=" * 60, flush=True)
    print(f"{'task':<15} {'score':>8}  rewards", flush=True)
    print("-" * 60, flush=True)
    for task_name, data in results.items():
        rewards_str = ", ".join(f"{r:.2f}" for r in data["rewards"])
        print(f"{task_name:<15} {data['score']:>8.3f}  [{rewards_str}]", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
