"""Trajectory-level grader for multi-turn agent evaluations.

Uses a hybrid approach: 60% deterministic checks + 40% LLM-as-judge.
"""

import asyncio
import difflib
import json
import re
import sys

from langchain_openai import ChatOpenAI


def check_no_repetition(agent_messages: list[str]) -> float:
    """Score how unique consecutive agent messages are.

    Uses SequenceMatcher to detect near-duplicate consecutive messages.
    Returns 1.0 for all unique, 0.0 for all repeated.
    """
    if len(agent_messages) <= 1:
        return 1.0

    num_repetitions = 0
    for i in range(len(agent_messages) - 1):
        ratio = difflib.SequenceMatcher(None, agent_messages[i], agent_messages[i + 1]).ratio()
        if ratio > 0.8:
            num_repetitions += 1

    return 1.0 - (num_repetitions / (len(agent_messages) - 1))


async def llm_judge(conversation: list[dict], rubric: str) -> float:
    """Use an LLM to evaluate a conversation against a rubric.

    Returns a float between 0.0 and 1.0.
    """
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

    formatted_turns = "\n".join(f"User: {turn['user']}\nAgent: {turn['agent']}" for turn in conversation)

    response = await llm.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "You are an evaluation judge. Rate the following conversation "
                    "on a scale from 0.0 to 1.0 based on the rubric below. "
                    "Respond with ONLY a decimal number between 0.0 and 1.0.\n\n"
                    f"Rubric: {rubric}"
                ),
            },
            {
                "role": "user",
                "content": formatted_turns,
            },
        ]
    )

    text = response.content if isinstance(response.content, str) else str(response.content)
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return 0.0


async def _grade_trajectory_async(result_path: str, expected_path: str) -> float:
    """Async implementation of trajectory grading."""
    with open(result_path) as f:
        result = json.load(f)
    with open(expected_path) as f:
        expected = json.load(f)

    scores: dict[str, float] = {}

    # --- Deterministic checks (weighted 60%) ---

    # Correctness: account created with correct fields?
    final = result["final_state"]
    exp_final = expected["expected_final_state"]
    field_checks = []
    for field in ["user_name", "email", "plan", "account_created"]:
        field_checks.append(final.get(field) == exp_final.get(field))
    scores["correctness"] = sum(field_checks) / len(field_checks)

    # Efficiency: completed within turn budget?
    max_turns = expected.get("max_turns", 15)
    actual_turns = len(result["turns"])
    scores["efficiency"] = 1.0 if actual_turns <= max_turns else max(0, 1 - (actual_turns - max_turns) / 5)

    # No repeated questions?
    agent_messages = [t["agent"] for t in result["turns"]]
    scores["no_repetition"] = check_no_repetition(agent_messages)

    # --- LLM-as-judge checks (weighted 40%) ---

    scores["tone"] = await llm_judge(
        conversation=result["turns"],
        rubric="Rate 0-1: Was the agent professional, warm, and helpful throughout? "
        "Deduct for robotic/scripted feel, condescension, or excessive formality.",
    )

    scores["edge_handling"] = await llm_judge(
        conversation=result["turns"],
        rubric="Rate 0-1: When the user did something unexpected (off-topic, changed mind, "
        "gave ambiguous input), did the agent handle it smoothly without losing context?",
    )

    # Weighted final score
    deterministic = 0.6 * (0.5 * scores["correctness"] + 0.3 * scores["efficiency"] + 0.2 * scores["no_repetition"])
    qualitative = 0.4 * (0.5 * scores["tone"] + 0.5 * scores["edge_handling"])

    return deterministic + qualitative


def grade_trajectory(result_path: str, expected_path: str) -> float:
    """Score a full multi-turn agent trajectory.

    Combines deterministic checks (60%) with LLM-as-judge evaluation (40%).
    """
    return asyncio.run(_grade_trajectory_async(result_path, expected_path))


if __name__ == "__main__":
    score = grade_trajectory(sys.argv[1], sys.argv[2])
    with open("/logs/verifier/reward.txt", "w") as f:
        f.write(str(score))
