"""Deterministic step-level grader for multi-turn agent evaluations."""

import json
import re
import sys


def grade_step(result_path: str, expected_path: str) -> float:
    """Score an agent's single-step output against expected results.

    Runs up to 5 checks: stage transition, state field extraction,
    required tools called, disallowed tools not called, and disallowed
    patterns not in response.

    Returns sum(checks) / len(checks) if checks exist, else 0.0.
    """
    with open(result_path) as f:
        result = json.load(f)
    with open(expected_path) as f:
        expected = json.load(f)

    checks: list[bool] = []

    # 1. Stage transition correct?
    if "expected_stage" in expected:
        checks.append(result["new_stage"] == expected["expected_stage"])

    # 2. State fields extracted correctly?
    for field, value in expected.get("expected_state", {}).items():
        checks.append(result["state_updates"].get(field) == value)

    # 3. Required tool called?
    if "required_tools" in expected:
        called = {tc["name"] for tc in result.get("tool_calls", [])}
        for tool in expected["required_tools"]:
            checks.append(tool in called)

    # 4. Disallowed tool NOT called?
    if "disallowed_tools" in expected:
        called = {tc["name"] for tc in result.get("tool_calls", [])}
        for tool in expected["disallowed_tools"]:
            checks.append(tool not in called)

    # 5. Response doesn't contain disallowed content
    response = result.get("agent_response", "")
    for pattern in expected.get("disallowed_patterns", []):
        checks.append(not re.search(pattern, response, re.IGNORECASE))

    return sum(checks) / len(checks) if checks else 0.0


if __name__ == "__main__":
    score = grade_step(sys.argv[1], sys.argv[2])
    with open("/logs/verifier/reward.txt", "w") as f:
        f.write(str(score))
