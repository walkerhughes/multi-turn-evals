"""Unit tests for scenario generator script."""

import json
import tomllib
from pathlib import Path

import pytest

from src.scripts.generate_scenarios import (
    _determine_expected_stage,
    _determine_expected_state,
    _determine_required_tools,
    build_history_for_stage,
    generate_step_task,
)


@pytest.mark.unit
def test_build_history_alternates_roles() -> None:
    """Verify history messages have alternating roles."""
    history = build_history_for_stage(
        "collect_email", "Walker Thompson", "walker@example.com", "free"
    )

    # Extract roles
    roles = [msg["role"] for msg in history]

    # Check alternation
    for i in range(len(roles) - 1):
        assert roles[i] != roles[i + 1], f"Roles should alternate at index {i}"

    # Should start with assistant greeting
    assert roles[0] == "assistant"


@pytest.mark.unit
def test_build_history_stages() -> None:
    """Verify correct history for different stages."""
    # collect_name stage
    history = build_history_for_stage("collect_name", "Walker", "", None)
    assert len(history) == 2  # greeting + user response
    assert history[0]["role"] == "assistant"
    assert history[1]["role"] == "user"

    # collect_email stage
    history = build_history_for_stage(
        "collect_email", "Walker", "walker@example.com", None
    )
    assert len(history) == 4  # greeting + name exchange + email prompt + user response
    assert "NAME: Walker" in history[2]["content"]

    # verify_email stage
    history = build_history_for_stage(
        "verify_email", "Walker", "walker@example.com", None
    )
    assert len(history) == 6
    assert "verification code" in history[4]["content"].lower()

    # select_plan stage
    history = build_history_for_stage(
        "select_plan", "Walker", "walker@example.com", "pro"
    )
    assert len(history) == 8
    assert "plan" in history[6]["content"].lower()

    # collect_preferences stage
    history = build_history_for_stage(
        "collect_preferences", "Walker", "walker@example.com", "pro"
    )
    assert len(history) == 10
    assert "PLAN: pro" in history[8]["content"]

    # confirm stage
    history = build_history_for_stage(
        "confirm", "Walker", "walker@example.com", "pro"
    )
    assert len(history) == 12
    assert "PREFERENCES" in history[10]["content"]


@pytest.mark.unit
def test_generate_step_task_creates_files(tmp_path: Path) -> None:
    """Verify task_dir has instruction.md, task.toml, expected.json."""
    task_dir = tmp_path / "test-task"
    generate_step_task("Walker", "walker@example.com", "free", "collect_email", task_dir)

    assert (task_dir / "instruction.md").exists()
    assert (task_dir / "task.toml").exists()
    assert (task_dir / "expected.json").exists()


@pytest.mark.unit
def test_generated_instruction_valid_json(tmp_path: Path) -> None:
    """Parse instruction.md, extract JSON from code block, verify valid."""
    task_dir = tmp_path / "test-task"
    generate_step_task("Walker", "walker@example.com", "free", "collect_email", task_dir)

    instruction_content = (task_dir / "instruction.md").read_text()

    # Extract JSON from code block
    assert instruction_content.startswith("```json\n")
    assert instruction_content.endswith("```\n")

    json_str = instruction_content.removeprefix("```json\n").removesuffix("```\n")
    data = json.loads(json_str)

    # Verify structure
    assert data["eval_mode"] == "step"
    assert data["current_stage"] == "collect_email"
    assert isinstance(data["history"], list)
    assert isinstance(data["accumulated_state"], dict)


@pytest.mark.unit
def test_generated_expected_valid_json(tmp_path: Path) -> None:
    """Verify expected.json is valid."""
    task_dir = tmp_path / "test-task"
    generate_step_task("Walker", "walker@example.com", "free", "collect_email", task_dir)

    expected_content = (task_dir / "expected.json").read_text()
    data = json.loads(expected_content)

    # Verify structure
    assert "expected_stage" in data
    assert "expected_state" in data
    assert "required_tools" in data
    assert "disallowed_tools" in data
    assert "disallowed_patterns" in data

    assert isinstance(data["required_tools"], list)
    assert isinstance(data["disallowed_tools"], list)
    assert isinstance(data["disallowed_patterns"], list)


@pytest.mark.unit
def test_generated_task_toml_valid(tmp_path: Path) -> None:
    """Verify task.toml has expected keys."""
    task_dir = tmp_path / "test-task"
    generate_step_task("Walker", "walker@example.com", "free", "collect_email", task_dir)

    toml_bytes = (task_dir / "task.toml").read_bytes()
    data = tomllib.loads(toml_bytes.decode())

    # Verify structure
    assert "task" in data
    assert "timeout_sec" in data["task"]
    assert data["task"]["timeout_sec"] == 60

    assert "metadata" in data["task"]
    assert data["task"]["metadata"]["category"] == "step-level"
    assert data["task"]["metadata"]["subcategory"] == "generated"


@pytest.mark.unit
def test_expected_outcomes_match_params() -> None:
    """Verify expected stage/state match the input parameters."""
    # Valid name should advance from collect_name to collect_email
    assert (
        _determine_expected_stage("collect_name", "Walker", "", None) == "collect_email"
    )

    # Empty name should stay at collect_name
    assert _determine_expected_stage("collect_name", "", "", None) == "collect_name"

    # Valid email should advance from collect_email to verify_email
    assert (
        _determine_expected_stage("collect_email", "Walker", "walker@example.com", None)
        == "verify_email"
    )

    # Invalid email should stay at collect_email
    assert (
        _determine_expected_stage("collect_email", "Walker", "bad-email", None)
        == "collect_email"
    )

    # Valid plan should advance from select_plan to collect_preferences
    assert (
        _determine_expected_stage("select_plan", "Walker", "walker@example.com", "pro")
        == "collect_preferences"
    )

    # Invalid plan should stay at select_plan
    assert (
        _determine_expected_stage(
            "select_plan", "Walker", "walker@example.com", "invalid_plan"
        )
        == "select_plan"
    )

    # State updates
    state = _determine_expected_state("collect_name", "Walker", "", None)
    assert state["user_name"] == "Walker"

    state = _determine_expected_state("collect_email", "Walker", "walker@example.com", None)
    assert state["email"] == "walker@example.com"

    state = _determine_expected_state("select_plan", "Walker", "walker@example.com", "pro")
    assert state["plan"] == "pro"

    # Required tools
    tools = _determine_required_tools("collect_email", "walker@example.com")
    assert "validate_email" in tools

    tools = _determine_required_tools("verify_email", "walker@example.com")
    assert "send_verification_code" in tools
