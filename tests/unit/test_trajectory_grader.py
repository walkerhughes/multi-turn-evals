"""Unit tests for trajectory-level grader."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from grading.trajectory_grader import check_no_repetition, grade_trajectory


def _write_json(path, data):
    path.write_text(json.dumps(data))


def _make_turns(agent_messages, user_prefix="Hi"):
    """Build a turns list from agent message strings."""
    return [{"user": f"{user_prefix} {i}", "agent": msg, "stage": "onboarding"} for i, msg in enumerate(agent_messages)]


# Truly distinct messages — vary both length and content to stay below 0.8 similarity
_DISTINCT_5 = [
    "Welcome aboard!",
    "Name received, thanks.",
    "Got your email now.",
    "Plan selection confirmed.",
    "Account is all set!",
]
_DISTINCT_11 = [
    "Hello there!",
    "Name captured successfully.",
    "Email looks valid.",
    "Which plan works for you?",
    "Premium selected.",
    "Confirming your details now.",
    "Everything checks out.",
    "Setting up your profile.",
    "Almost finished here.",
    "One last verification step.",
    "Account creation complete!",
]
_DISTINCT_15 = [
    "Hi, welcome!",
    "What should I call you?",
    "Got it, thanks for the name.",
    "Now I need an email address.",
    "Email confirmed successfully.",
    "Let me show you our plans.",
    "Great choice on premium!",
    "Verifying your information.",
    "Profile is looking good.",
    "Adding final preferences.",
    "Security setup in progress.",
    "Nearly done with everything.",
    "Running a quick check.",
    "Wrapping things up now.",
    "Your account is ready!",
]
_DISTINCT_18 = [
    "Greetings!",
    "Tell me your full name please.",
    "Nice to meet you!",
    "What email shall I use?",
    "That email works perfectly.",
    "Here are the plan options.",
    "Premium is a solid pick.",
    "Let me verify everything.",
    "Checking your name again.",
    "Email domain validated.",
    "Plan pricing confirmed.",
    "Setting preferences next.",
    "Do you want notifications?",
    "Got your notification choice.",
    "Security question setup.",
    "Almost at the finish line.",
    "Final review in progress.",
    "Congratulations, all done!",
]

# Shorthand for tests that just need all-matching final state fields
_PERFECT_STATE = {
    "user_name": "A",
    "email": "[email protected]",
    "plan": "p",
    "account_created": True,
}


@pytest.mark.unit
class TestCheckNoRepetition:
    def test_all_unique_messages(self):
        messages = [
            "Welcome! What is your name?",
            "Great, and your email?",
            "Which plan would you like?",
            "Your account is created!",
        ]
        assert check_no_repetition(messages) == 1.0

    def test_exact_duplicate_consecutive(self):
        messages = [
            "What is your name?",
            "What is your name?",
            "What is your name?",
        ]
        score = check_no_repetition(messages)
        assert score == 0.0

    def test_near_identical_consecutive(self):
        messages = [
            "Could you please provide your email address?",
            "Could you please provide your email address please?",
            "Thanks, account created!",
        ]
        score = check_no_repetition(messages)
        assert score < 1.0

    def test_single_message(self):
        assert check_no_repetition(["Hello!"]) == 1.0

    def test_empty_messages(self):
        assert check_no_repetition([]) == 1.0


@pytest.mark.unit
class TestCorrectness:
    def test_partial_match(self, tmp_path):
        """2 of 4 fields match -> correctness = 0.5."""
        result = {
            "turns": _make_turns(["a", "b"]),
            "final_state": {
                "user_name": "Alice",
                "email": "[email protected]",
                "plan": "basic",
                "account_created": False,
            },
        }
        expected = {
            "expected_final_state": {
                "user_name": "Alice",
                "email": "[email protected]",
                "plan": "premium",
                "account_created": True,
            },
            "max_turns": 10,
        }
        _write_json(tmp_path / "result.json", result)
        _write_json(tmp_path / "expected.json", expected)

        with patch(
            "grading.trajectory_grader.llm_judge",
            new_callable=AsyncMock,
            return_value=1.0,
        ):
            score = grade_trajectory(
                str(tmp_path / "result.json"),
                str(tmp_path / "expected.json"),
            )

        # correctness = 2/4 = 0.5
        # efficiency = 1.0 (2 <= 10), no_rep = 1.0
        # det = 0.6 * (0.5*0.5 + 0.3*1.0 + 0.2*1.0) = 0.6 * 0.75 = 0.45
        # qual = 0.4 * (0.5*1.0 + 0.5*1.0) = 0.4
        # total = 0.85
        assert score == pytest.approx(0.85)

    def test_none_match(self, tmp_path):
        """No fields match -> correctness = 0.0."""
        result = {
            "turns": _make_turns(["a", "b"]),
            "final_state": {
                "user_name": "Wrong",
                "email": "wrong_email",
                "plan": "wrong",
                "account_created": False,
            },
        }
        expected = {
            "expected_final_state": {
                "user_name": "Alice",
                "email": "correct_email",
                "plan": "premium",
                "account_created": True,
            },
            "max_turns": 10,
        }
        _write_json(tmp_path / "result.json", result)
        _write_json(tmp_path / "expected.json", expected)

        with patch(
            "grading.trajectory_grader.llm_judge",
            new_callable=AsyncMock,
            return_value=1.0,
        ):
            score = grade_trajectory(
                str(tmp_path / "result.json"),
                str(tmp_path / "expected.json"),
            )

        # correctness = 0/4 = 0.0
        # efficiency = 1.0, no_rep = 1.0
        # det = 0.6 * (0.5*0.0 + 0.3*1.0 + 0.2*1.0) = 0.3
        # qual = 0.4
        # total = 0.7
        assert score == pytest.approx(0.7)


@pytest.mark.unit
class TestEfficiency:
    def test_within_budget(self, tmp_path):
        """actual_turns <= max_turns -> efficiency = 1.0."""
        result = {
            "turns": _make_turns(_DISTINCT_5),
            "final_state": _PERFECT_STATE,
        }
        expected = {
            "expected_final_state": _PERFECT_STATE,
            "max_turns": 8,
        }
        _write_json(tmp_path / "result.json", result)
        _write_json(tmp_path / "expected.json", expected)

        with patch(
            "grading.trajectory_grader.llm_judge",
            new_callable=AsyncMock,
            return_value=1.0,
        ):
            score = grade_trajectory(
                str(tmp_path / "result.json"),
                str(tmp_path / "expected.json"),
            )

        # correctness=1.0, efficiency=1.0, no_rep=1.0
        # det = 0.6, qual = 0.4 => 1.0
        assert score == pytest.approx(1.0)

    def test_over_budget(self, tmp_path):
        """actual_turns = max_turns + 3 -> efficiency = max(0, 1-3/5)=0.4."""
        result = {
            "turns": _make_turns(_DISTINCT_11),
            "final_state": _PERFECT_STATE,
        }
        expected = {
            "expected_final_state": _PERFECT_STATE,
            "max_turns": 8,
        }
        _write_json(tmp_path / "result.json", result)
        _write_json(tmp_path / "expected.json", expected)

        with patch(
            "grading.trajectory_grader.llm_judge",
            new_callable=AsyncMock,
            return_value=1.0,
        ):
            score = grade_trajectory(
                str(tmp_path / "result.json"),
                str(tmp_path / "expected.json"),
            )

        # correctness=1.0, efficiency=0.4, no_rep=1.0
        # det = 0.6 * (0.5 + 0.12 + 0.2) = 0.6 * 0.82 = 0.492
        # qual = 0.4
        # total = 0.892
        assert score == pytest.approx(0.892)

    def test_way_over_budget_clamped_to_zero(self, tmp_path):
        """actual_turns = max_turns+10 -> efficiency = max(0, 1-10/5)=0.0."""
        result = {
            "turns": _make_turns(_DISTINCT_18),
            "final_state": _PERFECT_STATE,
        }
        expected = {
            "expected_final_state": _PERFECT_STATE,
            "max_turns": 8,
        }
        _write_json(tmp_path / "result.json", result)
        _write_json(tmp_path / "expected.json", expected)

        with patch(
            "grading.trajectory_grader.llm_judge",
            new_callable=AsyncMock,
            return_value=1.0,
        ):
            score = grade_trajectory(
                str(tmp_path / "result.json"),
                str(tmp_path / "expected.json"),
            )

        # correctness=1.0, efficiency=0.0, no_rep=1.0
        # det = 0.6 * (0.5 + 0.0 + 0.2) = 0.6 * 0.7 = 0.42
        # qual = 0.4
        # total = 0.82
        assert score == pytest.approx(0.82)


@pytest.mark.unit
class TestGradeTrajectory:
    def test_combines_weights_correctly(self, tmp_path):
        """Mock llm_judge with specific values, verify weighted combo."""
        result = {
            "turns": _make_turns(["Hello", "Got it", "Done"]),
            "final_state": {
                "user_name": "Alice",
                "email": "[email protected]",
                "plan": "premium",
                "account_created": True,
            },
        }
        expected = {
            "expected_final_state": {
                "user_name": "Alice",
                "email": "[email protected]",
                "plan": "premium",
                "account_created": True,
            },
            "max_turns": 10,
        }
        _write_json(tmp_path / "result.json", result)
        _write_json(tmp_path / "expected.json", expected)

        # llm_judge called twice: tone=0.8, edge_handling=0.6
        mock_judge = AsyncMock(side_effect=[0.8, 0.6])
        with patch("grading.trajectory_grader.llm_judge", mock_judge):
            score = grade_trajectory(
                str(tmp_path / "result.json"),
                str(tmp_path / "expected.json"),
            )

        # correctness=1.0, efficiency=1.0, no_rep=1.0
        # det = 0.6
        # qual = 0.4 * (0.5*0.8 + 0.5*0.6) = 0.4 * 0.7 = 0.28
        # total = 0.88
        assert score == pytest.approx(0.88)

    def test_default_max_turns(self, tmp_path):
        """No max_turns in expected -> defaults to 15."""
        result = {
            "turns": _make_turns(_DISTINCT_15),
            "final_state": _PERFECT_STATE,
        }
        expected = {
            "expected_final_state": _PERFECT_STATE,
            # no max_turns -> defaults to 15
        }
        _write_json(tmp_path / "result.json", result)
        _write_json(tmp_path / "expected.json", expected)

        with patch(
            "grading.trajectory_grader.llm_judge",
            new_callable=AsyncMock,
            return_value=1.0,
        ):
            score = grade_trajectory(
                str(tmp_path / "result.json"),
                str(tmp_path / "expected.json"),
            )

        # 15 turns <= 15 max => efficiency = 1.0, all perfect => 1.0
        assert score == pytest.approx(1.0)

    def test_llm_judge_called_with_correct_rubrics(self, tmp_path):
        """Verify llm_judge is called twice with expected rubrics."""
        result = {
            "turns": _make_turns(["Hi"]),
            "final_state": _PERFECT_STATE,
        }
        expected = {
            "expected_final_state": _PERFECT_STATE,
            "max_turns": 10,
        }
        _write_json(tmp_path / "result.json", result)
        _write_json(tmp_path / "expected.json", expected)

        mock_judge = AsyncMock(return_value=0.9)
        with patch("grading.trajectory_grader.llm_judge", mock_judge):
            grade_trajectory(
                str(tmp_path / "result.json"),
                str(tmp_path / "expected.json"),
            )

        assert mock_judge.call_count == 2
        tone_call = mock_judge.call_args_list[0]
        assert "professional" in str(tone_call).lower()
        edge_call = mock_judge.call_args_list[1]
        assert "unexpected" in str(edge_call).lower()

    def test_perfect_deterministic_score(self, tmp_path):
        """All perfect, LLM=0.0 -> deterministic portion = 0.6."""
        result = {
            "turns": _make_turns(["Hello", "Got it", "Done"]),
            "final_state": {
                "user_name": "Alice",
                "email": "[email protected]",
                "plan": "premium",
                "account_created": True,
            },
        }
        expected = {
            "expected_final_state": {
                "user_name": "Alice",
                "email": "[email protected]",
                "plan": "premium",
                "account_created": True,
            },
            "max_turns": 10,
        }
        _write_json(tmp_path / "result.json", result)
        _write_json(tmp_path / "expected.json", expected)

        with patch(
            "grading.trajectory_grader.llm_judge",
            new_callable=AsyncMock,
            return_value=0.0,
        ):
            score = grade_trajectory(
                str(tmp_path / "result.json"),
                str(tmp_path / "expected.json"),
            )

        # det = 0.6 * (0.5*1.0 + 0.3*1.0 + 0.2*1.0) = 0.6
        # qual = 0.4 * 0.0 = 0.0
        assert score == pytest.approx(0.6)
