import json

import pytest

from grading.step_grader import grade_step


@pytest.mark.unit
class TestGradeStep:
    def test_perfect_score_all_checks_pass(self, tmp_path):
        result = {
            "new_stage": "collect_email",
            "state_updates": {"user_name": "Alice", "email": "[email protected]"},
            "tool_calls": [{"name": "send_email"}, {"name": "validate_input"}],
            "agent_response": "Hello Alice, welcome aboard!",
        }
        expected = {
            "expected_stage": "collect_email",
            "expected_state": {"user_name": "Alice", "email": "[email protected]"},
            "required_tools": ["send_email", "validate_input"],
            "disallowed_tools": ["delete_account"],
            "disallowed_patterns": [r"error", r"fail"],
        }
        result_path = tmp_path / "result.json"
        expected_path = tmp_path / "expected.json"
        result_path.write_text(json.dumps(result))
        expected_path.write_text(json.dumps(expected))

        score = grade_step(str(result_path), str(expected_path))
        assert score == 1.0

    def test_partial_score(self, tmp_path):
        result = {
            "new_stage": "collect_email",
            "state_updates": {"user_name": "Bob"},
            "tool_calls": [{"name": "send_email"}],
            "agent_response": "Hello!",
        }
        expected = {
            "expected_stage": "collect_email",
            "expected_state": {"user_name": "Bob", "email": "[email protected]"},
            "required_tools": ["send_email", "validate_input"],
        }
        result_path = tmp_path / "result.json"
        expected_path = tmp_path / "expected.json"
        result_path.write_text(json.dumps(result))
        expected_path.write_text(json.dumps(expected))

        score = grade_step(str(result_path), str(expected_path))
        # 5 checks: stage OK (1), user_name OK (1), email FAIL (0),
        # send_email OK (1), validate_input FAIL (0) = 3/5
        assert score == pytest.approx(3 / 5)

    def test_zero_score_all_checks_fail(self, tmp_path):
        result = {
            "new_stage": "error",
            "state_updates": {"user_name": "Wrong"},
            "tool_calls": [{"name": "delete_account"}],
            "agent_response": "Something went wrong with an error",
        }
        expected = {
            "expected_stage": "collect_email",
            "expected_state": {"user_name": "Alice"},
            "required_tools": ["send_email"],
            "disallowed_tools": ["delete_account"],
            "disallowed_patterns": [r"error"],
        }
        result_path = tmp_path / "result.json"
        expected_path = tmp_path / "expected.json"
        result_path.write_text(json.dumps(result))
        expected_path.write_text(json.dumps(expected))

        score = grade_step(str(result_path), str(expected_path))
        # stage FAIL, user_name FAIL, send_email FAIL,
        # delete_account FAIL, pattern FAIL = 0/5
        assert score == 0.0

    def test_missing_optional_fields_no_crash(self, tmp_path):
        result = {
            "new_stage": "greeting",
            "agent_response": "Hi there!",
        }
        expected = {
            "expected_stage": "greeting",
        }
        result_path = tmp_path / "result.json"
        expected_path = tmp_path / "expected.json"
        result_path.write_text(json.dumps(result))
        expected_path.write_text(json.dumps(expected))

        score = grade_step(str(result_path), str(expected_path))
        assert score == 1.0

    def test_disallowed_patterns_detected(self, tmp_path):
        result = {
            "new_stage": "collect_name",
            "state_updates": {},
            "tool_calls": [],
            "agent_response": "Sorry, there was a FAILURE processing your request.",
        }
        expected = {
            "expected_stage": "collect_name",
            "disallowed_patterns": [r"failure", r"credit\s*card"],
        }
        result_path = tmp_path / "result.json"
        expected_path = tmp_path / "expected.json"
        result_path.write_text(json.dumps(result))
        expected_path.write_text(json.dumps(expected))

        score = grade_step(str(result_path), str(expected_path))
        # stage OK (1), "failure" matches (0), "credit card" no match (1) = 2/3
        assert score == pytest.approx(2 / 3)

    def test_empty_checks_returns_zero(self, tmp_path):
        result = {
            "new_stage": "greeting",
            "state_updates": {},
            "tool_calls": [],
            "agent_response": "Hello!",
        }
        expected = {}
        result_path = tmp_path / "result.json"
        expected_path = tmp_path / "expected.json"
        result_path.write_text(json.dumps(result))
        expected_path.write_text(json.dumps(expected))

        score = grade_step(str(result_path), str(expected_path))
        assert score == 0.0

    def test_state_field_missing_from_result(self, tmp_path):
        result = {
            "new_stage": "collect_email",
            "state_updates": {},
            "tool_calls": [],
            "agent_response": "",
        }
        expected = {
            "expected_state": {"email": "[email protected]"},
        }
        result_path = tmp_path / "result.json"
        expected_path = tmp_path / "expected.json"
        result_path.write_text(json.dumps(result))
        expected_path.write_text(json.dumps(expected))

        score = grade_step(str(result_path), str(expected_path))
        # state_updates.get("email") is None != "[email protected]" -> 0/1
        assert score == 0.0
