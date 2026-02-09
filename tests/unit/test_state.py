import pytest

from agent.state import OnboardingState

EXPECTED_KEYS = {
    "messages",
    "user_name",
    "email",
    "email_verified",
    "plan",
    "preferences",
    "account_created",
    "current_stage",
    "retry_count",
    "error_message",
}

VALID_STAGES = {
    "greeting",
    "collect_name",
    "collect_email",
    "verify_email",
    "select_plan",
    "collect_preferences",
    "confirm",
    "complete",
    "error",
}

VALID_PLANS = {"free", "pro", "enterprise"}


@pytest.mark.unit
class TestOnboardingStateSchema:
    def test_has_all_expected_keys(self):
        annotations = OnboardingState.__annotations__
        assert EXPECTED_KEYS == set(annotations)

    def test_can_instantiate_with_all_fields(self):
        state: OnboardingState = {
            "messages": [],
            "user_name": "Alice",
            "email": "[email protected]",
            "email_verified": True,
            "plan": "pro",
            "preferences": {"timezone": "UTC"},
            "account_created": False,
            "current_stage": "greeting",
            "retry_count": 0,
            "error_message": None,
        }
        assert state["user_name"] == "Alice"
        assert state["current_stage"] == "greeting"

    def test_can_instantiate_with_optional_fields_as_none(self):
        state: OnboardingState = {
            "messages": [],
            "user_name": None,
            "email": None,
            "email_verified": False,
            "plan": None,
            "preferences": {},
            "account_created": False,
            "current_stage": "greeting",
            "retry_count": 0,
            "error_message": None,
        }
        assert state["user_name"] is None
        assert state["email"] is None
        assert state["plan"] is None
        assert state["error_message"] is None

    def test_valid_plan_literals(self):
        for plan in VALID_PLANS:
            state: OnboardingState = {
                "messages": [],
                "user_name": None,
                "email": None,
                "email_verified": False,
                "plan": plan,
                "preferences": {},
                "account_created": False,
                "current_stage": "greeting",
                "retry_count": 0,
                "error_message": None,
            }
            assert state["plan"] == plan

    def test_valid_stage_literals(self):
        for stage in VALID_STAGES:
            state: OnboardingState = {
                "messages": [],
                "user_name": None,
                "email": None,
                "email_verified": False,
                "plan": None,
                "preferences": {},
                "account_created": False,
                "current_stage": stage,
                "retry_count": 0,
                "error_message": None,
            }
            assert state["current_stage"] == stage

    def test_extends_messages_state(self):
        from langgraph.graph import MessagesState

        assert MessagesState in OnboardingState.__orig_bases__

    def test_messages_key_inherited(self):
        assert "messages" in OnboardingState.__annotations__
