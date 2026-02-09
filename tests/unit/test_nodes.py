"""Unit tests for onboarding agent node functions."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes import (
    _parse_name_from_response,
    _parse_plan_from_response,
    _parse_preferences_from_response,
    collect_email,
    collect_name,
    collect_preferences,
    complete,
    confirm,
    greeting,
    select_plan,
    verify_email,
)

from .conftest import make_config, make_fake_llm, make_state

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseName:
    def test_extracts_name(self):
        assert _parse_name_from_response("NAME: Alice\nNice to meet you!") == "Alice"

    def test_case_insensitive(self):
        assert _parse_name_from_response("name: Bob") == "Bob"

    def test_returns_none_when_no_prefix(self):
        assert _parse_name_from_response("Hello, what is your name?") is None

    def test_strips_whitespace(self):
        assert _parse_name_from_response("NAME:   Charlie  ") == "Charlie"

    def test_unicode_name(self):
        assert _parse_name_from_response("NAME: José García") == "José García"


class TestParsePlan:
    def test_extracts_free(self):
        assert _parse_plan_from_response("PLAN: free\nGreat choice!") == "free"

    def test_extracts_pro(self):
        assert _parse_plan_from_response("PLAN: Pro") == "pro"

    def test_extracts_enterprise(self):
        assert _parse_plan_from_response("PLAN: ENTERPRISE") == "enterprise"

    def test_returns_none_for_invalid_plan(self):
        assert _parse_plan_from_response("PLAN: ultimate") is None

    def test_returns_none_when_no_prefix(self):
        assert _parse_plan_from_response("Let me show you our plans") is None


class TestParsePreferences:
    def test_extracts_valid_json(self):
        result = _parse_preferences_from_response('PREFERENCES: {"theme": "dark", "lang": "en"}')
        assert result == {"theme": "dark", "lang": "en"}

    def test_returns_none_for_invalid_json(self):
        assert _parse_preferences_from_response("PREFERENCES: not json") is None

    def test_returns_none_when_no_prefix(self):
        assert _parse_preferences_from_response("What are your preferences?") is None

    def test_case_insensitive(self):
        result = _parse_preferences_from_response('preferences: {"notifications": true}')
        assert result == {"notifications": True}


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------


class TestGreeting:
    async def test_transitions_to_collect_name(self, mock_tools):
        llm = make_fake_llm([AIMessage(content="Welcome! What's your name?")])
        state = make_state()
        config = make_config(llm, mock_tools)

        result = await greeting(state, config)

        assert result["current_stage"] == "collect_name"

    async def test_returns_ai_message(self, mock_tools):
        llm = make_fake_llm([AIMessage(content="Hello there!")])
        state = make_state()
        config = make_config(llm, mock_tools)

        result = await greeting(state, config)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "Hello there!"


class TestCollectName:
    async def test_extracts_name_and_transitions(self, mock_tools):
        llm = make_fake_llm([AIMessage(content="NAME: Alice\nNice to meet you, Alice!")])
        state = make_state({"messages": [HumanMessage(content="My name is Alice")]})
        config = make_config(llm, mock_tools)

        result = await collect_name(state, config)

        assert result["user_name"] == "Alice"
        assert result["current_stage"] == "collect_email"

    async def test_stays_when_no_name(self, mock_tools):
        llm = make_fake_llm([AIMessage(content="I didn't catch your name. Could you tell me?")])
        state = make_state({"messages": [HumanMessage(content="hello")]})
        config = make_config(llm, mock_tools)

        result = await collect_name(state, config)

        assert "user_name" not in result
        assert "current_stage" not in result

    async def test_handles_unicode_name(self, mock_tools):
        llm = make_fake_llm([AIMessage(content="NAME: José\nWelcome José!")])
        state = make_state({"messages": [HumanMessage(content="I'm José")]})
        config = make_config(llm, mock_tools)

        result = await collect_name(state, config)

        assert result["user_name"] == "José"


class TestCollectEmail:
    async def test_valid_email_transitions(self, mock_tools):
        response = AIMessage(
            content="",
            tool_calls=[{"name": "validate_email", "args": {"email": "alice@example.com"}, "id": "tc1"}],
        )
        llm = make_fake_llm([response])
        state = make_state({"current_stage": "collect_email", "messages": [HumanMessage(content="alice@example.com")]})
        config = make_config(llm, mock_tools)

        result = await collect_email(state, config)

        assert result["email"] == "alice@example.com"
        assert result["current_stage"] == "verify_email"
        assert result["retry_count"] == 0

    async def test_invalid_email_increments_retry(self, mock_tools):
        response = AIMessage(
            content="",
            tool_calls=[{"name": "validate_email", "args": {"email": "not-an-email"}, "id": "tc1"}],
        )
        # Override validate_email mock to return False for invalid
        from langchain_core.tools import StructuredTool

        tools = dict(mock_tools)
        tools["validate_email"] = StructuredTool.from_function(
            func=lambda email: False,
            name="validate_email",
            description="Validate email",
        )
        llm = make_fake_llm([response])
        state = make_state({"current_stage": "collect_email", "retry_count": 0})
        config = make_config(llm, tools)

        result = await collect_email(state, config)

        assert result["retry_count"] == 1
        assert "email" not in result

    async def test_retry_exceeds_limit_transitions_to_error(self, mock_tools):
        response = AIMessage(
            content="",
            tool_calls=[{"name": "validate_email", "args": {"email": "bad"}, "id": "tc1"}],
        )
        from langchain_core.tools import StructuredTool

        tools = dict(mock_tools)
        tools["validate_email"] = StructuredTool.from_function(
            func=lambda email: False,
            name="validate_email",
            description="Validate email",
        )
        llm = make_fake_llm([response])
        state = make_state({"current_stage": "collect_email", "retry_count": 3})
        config = make_config(llm, tools)

        result = await collect_email(state, config)

        assert result["current_stage"] == "error"
        assert result["error_message"] == "Too many invalid email attempts."

    async def test_no_tool_calls_returns_message_only(self, mock_tools):
        llm = make_fake_llm([AIMessage(content="Please provide your email.")])
        state = make_state({"current_stage": "collect_email"})
        config = make_config(llm, mock_tools)

        result = await collect_email(state, config)

        assert len(result["messages"]) == 1
        assert "email" not in result


class TestVerifyEmail:
    async def test_send_code(self, mock_tools):
        response = AIMessage(
            content="",
            tool_calls=[{"name": "send_verification_code", "args": {"email": "alice@example.com"}, "id": "tc1"}],
        )
        llm = make_fake_llm([response])
        state = make_state({"current_stage": "verify_email", "email": "alice@example.com"})
        config = make_config(llm, mock_tools)

        result = await verify_email(state, config)

        # Should have AIMessage + ToolMessage
        assert len(result["messages"]) == 2
        assert result["messages"][1].content == "123456"

    async def test_verify_success(self, mock_tools):
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "check_verification_code",
                    "args": {"email": "alice@example.com", "code": "123456"},
                    "id": "tc2",
                },
            ],
        )
        llm = make_fake_llm([response])
        state = make_state({"current_stage": "verify_email", "email": "alice@example.com"})
        config = make_config(llm, mock_tools)

        result = await verify_email(state, config)

        assert result["email_verified"] is True
        assert result["current_stage"] == "select_plan"

    async def test_verify_failure(self, mock_tools):
        from langchain_core.tools import StructuredTool

        tools = dict(mock_tools)
        tools["check_verification_code"] = StructuredTool.from_function(
            func=lambda email, code: False,
            name="check_verification_code",
            description="Check code",
        )
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "check_verification_code",
                    "args": {"email": "alice@example.com", "code": "wrong"},
                    "id": "tc2",
                },
            ],
        )
        llm = make_fake_llm([response])
        state = make_state({"current_stage": "verify_email", "email": "alice@example.com"})
        config = make_config(llm, tools)

        result = await verify_email(state, config)

        assert "email_verified" not in result
        assert "current_stage" not in result


class TestSelectPlan:
    async def test_selects_plan_and_transitions(self, mock_tools):
        llm = make_fake_llm([AIMessage(content="PLAN: pro\nGreat choice!")])
        state = make_state({"current_stage": "select_plan"})
        config = make_config(llm, mock_tools)

        result = await select_plan(state, config)

        assert result["plan"] == "pro"
        assert result["current_stage"] == "collect_preferences"

    async def test_calls_get_plan_details(self, mock_tools):
        response = AIMessage(
            content="Here are the plan details:",
            tool_calls=[{"name": "get_plan_details", "args": {"plan": "pro"}, "id": "tc1"}],
        )
        llm = make_fake_llm([response])
        state = make_state({"current_stage": "select_plan"})
        config = make_config(llm, mock_tools)

        result = await select_plan(state, config)

        # Tool message should contain plan details
        assert len(result["messages"]) == 2
        plan_info = json.loads(result["messages"][1].content)
        assert plan_info["plan"] == "pro"

    async def test_stays_when_browsing(self, mock_tools):
        llm = make_fake_llm([AIMessage(content="We have three plans: free, pro, and enterprise.")])
        state = make_state({"current_stage": "select_plan"})
        config = make_config(llm, mock_tools)

        result = await select_plan(state, config)

        assert "plan" not in result
        assert "current_stage" not in result


class TestCollectPreferences:
    async def test_extracts_prefs_and_transitions(self, mock_tools):
        prefs = {"theme": "dark", "notifications": True, "language": "en"}
        llm = make_fake_llm([AIMessage(content=f"PREFERENCES: {json.dumps(prefs)}\nAll set!")])
        state = make_state({"current_stage": "collect_preferences"})
        config = make_config(llm, mock_tools)

        result = await collect_preferences(state, config)

        assert result["preferences"] == prefs
        assert result["current_stage"] == "confirm"

    async def test_stays_when_incomplete(self, mock_tools):
        llm = make_fake_llm([AIMessage(content="What theme would you prefer?")])
        state = make_state({"current_stage": "collect_preferences"})
        config = make_config(llm, mock_tools)

        result = await collect_preferences(state, config)

        assert "preferences" not in result
        assert "current_stage" not in result


class TestConfirm:
    async def test_summarizes_data(self, mock_tools):
        llm = make_fake_llm([AIMessage(content="Here's your summary. Does everything look correct?")])
        state = make_state({
            "current_stage": "confirm",
            "user_name": "Alice",
            "email": "alice@example.com",
            "plan": "pro",
            "preferences": {"theme": "dark"},
        })
        config = make_config(llm, mock_tools)

        result = await confirm(state, config)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    async def test_does_not_set_current_stage(self, mock_tools):
        llm = make_fake_llm([AIMessage(content="Confirm your details.")])
        state = make_state({"current_stage": "confirm"})
        config = make_config(llm, mock_tools)

        result = await confirm(state, config)

        assert "current_stage" not in result


class TestComplete:
    async def test_calls_create_account(self, mock_tools):
        response = AIMessage(
            content="",
            tool_calls=[{
                "name": "create_account",
                "args": {
                    "name": "Alice",
                    "email": "alice@example.com",
                    "plan": "pro",
                    "preferences": {"theme": "dark"},
                },
                "id": "tc1",
            }],
        )
        llm = make_fake_llm([response])
        state = make_state({
            "current_stage": "complete",
            "user_name": "Alice",
            "email": "alice@example.com",
            "plan": "pro",
            "preferences": {"theme": "dark"},
        })
        config = make_config(llm, mock_tools)

        result = await complete(state, config)

        assert result["account_created"] is True
        # Should have AIMessage + ToolMessage
        assert len(result["messages"]) == 2

    async def test_sets_account_created(self, mock_tools):
        response = AIMessage(
            content="",
            tool_calls=[{
                "name": "create_account",
                "args": {"name": "Bob", "email": "bob@test.com", "plan": "free", "preferences": {}},
                "id": "tc1",
            }],
        )
        llm = make_fake_llm([response])
        state = make_state({"current_stage": "complete"})
        config = make_config(llm, mock_tools)

        result = await complete(state, config)

        assert result["account_created"] is True

    async def test_no_tool_calls_does_not_set_created(self, mock_tools):
        llm = make_fake_llm([AIMessage(content="Let me create your account.")])
        state = make_state({"current_stage": "complete"})
        config = make_config(llm, mock_tools)

        result = await complete(state, config)

        assert "account_created" not in result
