"""Unit tests for Harbor BaseAgent wrapper."""

from unittest.mock import AsyncMock

import pytest

from src.harbor_agent.onboarding_agent import OnboardingEvalAgent


@pytest.mark.unit
def test_name():
    """Test that OnboardingEvalAgent.name() returns 'onboarding-agent'."""
    assert OnboardingEvalAgent.name() == "onboarding-agent"


@pytest.mark.unit
def test_version():
    """Test that OnboardingEvalAgent().version() returns '0.1.0'."""
    agent = OnboardingEvalAgent()
    assert agent.version() == "0.1.0"


@pytest.mark.unit
def test_supports_atif():
    """Test that SUPPORTS_ATIF is True."""
    assert OnboardingEvalAgent.SUPPORTS_ATIF is True


@pytest.mark.unit
def test_extract_tool_calls_with_tool_calls():
    """Test _extract_tool_calls with messages that have tool_calls."""

    class MockMessage:
        def __init__(self, has_tools=False):
            if has_tools:
                self.tool_calls = [{"name": "tool1"}, {"name": "tool2"}]
            else:
                self.tool_calls = []

    state = {
        "messages": [
            MockMessage(has_tools=True),
            MockMessage(has_tools=False),
            MockMessage(has_tools=True),
        ]
    }

    result = OnboardingEvalAgent._extract_tool_calls(state)
    assert len(result) == 4
    assert result[0] == {"name": "tool1"}
    assert result[1] == {"name": "tool2"}
    assert result[2] == {"name": "tool1"}
    assert result[3] == {"name": "tool2"}


@pytest.mark.unit
def test_extract_tool_calls_without_tool_calls():
    """Test _extract_tool_calls with messages that don't have tool_calls."""

    class MockMessage:
        pass

    state = {"messages": [MockMessage(), MockMessage()]}

    result = OnboardingEvalAgent._extract_tool_calls(state)
    assert result == []


@pytest.mark.unit
async def test_run_step_eval_builds_correct_initial_state():
    """Test that _run_step_eval builds correct initial_state structure."""

    class MockMessage:
        def __init__(self, content):
            self.content = content
            self.tool_calls = []

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "messages": [MockMessage("Hello!")],
        "current_stage": "collect_name",
        "user_name": "John",
        "email": "john@example.com",
        "email_verified": True,
        "plan": "pro",
    }

    scenario = {
        "history": [{"role": "user", "content": "Hi"}],
        "current_stage": "greeting",
        "accumulated_state": {
            "user_name": "Jane",
            "email": "jane@example.com",
            "email_verified": False,
            "plan": "basic",
            "preferences": {"theme": "dark"},
            "retry_count": 1,
        },
    }

    agent = OnboardingEvalAgent()
    result = await agent._run_step_eval(mock_graph, scenario)

    # Verify the initial_state passed to graph.ainvoke
    call_args = mock_graph.ainvoke.call_args[0][0]
    assert call_args["messages"] == [{"role": "user", "content": "Hi"}]
    assert call_args["current_stage"] == "greeting"
    assert call_args["user_name"] == "Jane"
    assert call_args["email"] == "jane@example.com"
    assert call_args["email_verified"] is False
    assert call_args["plan"] == "basic"
    assert call_args["preferences"] == {"theme": "dark"}
    assert call_args["account_created"] is False
    assert call_args["retry_count"] == 1
    assert call_args["error_message"] is None

    # Verify the output structure
    assert result["mode"] == "step"
    assert result["agent_response"] == "Hello!"
    assert result["new_stage"] == "collect_name"
    assert result["state_updates"]["user_name"] == "John"
    assert result["state_updates"]["email"] == "john@example.com"
    assert result["state_updates"]["plan"] == "pro"
    assert result["state_updates"]["email_verified"] is True
    assert result["tool_calls"] == []


@pytest.mark.unit
async def test_run_trajectory_eval_processes_all_turns():
    """Test that _run_trajectory_eval processes all turns."""

    class MockMessage:
        def __init__(self, content):
            self.content = content

    # Mock graph that updates state on each call
    mock_graph = AsyncMock()
    call_count = 0

    async def mock_ainvoke(state):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            state["messages"].append(MockMessage("What's your name?"))
            state["current_stage"] = "collect_name"
        elif call_count == 2:
            state["messages"].append(MockMessage("Great! What's your email?"))
            state["current_stage"] = "collect_email"
            state["user_name"] = "Alice"
        elif call_count == 3:
            state["messages"].append(MockMessage("Perfect! Account created."))
            state["current_stage"] = "complete"
            state["email"] = "alice@example.com"
            state["account_created"] = True
            state["plan"] = "pro"
        return state

    mock_graph.ainvoke = mock_ainvoke

    scenario = {
        "turns": [
            {"user_message": "Hello"},
            {"user_message": "My name is Alice"},
            {"user_message": "alice@example.com"},
        ]
    }

    agent = OnboardingEvalAgent()
    result = await agent._run_trajectory_eval(mock_graph, scenario)

    # Verify all turns were processed
    assert result["mode"] == "trajectory"
    assert len(result["turns"]) == 3

    assert result["turns"][0]["user"] == "Hello"
    assert result["turns"][0]["agent"] == "What's your name?"
    assert result["turns"][0]["stage"] == "collect_name"

    assert result["turns"][1]["user"] == "My name is Alice"
    assert result["turns"][1]["agent"] == "Great! What's your email?"
    assert result["turns"][1]["stage"] == "collect_email"

    assert result["turns"][2]["user"] == "alice@example.com"
    assert result["turns"][2]["agent"] == "Perfect! Account created."
    assert result["turns"][2]["stage"] == "complete"

    # Verify final state
    assert result["final_state"]["account_created"] is True
    assert result["final_state"]["user_name"] == "Alice"
    assert result["final_state"]["email"] == "alice@example.com"
    assert result["final_state"]["plan"] == "pro"
    assert result["final_state"]["preferences"] == {}
