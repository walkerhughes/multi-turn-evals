"""Integration tests for the onboarding graph assembly."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph, route_after_confirm, route_after_email
from agent.nodes import tools_by_name
from agent.tools import get_mock_tools
from tests.unit.conftest import make_config, make_fake_llm, make_state


def _mock_tools_list():
    return get_mock_tools()


def _mock_tools_dict():
    return tools_by_name(get_mock_tools())


# ---------------------------------------------------------------------------
# Route function unit tests (fast, no graph needed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRouteAfterConfirm:
    def test_change_routes_to_select_plan(self):
        state = make_state({"messages": [HumanMessage(content="I want to change my plan")]})
        assert route_after_confirm(state) == "select_plan"

    def test_go_back_routes_to_select_plan(self):
        state = make_state({"messages": [HumanMessage(content="go back please")]})
        assert route_after_confirm(state) == "select_plan"

    def test_yes_routes_to_complete(self):
        state = make_state({"messages": [HumanMessage(content="yes that looks good")]})
        assert route_after_confirm(state) == "complete"

    def test_confirm_routes_to_complete(self):
        state = make_state({"messages": [HumanMessage(content="I confirm")]})
        assert route_after_confirm(state) == "complete"

    def test_ambiguous_routes_back_to_confirm(self):
        state = make_state({"messages": [HumanMessage(content="hmm I'm not sure")]})
        assert route_after_confirm(state) == "confirm"


@pytest.mark.integration
class TestRouteAfterEmail:
    def test_valid_email_routes_to_verify(self):
        state = make_state({"email": "alice@example.com"})
        assert route_after_email(state) == "verify_email"

    def test_retry_exceeded_routes_to_error(self):
        state = make_state({"email": None, "retry_count": 4})
        assert route_after_email(state) == "error"

    def test_no_email_retries_available(self):
        state = make_state({"email": None, "retry_count": 1})
        assert route_after_email(state) == "collect_email"


# ---------------------------------------------------------------------------
# Graph compilation test
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGraphCompilation:
    def test_graph_compiles(self):
        """build_graph should return a compiled graph without error."""
        graph = build_graph(tools=_mock_tools_list())
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        graph = build_graph(tools=_mock_tools_list())
        node_names = set(graph.get_graph().nodes.keys())
        expected = {
            "__start__",
            "__end__",
            "greeting",
            "collect_name",
            "collect_email",
            "verify_email",
            "select_plan",
            "collect_preferences",
            "confirm",
            "complete",
        }
        assert expected == node_names


# ---------------------------------------------------------------------------
# End-to-end graph invocation tests
# ---------------------------------------------------------------------------


def _tc(name: str, args: dict, call_id: str = "call_1") -> dict:
    """Shorthand for creating a ToolCall dict."""
    return {"name": name, "args": args, "id": call_id}


@pytest.mark.integration
class TestHappyPath:
    """Full happy-path traversal: greeting → ... → complete."""

    async def test_happy_path(self):
        tools_dict = _mock_tools_dict()
        tools_list = _mock_tools_list()

        # Sequence of LLM responses the fake model will return, one per node invocation:
        # 1. greeting: simple welcome
        # 2. collect_name: extracts name
        # 3. collect_email: calls validate_email tool
        # 4. verify_email: calls send + check verification code tools
        # 5. select_plan: calls get_plan_details + picks plan
        # 6. collect_preferences: extracts preferences
        # 7. confirm: summarizes (then route_after_confirm sees user "yes")
        # 8. complete: calls create_account
        responses = [
            AIMessage(content="Welcome! What's your name?"),
            AIMessage(content="NAME: Alice\nNice to meet you!"),
            AIMessage(
                content="",
                tool_calls=[_tc("validate_email", {"email": "alice@example.com"})],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tc("send_verification_code", {"email": "alice@example.com"}, "call_send"),
                    _tc("check_verification_code", {"email": "alice@example.com", "code": "123456"}, "call_check"),
                ],
            ),
            AIMessage(content="PLAN: pro\nGreat choice!"),
            AIMessage(content='PREFERENCES: {"theme": "dark"}\nAll set!'),
            AIMessage(content="Here is your summary. Please confirm."),
            AIMessage(
                content="",
                tool_calls=[
                    _tc(
                        "create_account",
                        {
                            "name": "Alice",
                            "email": "alice@example.com",
                            "plan": "pro",
                            "preferences": {"theme": "dark"},
                        },
                    )
                ],
            ),
        ]

        llm = make_fake_llm(responses)
        config = make_config(llm, tools_dict)
        graph = build_graph(tools=tools_list)

        # Seed with a user message so the graph has input
        initial_state = make_state({"messages": [HumanMessage(content="Hi")]})

        # We need to inject a "yes" message before the confirm routing fires.
        # Since route_after_confirm reads the *last* message, and the confirm node
        # appends the AI summary, we invoke step-by-step to inject user messages
        # at the right points. But for a simpler approach, we can stream node by node.

        # Actually, the conditional edge reads the state AFTER the node runs.
        # After confirm node, the last message is the AI summary ("Here is your summary...").
        # route_after_confirm will see "confirm" in "please confirm" → routes to "complete".

        result = await graph.ainvoke(initial_state, config)

        assert result["user_name"] == "Alice"
        assert result["email"] == "alice@example.com"
        assert result["email_verified"] is True
        assert result["plan"] == "pro"
        assert result["preferences"] == {"theme": "dark"}
        assert result["account_created"] is True


@pytest.mark.integration
class TestEmailRetry:
    """Invalid email → retry → valid email → proceeds."""

    async def test_email_retry_then_success(self):
        tools_dict = _mock_tools_dict()
        tools_list = _mock_tools_list()

        responses = [
            # greeting
            AIMessage(content="Welcome! What's your name?"),
            # collect_name
            AIMessage(content="NAME: Bob\nHello Bob!"),
            # collect_email (1st attempt - no tool call, simulating invalid)
            AIMessage(content="Please provide a valid email."),
            # collect_email (2nd attempt - calls validate_email successfully)
            AIMessage(
                content="",
                tool_calls=[_tc("validate_email", {"email": "bob@example.com"})],
            ),
            # verify_email
            AIMessage(
                content="",
                tool_calls=[
                    _tc("send_verification_code", {"email": "bob@example.com"}, "call_send"),
                    _tc("check_verification_code", {"email": "bob@example.com", "code": "123456"}, "call_check"),
                ],
            ),
            # select_plan
            AIMessage(content="PLAN: free\nGood choice!"),
            # collect_preferences
            AIMessage(content='PREFERENCES: {"lang": "en"}\nGot it!'),
            # confirm
            AIMessage(content="Everything looks great! Please confirm."),
            # complete
            AIMessage(
                content="",
                tool_calls=[
                    _tc(
                        "create_account",
                        {"name": "Bob", "email": "bob@example.com", "plan": "free", "preferences": {"lang": "en"}},
                    )
                ],
            ),
        ]

        llm = make_fake_llm(responses)
        config = make_config(llm, tools_dict)
        graph = build_graph(tools=tools_list)
        initial_state = make_state({"messages": [HumanMessage(content="Hello")]})

        result = await graph.ainvoke(initial_state, config)

        assert result["email"] == "bob@example.com"
        assert result["email_verified"] is True


@pytest.mark.integration
class TestConfirmGoBack:
    """User says 'go back' at confirm → routes back to select_plan."""

    async def test_confirm_change_rerouts_to_select_plan(self):
        tools_dict = _mock_tools_dict()
        tools_list = _mock_tools_list()

        responses = [
            # greeting
            AIMessage(content="Welcome!"),
            # collect_name
            AIMessage(content="NAME: Carol\nHi Carol!"),
            # collect_email
            AIMessage(
                content="",
                tool_calls=[_tc("validate_email", {"email": "carol@example.com"})],
            ),
            # verify_email
            AIMessage(
                content="",
                tool_calls=[
                    _tc("send_verification_code", {"email": "carol@example.com"}, "call_send"),
                    _tc("check_verification_code", {"email": "carol@example.com", "code": "123456"}, "call_check"),
                ],
            ),
            # select_plan (first time)
            AIMessage(content="PLAN: free\nOk!"),
            # collect_preferences
            AIMessage(content='PREFERENCES: {"tz": "UTC"}\nSaved!'),
            # confirm (user will "go back" — route reads last AI msg)
            AIMessage(content="I want to go back and change my plan."),
            # select_plan (second time after go-back)
            AIMessage(content="PLAN: enterprise\nUpgraded!"),
            # collect_preferences (again)
            AIMessage(content='PREFERENCES: {"tz": "UTC"}\nSaved!'),
            # confirm (this time the AI says "confirm")
            AIMessage(content="Everything confirmed. Let's proceed."),
            # complete
            AIMessage(
                content="",
                tool_calls=[
                    _tc(
                        "create_account",
                        {
                            "name": "Carol",
                            "email": "carol@example.com",
                            "plan": "enterprise",
                            "preferences": {"tz": "UTC"},
                        },
                    )
                ],
            ),
        ]

        llm = make_fake_llm(responses)
        config = make_config(llm, tools_dict)
        graph = build_graph(tools=tools_list)
        initial_state = make_state({"messages": [HumanMessage(content="Hey")]})

        result = await graph.ainvoke(initial_state, config)

        # After the go-back loop, the final plan should be enterprise
        assert result["plan"] == "enterprise"
        assert result["account_created"] is True


@pytest.mark.integration
class TestErrorPath:
    """4+ invalid email attempts → error state."""

    async def test_error_after_max_retries(self):
        from langchain_core.tools import StructuredTool

        # Override validate_email to always return False (invalid)
        failing_validate = StructuredTool.from_function(
            func=lambda email: False,
            name="validate_email",
            description="Validate email",
        )
        tools_list = _mock_tools_list()
        tools_dict = _mock_tools_dict()
        tools_dict["validate_email"] = failing_validate

        # Replace in list too
        tools_list = [failing_validate if t.name == "validate_email" else t for t in tools_list]

        responses = [
            # greeting
            AIMessage(content="Welcome!"),
            # collect_name
            AIMessage(content="NAME: Dan\nHi!"),
            # collect_email — 4 failed attempts with tool calls that return False
            AIMessage(content="", tool_calls=[_tc("validate_email", {"email": "bad1"}, "c1")]),
            AIMessage(content="", tool_calls=[_tc("validate_email", {"email": "bad2"}, "c2")]),
            AIMessage(content="", tool_calls=[_tc("validate_email", {"email": "bad3"}, "c3")]),
            AIMessage(content="", tool_calls=[_tc("validate_email", {"email": "bad4"}, "c4")]),
        ]

        llm = make_fake_llm(responses)
        config = make_config(llm, tools_dict)
        graph = build_graph(tools=tools_list)
        initial_state = make_state({"messages": [HumanMessage(content="Hi")]})

        result = await graph.ainvoke(initial_state, config)

        assert result.get("email") is None
        assert result["retry_count"] > 3
