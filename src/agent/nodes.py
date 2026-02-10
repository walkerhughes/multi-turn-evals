"""LangGraph node functions for the 8 onboarding stages."""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from agent.state import OnboardingState

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _get_llm(config: RunnableConfig) -> Any:
    """Extract the LLM from RunnableConfig."""
    return config["configurable"]["llm"]


def _get_tools(config: RunnableConfig) -> dict[str, StructuredTool]:
    """Extract tools dict from RunnableConfig."""
    return config["configurable"]["tools"]


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _build_messages(state: OnboardingState, system_prompt: str) -> list:
    """Prepend a system prompt to the conversation messages."""
    return [SystemMessage(content=system_prompt)] + list(state["messages"])


# ---------------------------------------------------------------------------
# Tool-call helpers
# ---------------------------------------------------------------------------


def _invoke_tool_calls(response: AIMessage, tools: dict[str, StructuredTool]) -> list[ToolMessage]:
    """Execute tool calls from an AIMessage and return ToolMessages."""
    tool_messages = []
    for tc in response.tool_calls:
        tool = tools[tc["name"]]
        result = tool.invoke(tc["args"])
        tool_messages.append(
            ToolMessage(content=json.dumps(result) if not isinstance(result, str) else result, tool_call_id=tc["id"])
        )
    return tool_messages


def tools_by_name(tools_list: list[StructuredTool]) -> dict[str, StructuredTool]:
    """Convert a list of tools to a dict keyed by tool name."""
    return {t.name: t for t in tools_list}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"^NAME:\s*(.+)", re.IGNORECASE)
_PLAN_RE = re.compile(r"^PLAN:\s*(.+)", re.IGNORECASE)
_PREFERENCES_RE = re.compile(r"^PREFERENCES:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_name_from_response(content: str) -> str | None:
    """Extract name from 'NAME: ...' prefix in first line."""
    first_line = content.strip().split("\n")[0]
    m = _NAME_RE.match(first_line)
    return m.group(1).strip() if m else None


def _parse_plan_from_response(content: str) -> str | None:
    """Extract plan from 'PLAN: ...' prefix in first line."""
    first_line = content.strip().split("\n")[0]
    m = _PLAN_RE.match(first_line)
    if not m:
        return None
    plan = m.group(1).strip().lower()
    if plan in ("free", "pro", "enterprise"):
        return plan
    return None


def _parse_preferences_from_response(content: str) -> dict | None:
    """Extract preferences dict from 'PREFERENCES: {...}' prefix."""
    first_line = content.strip().split("\n")[0]
    m = _PREFERENCES_RE.match(first_line)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def greeting(state: OnboardingState, config: RunnableConfig) -> dict:
    """Greet the user and transition to collect_name."""
    llm = _get_llm(config)
    messages = _build_messages(
        state,
        "You are a friendly onboarding assistant. Greet the user warmly and ask for their name.",
    )
    response = await llm.ainvoke(messages)
    return {
        "messages": [response],
        "current_stage": "collect_name",
    }


async def collect_name(state: OnboardingState, config: RunnableConfig) -> dict:
    """Collect the user's name from conversation."""
    llm = _get_llm(config)
    messages = _build_messages(
        state,
        "Extract the user's name from their message. "
        "If you can identify their name, respond with the first line as 'NAME: <their name>' "
        "followed by a friendly acknowledgment. "
        "If you cannot determine their name, ask them again politely.",
    )
    response = await llm.ainvoke(messages)
    name = _parse_name_from_response(response.content)
    result: dict[str, Any] = {"messages": [response]}
    if name:
        result["user_name"] = name
        result["current_stage"] = "collect_email"
    return result


async def collect_email(state: OnboardingState, config: RunnableConfig) -> dict:
    """Collect and validate the user's email address."""
    llm = _get_llm(config)
    tools = _get_tools(config)
    validate_email = tools["validate_email"]

    messages = _build_messages(
        state,
        "You are collecting the user's email address. "
        "Use the validate_email tool to check if the email is valid. "
        "If valid, confirm it. If invalid, ask them to try again.",
    )
    response = await llm.ainvoke(messages, tools=[validate_email])

    result: dict[str, Any] = {"messages": [response]}

    if response.tool_calls:
        tool_messages = _invoke_tool_calls(response, tools)
        result["messages"] = [response] + tool_messages

        tc = response.tool_calls[0]
        email = tc["args"].get("email", "")
        content = tool_messages[0].content
        is_valid = json.loads(str(content)) if content in ("true", "false") else False

        if is_valid:
            result["email"] = email
            result["current_stage"] = "verify_email"
            result["retry_count"] = 0
        else:
            retry = state.get("retry_count", 0) + 1
            result["retry_count"] = retry
            if retry > 3:
                result["current_stage"] = "error"
                result["error_message"] = "Too many invalid email attempts."
    return result


async def verify_email(state: OnboardingState, config: RunnableConfig) -> dict:
    """Send and verify an email verification code."""
    llm = _get_llm(config)
    tools = _get_tools(config)

    messages = _build_messages(
        state,
        "You are verifying the user's email address. "
        "Use send_verification_code to send a code, then use check_verification_code "
        "to verify the code the user provides.",
    )
    response = await llm.ainvoke(
        messages,
        tools=[tools["send_verification_code"], tools["check_verification_code"]],
    )

    result: dict[str, Any] = {"messages": [response]}

    if response.tool_calls:
        tool_messages = _invoke_tool_calls(response, tools)
        result["messages"] = [response] + tool_messages

        for tc, tm in zip(response.tool_calls, tool_messages):
            if tc["name"] == "check_verification_code":
                tm_content = tm.content
                is_verified = json.loads(str(tm_content)) if tm_content in ("true", "false") else False
                if is_verified:
                    result["email_verified"] = True
                    result["current_stage"] = "select_plan"
    return result


async def select_plan(state: OnboardingState, config: RunnableConfig) -> dict:
    """Help the user select a plan."""
    llm = _get_llm(config)
    tools = _get_tools(config)
    get_plan_details = tools["get_plan_details"]

    messages = _build_messages(
        state,
        "Help the user choose a plan (free, pro, or enterprise). "
        "Use the get_plan_details tool to show plan information. "
        "When the user has chosen a plan, respond with the first line as 'PLAN: <plan_name>'. "
        "If they're still browsing, continue helping without the PLAN prefix.",
    )
    response = await llm.ainvoke(messages, tools=[get_plan_details])

    result: dict[str, Any] = {"messages": [response]}

    if response.tool_calls:
        tool_messages = _invoke_tool_calls(response, tools)
        result["messages"] = [response] + tool_messages

    plan = _parse_plan_from_response(response.content)
    if plan:
        result["plan"] = plan
        result["current_stage"] = "collect_preferences"
    return result


async def collect_preferences(state: OnboardingState, config: RunnableConfig) -> dict:
    """Collect user preferences."""
    llm = _get_llm(config)
    messages = _build_messages(
        state,
        "Collect the user's preferences (e.g., notifications, theme, language). "
        "When you have gathered all preferences, respond with the first line as "
        '\'PREFERENCES: {"key": "value", ...}\' as valid JSON. '
        "If the user hasn't provided enough preferences yet, continue asking.",
    )
    response = await llm.ainvoke(messages)

    result: dict[str, Any] = {"messages": [response]}

    prefs = _parse_preferences_from_response(response.content)
    if prefs:
        result["preferences"] = prefs
        result["current_stage"] = "confirm"
    return result


async def confirm(state: OnboardingState, config: RunnableConfig) -> dict:
    """Summarize collected data and ask for confirmation. Does NOT set current_stage."""
    llm = _get_llm(config)
    summary = (
        f"Name: {state.get('user_name', 'N/A')}\n"
        f"Email: {state.get('email', 'N/A')}\n"
        f"Plan: {state.get('plan', 'N/A')}\n"
        f"Preferences: {json.dumps(state.get('preferences', {}))}"
    )
    messages = _build_messages(
        state,
        f"Summarize the user's information and ask them to confirm:\n\n{summary}\n\n"
        "Ask if everything looks correct. Do NOT set any stage transition.",
    )
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


async def complete(state: OnboardingState, config: RunnableConfig) -> dict:
    """Create the account using collected data."""
    llm = _get_llm(config)
    tools = _get_tools(config)
    create_account = tools["create_account"]

    messages = _build_messages(
        state,
        "The user has confirmed their details. Use create_account to finalize their account.",
    )
    response = await llm.ainvoke(messages, tools=[create_account])

    result: dict[str, Any] = {"messages": [response]}

    if response.tool_calls:
        tool_messages = _invoke_tool_calls(response, tools)
        result["messages"] = [response] + tool_messages
        result["account_created"] = True
    return result
