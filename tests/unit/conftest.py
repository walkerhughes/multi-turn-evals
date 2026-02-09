"""Unit test fixtures."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent.nodes import tools_by_name
from agent.tools import get_mock_tools


@pytest.fixture()
def mock_tools() -> dict:
    """Return mock tools as a dict keyed by name."""
    return tools_by_name(get_mock_tools())


def make_fake_llm(responses: list[AIMessage]) -> FakeMessagesListChatModel:
    """Create a FakeMessagesListChatModel from a list of AIMessage responses."""
    return FakeMessagesListChatModel(responses=responses)


def make_config(llm: FakeMessagesListChatModel, tools: dict) -> dict:
    """Build a RunnableConfig-compatible dict for node functions."""
    return {"configurable": {"llm": llm, "tools": tools}}


def make_state(overrides: dict | None = None) -> dict:
    """Build a minimal OnboardingState dict with sensible defaults."""
    base = {
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
    if overrides:
        base.update(overrides)
    return base
