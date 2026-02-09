from typing import Literal, Optional

from langgraph.graph import MessagesState


class OnboardingState(MessagesState):
    """State schema for the customer onboarding agent."""

    # Accumulated user data
    user_name: Optional[str]
    email: Optional[str]
    email_verified: bool
    plan: Optional[Literal["free", "pro", "enterprise"]]
    preferences: dict
    account_created: bool

    # Flow control
    current_stage: Literal[
        "greeting",
        "collect_name",
        "collect_email",
        "verify_email",
        "select_plan",
        "collect_preferences",
        "confirm",
        "complete",
        "error",
    ]
    retry_count: int
    error_message: Optional[str]
