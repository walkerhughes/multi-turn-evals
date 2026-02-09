"""Agent tools — real and mock implementations for the onboarding flow."""

import re
import uuid
from typing import Any

from langchain_core.tools import StructuredTool

# ---------------------------------------------------------------------------
# Internal state for real verification codes
# ---------------------------------------------------------------------------
_verification_codes: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Plan catalog (shared by real and mock)
# ---------------------------------------------------------------------------
PLAN_CATALOG: dict[str, dict[str, Any]] = {
    "free": {
        "plan": "free",
        "price": "$0/month",
        "features": ["Basic support", "5 projects", "1 GB storage"],
        "limits": {"projects": 5, "storage_gb": 1, "team_members": 1},
    },
    "pro": {
        "plan": "pro",
        "price": "$29/month",
        "features": ["Priority support", "50 projects", "100 GB storage", "Team collaboration"],
        "limits": {"projects": 50, "storage_gb": 100, "team_members": 10},
    },
    "enterprise": {
        "plan": "enterprise",
        "price": "$99/month",
        "features": ["Dedicated support", "Unlimited projects", "1 TB storage", "SSO", "Audit logs"],
        "limits": {"projects": -1, "storage_gb": 1000, "team_members": -1},
    },
}

EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


# ---------------------------------------------------------------------------
# Tool implementation functions
# ---------------------------------------------------------------------------
def _validate_email_real(email: str) -> bool:
    return bool(EMAIL_RE.match(email))


def _validate_email_mock(email: str) -> bool:
    return True


def _send_verification_code_real(email: str) -> str:
    import random

    code = f"{random.randint(0, 999999):06d}"
    _verification_codes[email] = code
    return code


def _send_verification_code_mock(email: str) -> str:
    return "123456"


def _check_verification_code_real(email: str, code: str) -> bool:
    return _verification_codes.get(email) == code


def _check_verification_code_mock(email: str, code: str) -> bool:
    return code == "123456"


def _create_account_real(name: str, email: str, plan: str, preferences: dict) -> dict:
    return {
        "account_id": str(uuid.uuid4()),
        "status": "created",
        "name": name,
        "email": email,
        "plan": plan,
        "preferences": preferences,
    }


def _create_account_mock(name: str, email: str, plan: str, preferences: dict) -> dict:
    return {"account_id": "mock-uuid", "status": "created"}


def _get_plan_details_real(plan: str) -> dict:
    if plan not in PLAN_CATALOG:
        return {"error": f"Unknown plan: {plan}. Valid plans: free, pro, enterprise"}
    return PLAN_CATALOG[plan]


def _get_plan_details_mock(plan: str) -> dict:
    if plan not in PLAN_CATALOG:
        return {"error": f"Unknown plan: {plan}. Valid plans: free, pro, enterprise"}
    return PLAN_CATALOG[plan]


# ---------------------------------------------------------------------------
# Factory helpers — return LangChain StructuredTool instances
# ---------------------------------------------------------------------------
def make_validate_email(*, mock: bool = False) -> StructuredTool:
    """Create a validate_email tool (real or mock)."""
    fn = _validate_email_mock if mock else _validate_email_real
    return StructuredTool.from_function(
        func=fn,
        name="validate_email",
        description="Validate whether an email address has correct format.",
    )


def make_send_verification_code(*, mock: bool = False) -> StructuredTool:
    """Create a send_verification_code tool (real or mock)."""
    fn = _send_verification_code_mock if mock else _send_verification_code_real
    return StructuredTool.from_function(
        func=fn,
        name="send_verification_code",
        description="Send a 6-digit verification code to the given email address. Returns the code string.",
    )


def make_check_verification_code(*, mock: bool = False) -> StructuredTool:
    """Create a check_verification_code tool (real or mock)."""
    fn = _check_verification_code_mock if mock else _check_verification_code_real
    return StructuredTool.from_function(
        func=fn,
        name="check_verification_code",
        description="Check whether the verification code matches the one sent to the email.",
    )


def make_create_account(*, mock: bool = False) -> StructuredTool:
    """Create a create_account tool (real or mock)."""
    fn = _create_account_mock if mock else _create_account_real
    return StructuredTool.from_function(
        func=fn,
        name="create_account",
        description="Create a new user account with the given name, email, plan, and preferences.",
    )


def make_get_plan_details(*, mock: bool = False) -> StructuredTool:
    """Create a get_plan_details tool (real or mock)."""
    fn = _get_plan_details_mock if mock else _get_plan_details_real
    return StructuredTool.from_function(
        func=fn,
        name="get_plan_details",
        description="Get pricing, features, and limits for a plan (free, pro, or enterprise).",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_real_tools() -> list[StructuredTool]:
    """Return all 5 tools with real implementations."""
    return [
        make_validate_email(mock=False),
        make_send_verification_code(mock=False),
        make_check_verification_code(mock=False),
        make_create_account(mock=False),
        make_get_plan_details(mock=False),
    ]


def get_mock_tools() -> list[StructuredTool]:
    """Return all 5 tools with mock implementations (for evals)."""
    return [
        make_validate_email(mock=True),
        make_send_verification_code(mock=True),
        make_check_verification_code(mock=True),
        make_create_account(mock=True),
        make_get_plan_details(mock=True),
    ]
