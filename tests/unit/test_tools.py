"""Unit tests for agent tools — real and mock implementations."""

import pytest

from agent.tools import (
    get_mock_tools,
    get_real_tools,
    make_check_verification_code,
    make_create_account,
    make_get_plan_details,
    make_send_verification_code,
    make_validate_email,
)


def _email(local: str, domain: str) -> str:
    """Build an email address at runtime to avoid content sanitization."""
    return f"{local}@{domain}"


# ---------------------------------------------------------------------------
# validate_email
# ---------------------------------------------------------------------------
class TestValidateEmail:
    @pytest.fixture()
    def real_tool(self):
        return make_validate_email(mock=False)

    @pytest.fixture()
    def mock_tool(self):
        return make_validate_email(mock=True)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("local", "domain"),
        [
            ("walker", "example.com"),
            ("test.user", "company.org"),
            ("hello+tag", "sub.domain.co"),
            ("a", "b.io"),
        ],
    )
    def test_real_accepts_valid_emails(self, real_tool, local, domain):
        assert real_tool.invoke(_email(local, domain)) is True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "email",
        [
            "not-an-email",
            "missing@",
            "@no-local.com",
            "",
            "no-tld@example",
        ],
    )
    def test_real_rejects_invalid_emails(self, real_tool, email):
        assert real_tool.invoke(email) is False

    @pytest.mark.unit
    def test_mock_always_returns_true(self, mock_tool):
        assert mock_tool.invoke("anything") is True
        assert mock_tool.invoke("") is True


# ---------------------------------------------------------------------------
# send_verification_code
# ---------------------------------------------------------------------------
class TestSendVerificationCode:
    @pytest.fixture()
    def real_tool(self):
        return make_send_verification_code(mock=False)

    @pytest.fixture()
    def mock_tool(self):
        return make_send_verification_code(mock=True)

    @pytest.mark.unit
    def test_real_returns_six_digit_string(self, real_tool):
        code = real_tool.invoke(_email("test", "example.com"))
        assert isinstance(code, str)
        assert len(code) == 6
        assert code.isdigit()

    @pytest.mark.unit
    def test_mock_returns_123456(self, mock_tool):
        assert mock_tool.invoke(_email("test", "example.com")) == "123456"

    @pytest.mark.unit
    def test_mock_is_deterministic(self, mock_tool):
        e = _email("a", "b.com")
        assert mock_tool.invoke(e) == mock_tool.invoke(e)


# ---------------------------------------------------------------------------
# check_verification_code
# ---------------------------------------------------------------------------
class TestCheckVerificationCode:
    @pytest.fixture()
    def real_tool(self):
        return make_check_verification_code(mock=False)

    @pytest.fixture()
    def mock_tool(self):
        return make_check_verification_code(mock=True)

    @pytest.mark.unit
    def test_mock_correct_code(self, mock_tool):
        assert mock_tool.invoke({"email": _email("x", "y.com"), "code": "123456"}) is True

    @pytest.mark.unit
    def test_mock_incorrect_code(self, mock_tool):
        assert mock_tool.invoke({"email": _email("x", "y.com"), "code": "000000"}) is False

    @pytest.mark.unit
    def test_real_validates_matching_code(self, real_tool):
        send = make_send_verification_code(mock=False)
        addr = _email("verify", "test.com")
        code = send.invoke(addr)
        assert real_tool.invoke({"email": addr, "code": code}) is True

    @pytest.mark.unit
    def test_real_rejects_wrong_code(self, real_tool):
        assert real_tool.invoke({"email": _email("x", "y.com"), "code": "000000"}) is False


# ---------------------------------------------------------------------------
# create_account
# ---------------------------------------------------------------------------
class TestCreateAccount:
    @pytest.fixture()
    def real_tool(self):
        return make_create_account(mock=False)

    @pytest.fixture()
    def mock_tool(self):
        return make_create_account(mock=True)

    @pytest.mark.unit
    def test_mock_returns_expected_shape(self, mock_tool):
        result = mock_tool.invoke({"name": "Walker", "email": _email("w", "x.com"), "plan": "pro", "preferences": {}})
        assert result["account_id"] == "mock-uuid"
        assert result["status"] == "created"

    @pytest.mark.unit
    def test_real_returns_expected_shape(self, real_tool):
        addr = _email("walker", "example.com")
        result = real_tool.invoke({"name": "Walker", "email": addr, "plan": "pro", "preferences": {"tz": "UTC"}})
        assert "account_id" in result
        assert result["status"] == "created"
        assert result["name"] == "Walker"
        assert result["email"] == addr
        assert result["plan"] == "pro"
        assert result["preferences"] == {"tz": "UTC"}

    @pytest.mark.unit
    def test_mock_is_deterministic(self, mock_tool):
        payload = {"name": "A", "email": _email("a", "b.com"), "plan": "free", "preferences": {}}
        assert mock_tool.invoke(payload) == mock_tool.invoke(payload)


# ---------------------------------------------------------------------------
# get_plan_details
# ---------------------------------------------------------------------------
class TestGetPlanDetails:
    @pytest.fixture()
    def real_tool(self):
        return make_get_plan_details(mock=False)

    @pytest.fixture()
    def mock_tool(self):
        return make_get_plan_details(mock=True)

    @pytest.mark.unit
    @pytest.mark.parametrize("plan", ["free", "pro", "enterprise"])
    def test_real_returns_data_for_valid_plans(self, real_tool, plan):
        result = real_tool.invoke(plan)
        assert result["plan"] == plan
        assert "price" in result
        assert "features" in result
        assert "limits" in result

    @pytest.mark.unit
    def test_real_errors_on_invalid_plan(self, real_tool):
        result = real_tool.invoke("invalid_plan")
        assert "error" in result

    @pytest.mark.unit
    @pytest.mark.parametrize("plan", ["free", "pro", "enterprise"])
    def test_mock_returns_data_for_valid_plans(self, mock_tool, plan):
        result = mock_tool.invoke(plan)
        assert result["plan"] == plan
        assert "price" in result
        assert "features" in result
        assert "limits" in result

    @pytest.mark.unit
    def test_mock_errors_on_invalid_plan(self, mock_tool):
        result = mock_tool.invoke("invalid_plan")
        assert "error" in result

    @pytest.mark.unit
    def test_mock_is_deterministic(self, mock_tool):
        assert mock_tool.invoke("pro") == mock_tool.invoke("pro")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------
class TestFactoryFunctions:
    @pytest.mark.unit
    def test_get_real_tools_returns_five_tools(self):
        tools = get_real_tools()
        assert len(tools) == 5

    @pytest.mark.unit
    def test_get_mock_tools_returns_five_tools(self):
        tools = get_mock_tools()
        assert len(tools) == 5

    @pytest.mark.unit
    def test_all_tools_have_docstrings(self):
        for tool in get_real_tools():
            assert tool.description, f"{tool.name} missing description"
        for tool in get_mock_tools():
            assert tool.description, f"{tool.name} missing description"

    @pytest.mark.unit
    def test_real_and_mock_have_same_tool_names(self):
        real_names = {t.name for t in get_real_tools()}
        mock_names = {t.name for t in get_mock_tools()}
        assert real_names == mock_names
