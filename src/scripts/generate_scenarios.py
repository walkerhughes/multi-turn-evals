"""Parametric scenario generator for step-level eval tasks."""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

NAMES = [
    "Walker Thompson",
    "María García-López",
    "张伟",
    "Dr. Sarah Patel",
    "O'Brien",
    "Jean-Pierre Dubois",
    "",
    "A",
]

EMAILS = [
    "[email protected]",
    "bad-email",
    "[email protected]",
    "[email protected]",
    "",
    "multiple@one .com [email protected]",
]

PLANS = ["free", "pro", "enterprise", "invalid_plan", None]

EDGE_CASES = [
    {"type": "offtopic", "message": "What's the weather?"},
    {"type": "hostile", "message": "This sucks, just make my account"},
    {"type": "change_mind", "message": "Actually go back to the plan step"},
    {"type": "overshare", "message": "I'm Walker, [email protected], pro plan please"},
]


def build_history_for_stage(
    stage: str, name: str, email: str, plan: str | None
) -> list[dict]:
    """Generate realistic conversation history up to the given stage.

    Returns a list of {"role": "...", "content": "..."} dicts that alternate
    between assistant and user messages appropriately for the given stage.
    """
    history = []

    # Always start with greeting
    history.append({
        "role": "assistant",
        "content": "Welcome! I'm here to help you set up your new account. To get started, could you please tell me your name?",
    })

    if stage == "collect_name":
        # User just arrived, history is just the greeting
        if name:
            history.append({"role": "user", "content": f"Hi! I'm {name}"})
        else:
            history.append({"role": "user", "content": "Hello!"})
        return history

    # Past collect_name — name was provided
    display_name = name if name else "User"
    history.append({
        "role": "user",
        "content": f"Hi there, my name is {name}" if name else "I'd rather not say",
    })
    history.append({
        "role": "assistant",
        "content": f"NAME: {display_name}\nGreat to meet you! Now I'll need your email address.",
    })

    if stage == "collect_email":
        if email:
            history.append({"role": "user", "content": f"Sure, it's {email}"})
        else:
            history.append({"role": "user", "content": "Here's my email"})
        return history

    # Past collect_email — email was provided
    history.append({
        "role": "user",
        "content": f"My email is {email}" if email else "I don't have one",
    })
    history.append({
        "role": "assistant",
        "content": f"Thanks! I've sent a verification code to {email}. Could you enter the code?",
    })

    if stage == "verify_email":
        history.append({"role": "user", "content": "The code is 123456"})
        return history

    # Past verify_email
    history.append({"role": "user", "content": "The code is 123456"})
    history.append({
        "role": "assistant",
        "content": "Email verified! Now let's choose a plan. We have free, pro, and enterprise options.",
    })

    if stage == "select_plan":
        if plan:
            history.append({"role": "user", "content": f"I'll go with the {plan} plan"})
        else:
            history.append({"role": "user", "content": "What plans do you have?"})
        return history

    # Past select_plan
    plan_str = plan if plan else "free"
    history.append({"role": "user", "content": f"I'd like the {plan_str} plan"})
    history.append({
        "role": "assistant",
        "content": f"PLAN: {plan_str}\nExcellent choice! Now let me ask about your preferences.",
    })

    if stage == "collect_preferences":
        history.append({
            "role": "user",
            "content": "I prefer dark mode and email notifications",
        })
        return history

    # Past collect_preferences
    history.append({
        "role": "user",
        "content": "Dark mode and weekly email digests please",
    })
    history.append({
        "role": "assistant",
        "content": 'PREFERENCES: {"theme": "dark", "notifications": "weekly"}\nLet me summarize everything for your confirmation.',
    })

    if stage == "confirm":
        history.append({"role": "user", "content": "Yes, that all looks correct!"})
        return history

    return history


def _determine_expected_stage(
    stage: str, name: str, email: str, plan: str | None
) -> str:
    """Determine the expected next stage based on inputs."""
    if stage == "collect_name":
        return "collect_email" if name else "collect_name"
    if stage == "collect_email":
        # valid email pattern check
        if email and "@" in email:
            parts = email.split("@")
            if len(parts) == 2 and "." in parts[1]:
                return "verify_email"
        return "collect_email"
    if stage == "verify_email":
        return "select_plan"
    if stage == "select_plan":
        if plan and plan in ("free", "pro", "enterprise"):
            return "collect_preferences"
        return "select_plan"
    if stage == "collect_preferences":
        return "confirm"
    if stage == "confirm":
        return "complete"
    return stage


def _determine_expected_state(
    stage: str, name: str, email: str, plan: str | None
) -> dict:
    """Determine expected state updates."""
    state = {}
    if stage == "collect_name" and name:
        state["user_name"] = name
    if stage == "collect_email" and email:
        if "@" in email:
            parts = email.split("@")
            if len(parts) == 2 and "." in parts[1]:
                state["email"] = email
    if stage == "select_plan" and plan in ("free", "pro", "enterprise"):
        state["plan"] = plan
    return state


def _determine_required_tools(stage: str, email: str) -> list[str]:
    """Determine which tools should be called at this stage."""
    if stage == "collect_email" and email:
        return ["validate_email"]
    if stage == "verify_email":
        return ["send_verification_code"]
    if stage == "confirm":
        return []
    if stage == "complete":
        return ["create_account"]
    return []


def generate_step_task(
    name: str, email: str, plan: str | None, stage: str, task_dir: Path
) -> None:
    """Generate a single step-level task directory."""
    task_dir.mkdir(parents=True, exist_ok=True)

    # instruction.md
    scenario = {
        "eval_mode": "step",
        "current_stage": stage,
        "history": build_history_for_stage(stage, name, email, plan),
        "accumulated_state": {
            "user_name": name if stage != "collect_name" else None,
            "email": email if stage not in ("collect_name", "collect_email") else None,
            "email_verified": stage
            not in ("collect_name", "collect_email", "verify_email"),
            "plan": plan
            if stage in ("collect_preferences", "confirm", "complete")
            else None,
            "preferences": {}
            if stage not in ("confirm", "complete")
            else {"theme": "dark", "notifications": "weekly"},
            "account_created": False,
        },
    }
    instruction_content = (
        "```json\n" + json.dumps(scenario, indent=2, ensure_ascii=False) + "\n```\n"
    )
    (task_dir / "instruction.md").write_text(instruction_content)

    # task.toml
    (task_dir / "task.toml").write_text(
        '[task]\ntimeout_sec = 60\n\n[task.metadata]\ncategory = "step-level"\nsubcategory = "generated"\n'
    )

    # expected.json
    expected = {
        "expected_stage": _determine_expected_stage(stage, name, email, plan),
        "expected_state": _determine_expected_state(stage, name, email, plan),
        "required_tools": _determine_required_tools(stage, email),
        "disallowed_tools": [],
        "disallowed_patterns": [],
    }
    (task_dir / "expected.json").write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate parametric step-level eval tasks"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tasks/step-level/generated",
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    output = Path(args.output)

    count = 0
    for name in NAMES[:4]:
        for email in EMAILS[:3]:
            task_dir = output / f"gen-{count:03d}"
            generate_step_task(name, email, "free", "collect_email", task_dir)
            count += 1

    logger.info("Generated %d tasks in %s", count, output)


if __name__ == "__main__":
    main()
