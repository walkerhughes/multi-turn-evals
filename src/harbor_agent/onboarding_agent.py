"""Harbor BaseAgent wrapper — adapts the LangGraph agent for Harbor's eval framework."""

import json
from typing import Any


class BaseAgent:
    """Minimal stub for harbor.agents.base.BaseAgent (not installed)."""

    SUPPORTS_ATIF = False

    @staticmethod
    def name() -> str:
        raise NotImplementedError

    def version(self) -> str:
        raise NotImplementedError

    async def setup(self, environment: Any) -> None:
        pass

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        raise NotImplementedError


class OnboardingEvalAgent(BaseAgent):
    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "onboarding-agent"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: Any) -> None:
        pass

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        from agent.graph import build_graph
        from agent.tools import get_mock_tools

        scenario = json.loads(instruction)
        graph = build_graph(tools=get_mock_tools())

        if scenario["eval_mode"] == "step":
            output = await self._run_step_eval(graph, scenario)
        elif scenario["eval_mode"] == "trajectory":
            output = await self._run_trajectory_eval(graph, scenario)
        else:
            raise ValueError(f"Unknown eval_mode: {scenario['eval_mode']}")

        with open("/output/result.json", "w") as f:
            json.dump(output, f, indent=2)

    async def _run_step_eval(self, graph: Any, scenario: dict) -> dict:
        """Inject history, evaluate single next turn."""
        initial_state = {
            "messages": scenario["history"],
            "current_stage": scenario["current_stage"],
            "user_name": scenario.get("accumulated_state", {}).get("user_name"),
            "email": scenario.get("accumulated_state", {}).get("email"),
            "email_verified": scenario.get("accumulated_state", {}).get(
                "email_verified", False
            ),
            "plan": scenario.get("accumulated_state", {}).get("plan"),
            "preferences": scenario.get("accumulated_state", {}).get("preferences", {}),
            "account_created": False,
            "retry_count": scenario.get("accumulated_state", {}).get("retry_count", 0),
            "error_message": None,
        }

        result_state = await graph.ainvoke(initial_state)

        return {
            "mode": "step",
            "agent_response": result_state["messages"][-1].content,
            "new_stage": result_state["current_stage"],
            "state_updates": {
                "user_name": result_state.get("user_name"),
                "email": result_state.get("email"),
                "plan": result_state.get("plan"),
                "email_verified": result_state.get("email_verified"),
            },
            "tool_calls": self._extract_tool_calls(result_state),
        }

    async def _run_trajectory_eval(self, graph: Any, scenario: dict) -> dict:
        """Drive full conversation, collect all turns."""
        state = {
            "messages": [],
            "current_stage": "greeting",
            "user_name": None,
            "email": None,
            "email_verified": False,
            "plan": None,
            "preferences": {},
            "account_created": False,
            "retry_count": 0,
            "error_message": None,
        }

        all_turns = []
        for turn in scenario["turns"]:
            state["messages"].append({"role": "user", "content": turn["user_message"]})
            state = await graph.ainvoke(state)
            all_turns.append(
                {
                    "user": turn["user_message"],
                    "agent": state["messages"][-1].content,
                    "stage": state["current_stage"],
                }
            )

        return {
            "mode": "trajectory",
            "turns": all_turns,
            "final_state": {
                "account_created": state["account_created"],
                "user_name": state.get("user_name"),
                "email": state.get("email"),
                "plan": state.get("plan"),
                "preferences": state.get("preferences"),
            },
        }

    @staticmethod
    def _extract_tool_calls(state: dict) -> list:
        tool_calls = []
        for msg in state["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls.extend(msg.tool_calls)
        return tool_calls
