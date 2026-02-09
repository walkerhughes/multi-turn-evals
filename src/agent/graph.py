"""LangGraph graph assembly — wires nodes, edges, and conditional routing."""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent import nodes
from agent.state import OnboardingState
from agent.tools import get_real_tools


def route_after_confirm(state: OnboardingState) -> str:
    """Route after the confirm node based on user response."""
    last_msg = state["messages"][-1].content.lower()
    if "change" in last_msg or "go back" in last_msg:
        return "select_plan"
    if "yes" in last_msg or "confirm" in last_msg:
        return "complete"
    return "confirm"


def route_after_email(state: OnboardingState) -> str:
    """Route after collect_email based on validation result."""
    if state.get("email"):
        return "verify_email"
    if state.get("retry_count", 0) > 3:
        return "error"
    return "collect_email"


def build_graph(tools: list | None = None) -> CompiledStateGraph:
    """Assemble and compile the onboarding StateGraph.

    Args:
        tools: Optional list of tools. If None, uses get_real_tools().

    Returns:
        A compiled LangGraph state graph.
    """
    if tools is None:
        tools = get_real_tools()

    graph = StateGraph(OnboardingState)

    # -- Add nodes --
    graph.add_node("greeting", nodes.greeting)
    graph.add_node("collect_name", nodes.collect_name)
    graph.add_node("collect_email", nodes.collect_email)
    graph.add_node("verify_email", nodes.verify_email)
    graph.add_node("select_plan", nodes.select_plan)
    graph.add_node("collect_preferences", nodes.collect_preferences)
    graph.add_node("confirm", nodes.confirm)
    graph.add_node("complete", nodes.complete)

    # -- Linear edges --
    graph.add_edge("greeting", "collect_name")
    graph.add_edge("collect_name", "collect_email")
    graph.add_edge("verify_email", "select_plan")
    graph.add_edge("select_plan", "collect_preferences")
    graph.add_edge("collect_preferences", "confirm")

    # -- Conditional edges --
    graph.add_conditional_edges(
        "collect_email",
        route_after_email,
        {"verify_email": "verify_email", "error": END, "collect_email": "collect_email"},
    )
    graph.add_conditional_edges(
        "confirm",
        route_after_confirm,
        {"select_plan": "select_plan", "complete": "complete", "confirm": "confirm"},
    )

    # -- Entry and finish --
    graph.set_entry_point("greeting")
    graph.add_edge("complete", END)

    return graph.compile()
