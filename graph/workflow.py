from langgraph.graph import StateGraph, START, END

from agents.counsellor import decide_action, generate_initial
from agents.feedback import apply_feedback
from graph.state import CounsellorState


def counsellor_node(state: CounsellorState):
    decision = decide_action(
        state["message"],
        state["profile"],
        state.get("history", []),
        state.get("existing_recommendations", []),
    )
    base = {
        "profile": decision.updated_profile,
        "trace": state.get("trace", []) + [f"Counsellor Agent selected action: {decision.action}"],
    }
    if decision.action == "handoff_to_feedback":
        return {**base, "route": "Feedback_Agent", "handoff_reason": decision.reasoning}
    if decision.action == "clarify":
        return {**base, "route": "Counsellor_Agent", "answer": decision.clarification_question, "recommendations": state.get("existing_recommendations", []), "confidence": "Low"}
    if decision.action == "answer_only":
        return {**base, "route": "Counsellor_Agent", "answer": decision.response, "recommendations": state.get("existing_recommendations", []), "confidence": "Medium"}

    rows, answer, confidence, evidence, trace = generate_initial(
        state["message"], decision.updated_profile, state.get("history", [])
    )
    return {
        **base,
        "route": "Counsellor_Agent",
        "answer": answer,
        "recommendations": rows,
        "confidence": confidence,
        "evidence": evidence,
        "trace": base["trace"] + trace,
    }


def route_after_counsellor(state: CounsellorState):
    return "feedback" if state.get("route") == "Feedback_Agent" else "end"


def feedback_node(state: CounsellorState):
    rows, answer, confidence, evidence, trace = apply_feedback(
        state["message"],
        state["profile"],
        state.get("history", []),
        state.get("existing_recommendations", []),
    )
    return {
        "route": "Feedback_Agent",
        "answer": answer,
        "recommendations": rows,
        "confidence": confidence,
        "evidence": evidence,
        "trace": state.get("trace", []) + trace,
    }


def build_workflow():
    graph = StateGraph(CounsellorState)
    graph.add_node("Counsellor_Agent", counsellor_node)
    graph.add_node("Feedback_Agent", feedback_node)
    graph.add_edge(START, "Counsellor_Agent")
    graph.add_conditional_edges(
        "Counsellor_Agent",
        route_after_counsellor,
        {"feedback": "Feedback_Agent", "end": END},
    )
    graph.add_edge("Feedback_Agent", END)
    return graph.compile()


workflow = build_workflow()
