"""
LangGraph Orchestrator — the core agentic pipeline.

Wires all agents into an explicit state graph with conditional routing:
  IntakeAgent → EmergencyDetector → (emergency? → exit) → KnowledgeAgent
  → RetrievalAgent → RankingAgent → ExplanationAgent → format output
"""

from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.intake_agent import run_intake_agent
from app.agents.emergency_detector import run_emergency_detector
from app.agents.knowledge_agent import run_knowledge_agent
from app.agents.retrieval_agent import run_retrieval_agent
from app.agents.ranking_agent import run_ranking_agent
from app.agents.explanation_agent import run_explanation_agent


# ── Safety Disclaimer ──
DISCLAIMER = (
    "\n\n---\n⚕️ *RemedyRadar is not a medical diagnostic tool. "
    "The specialty recommendations are based on symptom patterns and should not "
    "replace professional medical advice. Always consult a healthcare professional "
    "for proper diagnosis and treatment.*"
)


def _format_emergency_response(state: AgentState) -> AgentState:
    """Format the emergency override response."""
    state["final_response"] = (
        f"🚨 **EMERGENCY — SEEK IMMEDIATE MEDICAL ATTENTION**\n\n"
        f"{state.get('emergency_message', 'Please call emergency services immediately.')}\n\n"
        f"📞 **Emergency Numbers:**\n"
        f"- USA: 911\n"
        f"- India: 112\n"
        f"- International: Your local emergency number\n\n"
        f"⏱️ Do not delay — every minute matters in an emergency."
    )
    return state


def _format_no_results(state: AgentState) -> AgentState:
    """Format response when no doctors are found."""
    specialty = state.get("recommended_specialty", "a specialist")
    radius = state.get("search_radius_used", 50)
    state["final_response"] = (
        f"📚 **Recommended Specialty: {specialty}**\n"
        f"{state.get('specialty_explanation', '')}\n\n"
        f"😕 Unfortunately, we couldn't find any {specialty}s within {radius}km of your location.\n\n"
        f"**Suggestions:**\n"
        f"- Try a broader search radius\n"
        f"- Consider telehealth options\n"
        f"- Search for a General Practitioner who can provide a referral"
        f"{DISCLAIMER}"
    )
    return state


def _format_results(state: AgentState) -> AgentState:
    """Format the final results with specialty, doctors, and explanations."""
    specialty = state.get("recommended_specialty", "Specialist")
    confidence = state.get("specialty_confidence", 0)
    explanation = state.get("specialty_explanation", "")
    ranked = state.get("ranked_doctors", [])
    explanations = {e["doctor_name"]: e["explanation"] for e in state.get("doctor_explanations", [])}
    sources = state.get("rag_sources_medical", [])
    radius = state.get("search_radius_used", 10)

    # Build response
    response = f"## 📚 Recommended Specialty: **{specialty}**\n"
    response += f"*Confidence: {confidence:.0%}*\n\n"
    response += f"{explanation}\n\n"

    # Source citations
    if sources:
        cited = set(s.get("source", "") for s in sources if s.get("source"))
        if cited:
            response += f"*Sources: {', '.join(cited)}*\n\n"

    response += "---\n\n"
    response += f"## 🏥 Top Doctors Near You\n"
    response += f"*Searched within {radius}km of your location*\n\n"

    for i, doc in enumerate(ranked, 1):
        score = doc.get("score", 0)
        name = doc.get("name", "Unknown")
        doc_explanation = explanations.get(name, "")

        response += f"### {i}. {name}\n"
        response += f"**{doc.get('specialty', '')}**"
        if doc.get("sub_specialization"):
            response += f" · {doc['sub_specialization']}"
        response += "\n\n"

        response += f"| Metric | Value |\n|--------|-------|\n"
        response += f"| ⭐ Rating | {doc.get('rating', 'N/A')}/5 ({doc.get('review_count', 0)} reviews) |\n"
        response += f"| 🏥 Hospital | {doc.get('hospital_name', 'N/A')} |\n"
        response += f"| 📍 Distance | {doc.get('distance_km', 'N/A')} km |\n"
        response += f"| 💰 Fee | {doc.get('consultation_fee', 'N/A')} |\n"
        response += f"| 🧑‍⚕️ Experience | {doc.get('years_experience', 'N/A')} years |\n"
        response += f"| 📞 Telehealth | {'✅ Yes' if doc.get('telehealth_available') else '❌ No'} |\n"
        response += f"| 📊 Match Score | **{score:.0%}** |\n\n"

        if doc_explanation:
            response += f"💡 *{doc_explanation}*\n\n"

        # Score breakdown
        breakdown = doc.get("score_breakdown", {})
        if breakdown:
            response += "<details><summary>📊 Score Breakdown</summary>\n\n"
            for factor, data in breakdown.items():
                bar_len = int(data["raw_score"] * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                response += f"- {factor}: {bar} {data['raw_score']:.0%} (weight: {data['weight']:.0%})\n"
            response += "\n</details>\n\n"

        response += "---\n\n"

    response += (
        "💬 **Refine your search** — You can say things like:\n"
        '- "Show only women doctors"\n'
        '- "I prefer someone who speaks Hindi"\n'
        '- "Prioritize experience over cost"\n'
        '- "Any of them offer teleconsultation?"\n'
    )
    response += DISCLAIMER

    state["final_response"] = response
    return state


# ── Conditional Edge Functions ──

def _route_after_emergency(state: AgentState) -> str:
    """Route after emergency detection: emergency → exit, else → knowledge."""
    if state.get("is_emergency"):
        return "format_emergency"
    return "knowledge_agent"


def _route_after_retrieval(state: AgentState) -> str:
    """Route after retrieval: no results → format_no_results, else → ranking."""
    if not state.get("candidate_doctors"):
        return "format_no_results"
    return "ranking_agent"


def build_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph for the RemedyRadar pipeline.

    Graph structure:
        intake → emergency_detector → [emergency? → exit | knowledge]
        knowledge → retrieval → [no results? → exit | ranking]
        ranking → explanation → format_results → END
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──
    graph.add_node("intake_agent", run_intake_agent)
    graph.add_node("emergency_detector", run_emergency_detector)
    graph.add_node("knowledge_agent", run_knowledge_agent)
    graph.add_node("retrieval_agent", run_retrieval_agent)
    graph.add_node("ranking_agent", run_ranking_agent)
    graph.add_node("explanation_agent", run_explanation_agent)
    graph.add_node("format_emergency", _format_emergency_response)
    graph.add_node("format_no_results", _format_no_results)
    graph.add_node("format_results", _format_results)

    # ── Set entry point ──
    graph.set_entry_point("intake_agent")

    # ── Add edges ──
    graph.add_edge("intake_agent", "emergency_detector")

    # Conditional: emergency → exit, else → knowledge
    graph.add_conditional_edges(
        "emergency_detector",
        _route_after_emergency,
        {
            "format_emergency": "format_emergency",
            "knowledge_agent": "knowledge_agent",
        },
    )

    graph.add_edge("knowledge_agent", "retrieval_agent")

    # Conditional: no results → exit, else → ranking
    graph.add_conditional_edges(
        "retrieval_agent",
        _route_after_retrieval,
        {
            "format_no_results": "format_no_results",
            "ranking_agent": "ranking_agent",
        },
    )

    graph.add_edge("ranking_agent", "explanation_agent")
    graph.add_edge("explanation_agent", "format_results")

    # Terminal nodes
    graph.add_edge("format_emergency", END)
    graph.add_edge("format_no_results", END)
    graph.add_edge("format_results", END)

    return graph


# Compile the graph once for reuse
remedy_graph = build_graph().compile()
