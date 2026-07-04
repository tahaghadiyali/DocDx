"""
RemedyRadar — Chainlit Application.

Wires the LangGraph orchestrator to Chainlit's chat interface with
visible agent step tracking and multi-turn session support.
"""

import sys
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import sniffio
sniffio.current_async_library_cvar.set("asyncio")

import nest_asyncio
nest_asyncio.apply()

import uuid
import chainlit as cl

from app.agents.orchestrator import remedy_graph
from app.agents.state import AgentState


@cl.on_chat_start
async def on_chat_start():
    """Initialize session state when a new chat starts."""
    session_id = str(uuid.uuid4())[:8]
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("turn_number", 0)
    cl.user_session.set("conversation_history", [])
    cl.user_session.set("last_state", None)

    await cl.Message(
        content="👋 Hi! I'm **RemedyRadar**. Describe your symptoms and tell me your city — I'll find the right specialist for you.\n\n*Example: \"I've been having persistent chest tightness when I exercise, I'm in Bangalore\"*"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages — run the full agent pipeline with visible steps."""
    session_id = cl.user_session.get("session_id")
    turn_number = cl.user_session.get("turn_number", 0) + 1
    cl.user_session.set("turn_number", turn_number)

    history = cl.user_session.get("conversation_history", [])
    last_state = cl.user_session.get("last_state")

    # ── Build initial state ──
    state: AgentState = {
        "raw_query": message.content,
        "session_id": session_id,
        "turn_number": turn_number,
        "conversation_history": history,
    }

    # For multi-turn: carry over location and preferences from last state
    if last_state:
        if last_state.get("location"):
            state["location"] = last_state["location"]
        if last_state.get("preferences"):
            state["preferences"] = last_state["preferences"]
        if last_state.get("applied_filters"):
            state["applied_filters"] = last_state["applied_filters"]

    # ── Run the pipeline with visible agent steps ──

    # Step 1: Intake
    async with cl.Step(name="🧠 Analyzing Symptoms", type="tool") as step:
        step.input = message.content
        state = await remedy_graph.nodes["intake_agent"].invoke(state)
        symptoms = state.get("extracted_symptoms", [])
        urgency = state.get("urgency_level", "routine")
        step.output = f"Extracted: {', '.join(symptoms)} | Urgency: {urgency}"

    # Step 2: Emergency Check
    async with cl.Step(name="🚨 Checking Urgency", type="tool") as step:
        state = await remedy_graph.nodes["emergency_detector"].invoke(state)
        is_emergency = state.get("is_emergency", False)
        step.output = "🚨 EMERGENCY DETECTED" if is_emergency else "✅ No emergency — proceeding"

    if is_emergency:
        # Emergency short-circuit
        state = remedy_graph.nodes["format_emergency"].invoke(state)
        await cl.Message(content=state["final_response"]).send()
        _save_turn(message.content, state)
        return

    # Step 3: Specialty Classification
    async with cl.Step(name="📚 Identifying Specialty", type="tool") as step:
        state = await remedy_graph.nodes["knowledge_agent"].invoke(state)
        specialty = state.get("recommended_specialty", "Unknown")
        confidence = state.get("specialty_confidence", 0)
        step.output = f"**{specialty}** (confidence: {confidence:.0%})"

    # Step 4: Doctor Search
    async with cl.Step(name="🔍 Searching Nearby Doctors", type="tool") as step:
        state = await remedy_graph.nodes["retrieval_agent"].invoke(state)
        n_found = len(state.get("candidate_doctors", []))
        radius = state.get("search_radius_used", 0)
        step.output = f"Found {n_found} candidates within {radius}km"

    if not state.get("candidate_doctors"):
        state = remedy_graph.nodes["format_no_results"].invoke(state)
        await cl.Message(content=state["final_response"]).send()
        _save_turn(message.content, state)
        return

    # Step 5: Ranking
    async with cl.Step(name="📊 Ranking Candidates", type="tool") as step:
        state = await remedy_graph.nodes["ranking_agent"].invoke(state)
        n_ranked = len(state.get("ranked_doctors", []))
        top_name = state["ranked_doctors"][0]["name"] if state.get("ranked_doctors") else "N/A"
        step.output = f"Top {n_ranked} ranked. #1: {top_name}"

    # Step 6: Explanations
    async with cl.Step(name="✍️ Generating Explanations", type="tool") as step:
        state = await remedy_graph.nodes["explanation_agent"].invoke(state)
        step.output = f"Generated personalized explanations for {len(state.get('doctor_explanations', []))} doctors"

    # Format and send final results
    state = remedy_graph.nodes["format_results"].invoke(state)
    await cl.Message(content=state["final_response"]).send()

    # Save state for multi-turn
    _save_turn(message.content, state)


def _save_turn(user_message: str, state: AgentState):
    """Save the current turn to session for multi-turn refinement."""
    history = cl.user_session.get("conversation_history", [])
    history.append({
        "role": "user",
        "content": user_message,
    })
    history.append({
        "role": "assistant",
        "specialty": state.get("recommended_specialty"),
        "n_results": len(state.get("ranked_doctors", [])),
    })
    cl.user_session.set("conversation_history", history)
    cl.user_session.set("last_state", {
        "location": state.get("location"),
        "preferences": state.get("preferences"),
        "applied_filters": state.get("applied_filters"),
        "recommended_specialty": state.get("recommended_specialty"),
    })
