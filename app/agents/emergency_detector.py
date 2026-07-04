"""
Emergency Detector — two-layer emergency detection.

Layer 1: Deterministic keyword/pattern matching (fast, zero false negatives)
Layer 2: LLM confirmation for borderline cases (slower, more nuanced)
"""

import json
from pathlib import Path

import openai

from app.config import settings
from app.agents.state import AgentState

# Load red flags data at module level
RED_FLAGS_PATH = Path(__file__).parent.parent / "data" / "medical_kb" / "red_flags.json"
with open(RED_FLAGS_PATH, "r", encoding="utf-8") as f:
    RED_FLAGS_DATA = json.load(f)

INSTANT_KEYWORDS = [k.lower() for k in RED_FLAGS_DATA["instant_emergency_keywords"]]
EMERGENCY_PATTERNS = RED_FLAGS_DATA["emergency_patterns"]


def _check_instant_keywords(text: str) -> str | None:
    """Layer 1: Check for instant emergency keywords in the raw query."""
    text_lower = text.lower()
    for keyword in INSTANT_KEYWORDS:
        if keyword in text_lower:
            # Find the matching pattern for a relevant message
            for pattern in EMERGENCY_PATTERNS:
                for pk in pattern["keywords"]:
                    if pk.lower() in text_lower:
                        return pattern["message"]
            # Generic emergency if no specific pattern matched
            return (
                "⚠️ Your description contains concerning symptoms that may require "
                "immediate medical attention. Please call emergency services (911/112) "
                "or go to your nearest emergency room immediately."
            )
    return None


def _check_symptom_combinations(symptoms: list[str]) -> str | None:
    """Check extracted symptoms against known emergency combinations."""
    symptoms_lower = [s.lower() for s in symptoms]
    symptoms_text = " ".join(symptoms_lower)

    for pattern in EMERGENCY_PATTERNS:
        for combo in pattern.get("symptom_combinations", []):
            combo_lower = [c.lower() for c in combo]
            if all(any(c in s for s in symptoms_lower) or c in symptoms_text for c in combo_lower):
                return pattern["message"]
    return None


async def _llm_emergency_check(symptoms: list[str], raw_query: str) -> tuple[bool, str]:
    """
    Layer 2: LLM confirmation for borderline cases.
    Only called if Layer 1 didn't trigger but urgency_level is 'urgent'.
    """
    prompt = f"""A patient reports these symptoms: {', '.join(symptoms)}

Original description: "{raw_query}"

Is this a MEDICAL EMERGENCY requiring immediate emergency care (ER/911)?
Consider: Is there an immediate threat to life or risk of permanent harm?

Respond with ONLY a JSON object:
{{"is_emergency": true/false, "reason": "brief explanation"}}"""

    try:
        client = openai.AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "You are an emergency triage assistant. Be conservative — when in doubt, flag as emergency. Better safe than sorry."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=1.0,
            extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384}
        )
        parsed = json.loads(response.choices[0].message.content)
        return parsed.get("is_emergency", False), parsed.get("reason", "")
    except Exception:
        # If LLM fails, err on the side of caution for urgent cases
        return False, "LLM check unavailable"


async def run_emergency_detector(state: AgentState) -> AgentState:
    """
    Two-layer emergency detection.

    Layer 1 (deterministic): keyword + pattern matching — fast, reliable
    Layer 2 (LLM): confirmation for borderline 'urgent' cases
    """
    raw_query = state.get("raw_query", "")
    symptoms = state.get("extracted_symptoms", [])
    urgency = state.get("urgency_level", "routine")

    # ── Layer 1: Instant keyword check ──
    emergency_msg = _check_instant_keywords(raw_query)
    if emergency_msg:
        state["is_emergency"] = True
        state["emergency_message"] = emergency_msg
        return state

    # ── Layer 1b: Symptom combination check ──
    emergency_msg = _check_symptom_combinations(symptoms)
    if emergency_msg:
        state["is_emergency"] = True
        state["emergency_message"] = emergency_msg
        return state

    # ── Layer 2: LLM check for borderline urgent cases ──
    if urgency == "emergency":
        # Intake Agent flagged it — trust the assessment
        state["is_emergency"] = True
        state["emergency_message"] = (
            "⚠️ Based on your symptoms, this may require immediate medical attention. "
            "Please call emergency services (911/112) or go to your nearest emergency room."
        )
        return state

    if urgency == "urgent":
        # Borderline — ask the LLM to confirm
        is_emg, reason = await _llm_emergency_check(symptoms, raw_query)
        if is_emg:
            state["is_emergency"] = True
            state["emergency_message"] = (
                f"⚠️ Your symptoms may require immediate medical attention: {reason}. "
                "Please call emergency services (911/112) or go to your nearest emergency room."
            )
            return state

    # ── Not an emergency ──
    state["is_emergency"] = False
    state["emergency_message"] = ""
    return state
