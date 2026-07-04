"""
Intake Agent — NLU + entity extraction from free-text symptom descriptions.

Extracts: symptoms, conditions, urgency signals, location, and preferences
using Ollama's Llama 3.1 with structured JSON output.
"""

import json
import openai

from app.config import settings
from app.agents.state import AgentState


INTAKE_SYSTEM_PROMPT = """You are a medical intake assistant. Your job is to extract structured information from a patient's free-text description of their symptoms or health concern.

IMPORTANT RULES:
- You do NOT diagnose. You only extract what the user describes.
- Extract symptoms as simple phrases (e.g., "chest pain", "shortness of breath").
- Assess urgency: "emergency" (life-threatening), "urgent" (needs prompt care), or "routine" (can wait for appointment).
- If the user mentions a location, extract it.
- If the user mentions preferences (gender of doctor, language, budget, telehealth), extract those.

Respond with ONLY a valid JSON object in this exact format:
{
  "symptoms": ["symptom1", "symptom2"],
  "conditions": ["any conditions mentioned"],
  "urgency_level": "routine|urgent|emergency",
  "urgency_reason": "brief explanation of urgency assessment",
  "location_mentioned": "city or area if mentioned, null otherwise",
  "preferences": {
    "gender": "male|female|null",
    "language": "language preference or null",
    "max_fee": null,
    "telehealth": false
  }
}"""


async def run_intake_agent(state: AgentState) -> AgentState:
    """
    Extract medical entities and urgency from the user's free-text query.
    Uses Ollama's Llama 3.1 with JSON mode for structured output.
    """
    raw_query = state["raw_query"]

    # Call LLM with JSON format
    client = openai.AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": INTAKE_SYSTEM_PROMPT},
            {"role": "user", "content": raw_query},
        ],
        response_format={"type": "json_object"},
        temperature=1.0,
        extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384}
    )
    
    # Parse the LLM's structured output
    try:
        content = response.choices[0].message.content
        parsed = json.loads(content)
    except (json.JSONDecodeError, KeyError, AttributeError, Exception):
        # Fallback: treat the whole query as a single symptom
        parsed = {
            "symptoms": [raw_query],
            "conditions": [],
            "urgency_level": "routine",
            "urgency_reason": "Could not parse — defaulting to routine",
            "location_mentioned": None,
            "preferences": {},
        }

    # Update state with extracted information
    state["extracted_symptoms"] = parsed.get("symptoms", [raw_query])
    state["extracted_conditions"] = parsed.get("conditions", [])
    state["urgency_level"] = parsed.get("urgency_level", "routine")
    state["medical_entities"] = [
        {"type": "symptom", "value": s} for s in parsed.get("symptoms", [])
    ] + [
        {"type": "condition", "value": c} for c in parsed.get("conditions", [])
    ]

    # Merge any extracted preferences with existing preferences
    existing_prefs = state.get("preferences", {})
    extracted_prefs = parsed.get("preferences", {})
    for key, value in extracted_prefs.items():
        if value and not existing_prefs.get(key):
            existing_prefs[key] = value
    state["preferences"] = existing_prefs

    # If a location was mentioned and we don't have one yet, try to use it
    if parsed.get("location_mentioned") and not state.get("location"):
        state["location"] = {"city": parsed["location_mentioned"]}

    return state
