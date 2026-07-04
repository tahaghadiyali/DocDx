"""
Explanation Agent — generates grounded per-doctor explanations using RAG Store #2.

For each top-ranked doctor, retrieves their profile/review data and generates
a natural-language "why this doctor fits you" summary.
"""

import json
import openai

from app.config import settings
from app.agents.state import AgentState
from app.rag.profile_store import query_doctor_profile


EXPLANATION_SYSTEM_PROMPT = """You are a medical recommendation assistant. Generate a brief, helpful explanation of why a specific doctor is a good fit for a patient's needs.

RULES:
- Use ONLY the provided doctor profile and review information. Do NOT invent credentials, qualifications, or review quotes.
- Keep it to 2-3 sentences maximum.
- Mention specific relevant strengths (e.g., sub-specialization match, years of experience, patient review highlights).
- Be honest — if the doctor has lower ratings, frame it neutrally.
- Use a warm, reassuring tone.

Respond with ONLY the explanation text (no JSON, no formatting)."""


async def _generate_single_explanation(
    doctor: dict,
    symptoms: list[str],
    profile_context: str,
) -> str:
    """Generate a natural-language explanation for one doctor."""
    user_prompt = f"""DOCTOR PROFILE AND REVIEWS:
{profile_context}

PATIENT'S SYMPTOMS: {', '.join(symptoms)}

DOCTOR SCORE SUMMARY:
- Overall Score: {doctor.get('score', 0):.0%}
- Rating: {doctor.get('rating', 'N/A')}/5 ({doctor.get('review_count', 0)} reviews)
- Experience: {doctor.get('years_experience', 'N/A')} years
- Distance: {doctor.get('distance_km', 'N/A')} km away
- Fee: {doctor.get('consultation_fee', 'N/A')}
- Telehealth: {'Yes' if doctor.get('telehealth_available') else 'No'}

Write a 2-3 sentence explanation of why this doctor is a good fit for this patient."""

    try:
        client = openai.AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1.0,
            extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback: template-based explanation
        return (
            f"{doctor['name']} is a {doctor['specialty']} with "
            f"{doctor.get('years_experience', 'several')} years of experience "
            f"and a {doctor.get('rating', 'N/A')}/5 rating from patients."
        )


async def run_explanation_agent(state: AgentState) -> AgentState:
    """
    Generate grounded explanations for each top-ranked doctor.
    Queries RAG Store #2 for profile/review context per doctor.
    """
    ranked_doctors = state.get("ranked_doctors", [])
    symptoms = state.get("extracted_symptoms", [])

    explanations = []
    rag_sources = []

    for doctor in ranked_doctors:
        # Retrieve profile/review chunks from RAG Store #2
        try:
            profile_hits = query_doctor_profile(
                doctor_name=doctor["name"],
                specialty=doctor["specialty"],
                n_results=5,
            )
        except Exception:
            profile_hits = []

        # Build context from retrieved chunks
        if profile_hits:
            profile_context = "\n".join([h["document"] for h in profile_hits])
            rag_sources.extend([
                {
                    "doctor_name": doctor["name"],
                    "chunk_type": h["metadata"].get("type", "unknown"),
                    "source_id": h["id"],
                }
                for h in profile_hits
            ])
        else:
            # Fallback: build context from structured fields
            profile_context = (
                f"Doctor: {doctor['name']}\n"
                f"Specialty: {doctor['specialty']}\n"
                f"Experience: {doctor.get('years_experience', 'N/A')} years\n"
                f"Hospital: {doctor.get('hospital_name', 'N/A')}\n"
                f"Rating: {doctor.get('rating', 'N/A')}/5\n"
            )

        # Generate explanation
        explanation = await _generate_single_explanation(
            doctor, symptoms, profile_context
        )

        explanations.append({
            "doctor_id": doctor.get("id"),
            "doctor_name": doctor["name"],
            "explanation": explanation,
        })

    state["doctor_explanations"] = explanations
    state["rag_sources_profiles"] = rag_sources

    return state
