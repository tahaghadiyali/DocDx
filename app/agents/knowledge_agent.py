"""
Knowledge Agent — maps symptoms to specialties using RAG Store #1.

Queries the medical knowledge base, retrieves relevant condition/specialty
information, and uses the LLM to determine the best-fit specialty with
a grounded explanation.
"""

import json
import openai

from app.config import settings
from app.agents.state import AgentState
from app.rag.knowledge_store import query_knowledge


KNOWLEDGE_SYSTEM_PROMPT = """You are a medical specialty classifier. Given a patient's symptoms and reference medical information, determine the most appropriate medical specialty.

IMPORTANT RULES:
- You do NOT diagnose conditions. You only recommend which type of specialist to see.
- Base your recommendation on the provided reference information, NOT your own medical knowledge.
- Cite the specific reference sources you used.
- If multiple specialties could be relevant, pick the most likely one as primary and mention alternatives.
- Always explain WHY this specialty is recommended in plain language.

Respond with ONLY a valid JSON object:
{
  "specialty": "Specialty Name",
  "confidence": 0.85,
  "explanation": "Plain-language explanation of why this specialty, 2-3 sentences",
  "alternative_specialties": ["Alternative1"],
  "cited_sources": ["Source 1", "Source 2"]
}"""


async def run_knowledge_agent(state: AgentState) -> AgentState:
    """
    Query RAG Store #1 and use LLM to map symptoms → specialty.
    """
    symptoms = state.get("extracted_symptoms", [])
    conditions = state.get("extracted_conditions", [])

    # Build the query from symptoms + conditions
    query_text = "Patient symptoms: " + ", ".join(symptoms)
    if conditions:
        query_text += ". Mentioned conditions: " + ", ".join(conditions)

    # ── Retrieve from RAG Store #1 ──
    retrieved_chunks = query_knowledge(query_text, n_results=5)

    # Format retrieved context for the LLM
    context_text = "REFERENCE MEDICAL INFORMATION:\n\n"
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_text += f"--- Reference {i} (Source: {chunk['metadata'].get('source', 'Unknown')}) ---\n"
        context_text += chunk["document"] + "\n\n"

    # ── LLM call to classify specialty ──
    user_prompt = f"""{context_text}

PATIENT QUERY:
Symptoms: {', '.join(symptoms)}
{f'Conditions mentioned: {", ".join(conditions)}' if conditions else ''}

Based ONLY on the reference information above, what medical specialty should this patient see?"""

    client = openai.AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": KNOWLEDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=1.0,
        extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384}
    )

    # Parse response
    try:
        parsed = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, KeyError, AttributeError, Exception):
        # Fallback: use the top retrieved chunk's specialty
        fallback_specialty = "General Practitioner"
        if retrieved_chunks:
            fallback_specialty = retrieved_chunks[0]["metadata"].get("specialty", "General Practitioner")
        parsed = {
            "specialty": fallback_specialty,
            "confidence": 0.5,
            "explanation": f"Based on your symptoms, a {fallback_specialty} would be a good starting point.",
            "cited_sources": [],
        }

    state["recommended_specialty"] = parsed.get("specialty", "General Practitioner")
    state["specialty_confidence"] = parsed.get("confidence", 0.5)
    state["specialty_explanation"] = parsed.get("explanation", "")
    state["rag_sources_medical"] = [
        {
            "source": chunk["metadata"].get("source", "Unknown"),
            "condition": chunk["metadata"].get("condition_name", ""),
            "specialty": chunk["metadata"].get("specialty", ""),
            "relevance": round(1 - chunk["distance"], 3) if chunk.get("distance") else None,
        }
        for chunk in retrieved_chunks
    ]

    return state
