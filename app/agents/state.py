"""
AgentState — shared state flowing through the LangGraph pipeline.

Every agent reads from and writes to this TypedDict. LangGraph manages
the state transitions between nodes.
"""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state across all agents in the RemedyRadar pipeline.

    Fields are grouped by which agent produces them.
    `total=False` means all fields are optional — they get populated
    as the pipeline progresses through each node.
    """

    # ── Input (from user) ──
    raw_query: str
    location: dict          # { "lat": float, "lng": float, "city": str, "radius_km": float }
    preferences: dict       # { "gender": str, "language": str, "max_fee": float, "telehealth": bool }
    session_id: str
    turn_number: int

    # ── Intake Agent output ──
    extracted_symptoms: list[str]
    extracted_conditions: list[str]
    urgency_level: str      # "emergency" | "urgent" | "routine"
    medical_entities: list[dict]

    # ── Emergency Detector output ──
    is_emergency: bool
    emergency_message: str

    # ── Knowledge Agent output ──
    recommended_specialty: str
    specialty_confidence: float
    specialty_explanation: str
    rag_sources_medical: list[dict]

    # ── Retrieval Agent output ──
    candidate_doctors: list[dict]
    search_radius_used: float

    # ── Ranking Agent output ──
    ranked_doctors: list[dict]  # with "score" and "score_breakdown" fields

    # ── Explanation Agent output ──
    doctor_explanations: list[dict]  # per-doctor NL explanation
    rag_sources_profiles: list[dict]

    # ── Final output ──
    final_response: str

    # ── Multi-turn memory ──
    conversation_history: list[dict]
    applied_filters: dict

    # ── Errors ──
    error: str | None
