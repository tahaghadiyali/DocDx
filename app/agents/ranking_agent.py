"""
Ranking Agent — deterministic weighted scoring function.

NO LLM CALL. Pure Python computation. This is intentional:
ranking structured data is a math problem, not a language problem.
Transparent, debuggable, and fast.
"""

from app.config import settings
from app.agents.state import AgentState


# Default scoring weights
DEFAULT_WEIGHTS = {
    "specialty_match": 0.30,
    "rating": 0.25,
    "experience": 0.15,
    "distance": 0.15,
    "cost": 0.10,
    "availability": 0.05,
}


def _normalize(value: float | None, min_val: float, max_val: float) -> float:
    """Normalize a value to 0-1 range. Returns 0.5 if value is None."""
    if value is None:
        return 0.5
    if max_val == min_val:
        return 1.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def _compute_score(
    doctor: dict,
    specialty_confidence: float,
    max_distance: float,
    max_fee: float,
    weights: dict | None = None,
) -> tuple[float, dict]:
    """
    Compute a transparent composite score for a single doctor.

    Returns (total_score, score_breakdown_dict).
    """
    w = weights or DEFAULT_WEIGHTS

    scores = {
        "specialty_match": specialty_confidence,
        "rating": _normalize(doctor.get("rating"), 1.0, 5.0),
        "experience": _normalize(doctor.get("years_experience"), 0, 40),
        "distance": 1.0 - _normalize(doctor.get("distance_km"), 0, max_distance) if doctor.get("distance_km") is not None else 0.5,
        "cost": 1.0 - _normalize(doctor.get("consultation_fee"), 0, max_fee) if doctor.get("consultation_fee") is not None else 0.5,
        "availability": 0.7 + (0.3 if doctor.get("telehealth_available") else 0.0),
    }

    total = sum(w[k] * scores[k] for k in w if k in scores)

    # Format breakdown for display
    breakdown = {
        k: {
            "raw_score": round(scores.get(k, 0), 3),
            "weight": w.get(k, 0),
            "weighted": round(w.get(k, 0) * scores.get(k, 0), 3),
        }
        for k in w
    }

    return round(total, 4), breakdown


async def run_ranking_agent(state: AgentState) -> AgentState:
    """
    Rank candidate doctors using a weighted scoring function.
    Pure computation — no LLM call, no API cost, fully deterministic.
    """
    candidates = state.get("candidate_doctors", [])
    specialty_confidence = state.get("specialty_confidence", 0.8)
    top_n = settings.top_n_doctors

    if not candidates:
        state["ranked_doctors"] = []
        return state

    # Compute max values for normalization
    max_distance = max(
        (d.get("distance_km", 0) or 0 for d in candidates), default=10
    )
    max_fee = max(
        (d.get("consultation_fee", 0) or 0 for d in candidates), default=1000
    )

    # Prevent division by zero
    max_distance = max(max_distance, 1.0)
    max_fee = max(max_fee, 1.0)

    # Score each doctor
    scored = []
    for doctor in candidates:
        score, breakdown = _compute_score(
            doctor, specialty_confidence, max_distance, max_fee
        )
        doctor_with_score = {**doctor, "score": score, "score_breakdown": breakdown}
        scored.append(doctor_with_score)

    # Sort by score descending
    scored.sort(key=lambda d: d["score"], reverse=True)

    # Return top N
    state["ranked_doctors"] = scored[:top_n]
    return state
