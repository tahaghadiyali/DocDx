"""Geo-aware doctor queries using PostGIS."""

from sqlalchemy import select, func, text, cast, Float
from sqlalchemy.ext.asyncio import AsyncSession
from geoalchemy2 import Geography
from geoalchemy2.functions import ST_DWithin, ST_Distance, ST_MakePoint, ST_SetSRID

from app.db.models import Doctor


async def search_doctors_nearby(
    session: AsyncSession,
    lat: float,
    lng: float,
    specialty: str,
    radius_m: float = 10_000,
    limit: int = 30,
    filters: dict | None = None,
) -> list[dict]:
    """
    Find doctors of a given specialty within radius_m meters of (lat, lng).

    Uses ST_DWithin for index-assisted filtering + <-> operator for
    nearest-neighbor sorting. Returns dicts with distance_km added.
    """
    filters = filters or {}

    user_point = func.ST_SetSRID(func.ST_MakePoint(lng, lat), 4326).cast(Geography)

    stmt = (
        select(
            Doctor,
            func.ST_Distance(Doctor.location, user_point).label("distance_m"),
        )
        .where(
            Doctor.specialty == specialty,
            func.ST_DWithin(Doctor.location, user_point, radius_m),
        )
        .order_by(Doctor.location.op("<->")(user_point))
        .limit(limit)
    )

    # Apply optional structured filters
    if filters.get("gender"):
        stmt = stmt.where(Doctor.gender == filters["gender"])
    if filters.get("telehealth"):
        stmt = stmt.where(Doctor.telehealth_available == True)
    if filters.get("language"):
        stmt = stmt.where(Doctor.languages.any(filters["language"]))
    if filters.get("max_fee"):
        stmt = stmt.where(Doctor.consultation_fee <= filters["max_fee"])

    result = await session.execute(stmt)
    rows = result.all()

    doctors = []
    for doctor, distance_m in rows:
        doc_dict = {
            "id": doctor.id,
            "name": doctor.name,
            "gender": doctor.gender,
            "specialty": doctor.specialty,
            "sub_specialization": doctor.sub_specialization,
            "years_experience": doctor.years_experience,
            "board_certifications": doctor.board_certifications,
            "education": doctor.education,
            "address": doctor.address,
            "city": doctor.city,
            "rating": doctor.rating,
            "review_count": doctor.review_count,
            "consultation_fee": doctor.consultation_fee,
            "insurance_accepted": doctor.insurance_accepted,
            "languages": doctor.languages,
            "telehealth_available": doctor.telehealth_available,
            "phone": doctor.phone,
            "website": doctor.website,
            "hospital_name": doctor.hospital_name,
            "distance_km": round(distance_m / 1000, 2) if distance_m else None,
        }
        doctors.append(doc_dict)

    return doctors


async def search_emergency_rooms(
    session: AsyncSession,
    lat: float,
    lng: float,
    limit: int = 3,
) -> list[dict]:
    """Find nearest hospitals/ERs regardless of specialty."""
    user_point = func.ST_SetSRID(func.ST_MakePoint(lng, lat), 4326).cast(Geography)

    stmt = (
        select(
            Doctor,
            func.ST_Distance(Doctor.location, user_point).label("distance_m"),
        )
        .where(func.ST_DWithin(Doctor.location, user_point, 50_000))  # 50km max
        .order_by(Doctor.location.op("<->")(user_point))
        .limit(limit)
    )

    result = await session.execute(stmt)
    rows = result.all()

    return [
        {
            "name": doc.name,
            "hospital_name": doc.hospital_name,
            "address": doc.address,
            "phone": doc.phone,
            "distance_km": round(dist / 1000, 2) if dist else None,
        }
        for doc, dist in rows
    ]
