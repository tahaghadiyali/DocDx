"""SQLAlchemy + GeoAlchemy2 models for the doctors directory."""

from datetime import datetime

from geoalchemy2 import Geography
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all models."""
    pass


class Doctor(Base):
    """
    A doctor or healthcare provider, enriched with both real OSM geo data
    and synthetic attributes for the demo.
    """

    __tablename__ = "doctors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Identity ──
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    gender: Mapped[str | None] = mapped_column(String(20))  # "male" | "female" | "other"

    # ── Professional ──
    specialty: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    sub_specialization: Mapped[str | None] = mapped_column(String(200))
    years_experience: Mapped[int | None] = mapped_column(Integer)
    board_certifications: Mapped[list | None] = mapped_column(ARRAY(String))
    education: Mapped[str | None] = mapped_column(Text)

    # ── Location (PostGIS geography for accurate distance) ──
    location: Mapped[str] = mapped_column(
        Geography(geometry_type="POINT", srid=4326),
        nullable=False,
    )
    address: Mapped[str | None] = mapped_column(Text)
    city: Mapped[str | None] = mapped_column(String(100), index=True)

    # ── Ratings & Reviews ──
    rating: Mapped[float | None] = mapped_column(Float)
    review_count: Mapped[int | None] = mapped_column(Integer, default=0)

    # ── Cost & Access ──
    consultation_fee: Mapped[float | None] = mapped_column(Float)
    insurance_accepted: Mapped[list | None] = mapped_column(ARRAY(String))
    languages: Mapped[list | None] = mapped_column(ARRAY(String))
    telehealth_available: Mapped[bool] = mapped_column(Boolean, default=False)

    # ── Contact ──
    phone: Mapped[str | None] = mapped_column(String(50))
    website: Mapped[str | None] = mapped_column(Text)
    hospital_name: Mapped[str | None] = mapped_column(String(255))

    # ── Source Metadata ──
    osm_id: Mapped[str | None] = mapped_column(String(50), unique=True)
    source: Mapped[str | None] = mapped_column(String(50))  # "osm" | "synthetic"

    # ── Timestamps ──
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<Doctor(id={self.id}, name='{self.name}', specialty='{self.specialty}')>"


class Session(Base):
    """
    Stores multi-turn conversation session state for refinement.
    """

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    state: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<Session(id='{self.id}')>"
