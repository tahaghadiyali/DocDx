"""
Build / rebuild ChromaDB vector stores from data files.

Ingests:
  1. Medical knowledge base (conditions + specialties) → RAG Store #1
  2. Doctor profiles and reviews → RAG Store #2
"""

import json
from pathlib import Path

from app.rag.knowledge_store import ingest_medical_knowledge
from app.rag.profile_store import ingest_doctor_profiles

DATA_DIR = Path(__file__).parent.parent / "app" / "data" / "seed"


def build_all():
    """Build both vector stores."""
    print("=" * 60)
    print("Building RemedyRadar vector stores...")
    print("=" * 60)

    # -- RAG Store #1: Medical Knowledge --
    print("\n[1/2] Medical Knowledge Base (RAG Store #1)")
    count1 = ingest_medical_knowledge()
    print(f"  -> {count1} documents in knowledge store")

    # -- RAG Store #2: Doctor Profiles --
    print("\n[2/2] Doctor Profiles & Reviews (RAG Store #2)")
    doctors_path = DATA_DIR / "doctors.json"
    reviews_path = DATA_DIR / "reviews.json"

    if not doctors_path.exists():
        print("  ⚠ No doctors.json found. Run generate_synthetic.py first.")
        return

    with open(doctors_path, "r", encoding="utf-8") as f:
        doctors = json.load(f)
    
    reviews = []
    if reviews_path.exists():
        with open(reviews_path, "r", encoding="utf-8") as f:
            reviews = json.load(f)

    count2 = ingest_doctor_profiles(doctors, reviews)
    print(f"  → {count2} documents in profile store")

    print("\n" + "=" * 60)
    print(f"Done! Total: {count1 + count2} documents across 2 stores.")
    print("=" * 60)


if __name__ == "__main__":
    build_all()
