"""RAG Store #2 — Doctor profiles and reviews for explanation generation."""

import json
from pathlib import Path

import chromadb

from app.config import settings
from app.rag.embeddings import get_embedding_function

COLLECTION_NAME = "doctor_profiles"
DATA_DIR = Path(__file__).parent.parent / "data" / "seed"


def get_profile_store() -> chromadb.Collection:
    """Get or create the doctor profiles ChromaDB collection."""
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"description": "Doctor bios, credentials, and patient reviews"},
    )
    return collection


def ingest_doctor_profiles(doctors: list[dict], reviews: list[dict]) -> int:
    """
    Ingest doctor bios and reviews into ChromaDB.
    Returns the number of documents ingested.
    """
    collection = get_profile_store()

    if collection.count() > 0:
        print(f"  Profile store already has {collection.count()} documents. Skipping.")
        return collection.count()

    documents = []
    metadatas = []
    ids = []

    # ── Doctor bios ──
    for doc in doctors:
        bio_text = (
            f"Doctor: {doc['name']}\n"
            f"Specialty: {doc['specialty']}\n"
        )
        if doc.get("sub_specialization"):
            bio_text += f"Sub-specialization: {doc['sub_specialization']}\n"
        if doc.get("years_experience"):
            bio_text += f"Years of Experience: {doc['years_experience']}\n"
        if doc.get("education"):
            bio_text += f"Education: {doc['education']}\n"
        if doc.get("board_certifications"):
            bio_text += f"Board Certifications: {', '.join(doc['board_certifications'])}\n"
        if doc.get("hospital_name"):
            bio_text += f"Hospital/Clinic: {doc['hospital_name']}\n"
        if doc.get("languages"):
            bio_text += f"Languages: {', '.join(doc['languages'])}\n"
        if doc.get("telehealth_available"):
            bio_text += "Offers telehealth consultations.\n"

        documents.append(bio_text)
        metadatas.append({
            "type": "bio",
            "doctor_id": str(doc.get("id", "")),
            "doctor_name": doc["name"],
            "specialty": doc["specialty"],
        })
        ids.append(f"bio-{doc.get('id', doc['name'].lower().replace(' ', '-'))}")

    # ── Patient reviews ──
    for review in reviews:
        doc_text = (
            f"Patient review for Dr. {review['doctor_name']}:\n"
            f"Rating: {review['rating']}/5\n"
            f"Review: {review['text']}\n"
        )
        documents.append(doc_text)
        metadatas.append({
            "type": "review",
            "doctor_id": str(review.get("doctor_id", "")),
            "doctor_name": review["doctor_name"],
            "rating": review["rating"],
        })
        ids.append(f"review-{review.get('id', '')}")

    if documents:
        # ChromaDB has a batch limit, add in chunks of 100
        for i in range(0, len(documents), 100):
            collection.add(
                documents=documents[i:i+100],
                metadatas=metadatas[i:i+100],
                ids=ids[i:i+100],
            )
        print(f"  Ingested {len(documents)} documents into doctor profile store.")

    return len(documents)


def query_doctor_profile(doctor_name: str, specialty: str, n_results: int = 5) -> list[dict]:
    """
    Query the profile store for a specific doctor's bio and reviews.
    """
    collection = get_profile_store()
    results = collection.query(
        query_texts=[f"Doctor {doctor_name} {specialty}"],
        n_results=n_results,
        where={"doctor_name": doctor_name},
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits
