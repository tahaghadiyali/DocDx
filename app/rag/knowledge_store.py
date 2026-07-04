"""RAG Store #1 — Medical Knowledge Base (conditions → specialties)."""

import json
from pathlib import Path

import chromadb

from app.config import settings
from app.rag.embeddings import get_embedding_function

COLLECTION_NAME = "medical_knowledge"
DATA_DIR = Path(__file__).parent.parent / "data" / "medical_kb"


def get_knowledge_store() -> chromadb.Collection:
    """Get or create the medical knowledge ChromaDB collection."""
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"description": "Medical conditions → specialty mappings from MedlinePlus + curated sources"},
    )
    return collection


def ingest_medical_knowledge() -> int:
    """
    Ingest conditions.json + specialties.json into ChromaDB.
    Returns the number of documents ingested.
    """
    collection = get_knowledge_store()

    # Check if already ingested
    if collection.count() > 0:
        print(f"  Knowledge store already has {collection.count()} documents. Skipping ingestion.")
        return collection.count()

    documents = []
    metadatas = []
    ids = []

    # ── Ingest conditions ──
    conditions_path = DATA_DIR / "conditions.json"
    with open(conditions_path, "r", encoding="utf-8") as f:
        conditions = json.load(f)

    for condition in conditions:
        # Build a rich text chunk for embedding
        symptoms_str = ", ".join(condition["symptoms"])
        red_flags_str = ", ".join(condition.get("red_flags", []))

        doc_text = (
            f"Condition: {condition['condition']}\n"
            f"Recommended Specialty: {condition['specialty']}\n"
            f"Body System: {condition['body_system']}\n"
            f"Symptoms: {symptoms_str}\n"
            f"Description: {condition['description']}\n"
            f"Urgency Level: {condition['urgency']}\n"
        )
        if red_flags_str:
            doc_text += f"Red Flags (seek emergency care): {red_flags_str}\n"

        documents.append(doc_text)
        metadatas.append({
            "type": "condition",
            "specialty": condition["specialty"],
            "body_system": condition["body_system"],
            "urgency": condition["urgency"],
            "source": condition.get("source", "Curated"),
            "condition_name": condition["condition"],
        })
        ids.append(condition["id"])

    # ── Ingest specialty descriptions ──
    specialties_path = DATA_DIR / "specialties.json"
    with open(specialties_path, "r", encoding="utf-8") as f:
        specialties_data = json.load(f)

    for spec in specialties_data["specialties"]:
        conditions_str = ", ".join(spec["common_conditions"])
        doc_text = (
            f"Medical Specialty: {spec['name']}\n"
            f"Description: {spec['description']}\n"
            f"Common Conditions Treated: {conditions_str}\n"
            f"ICD-10 Chapters: {', '.join(spec['icd10_chapters'])}\n"
        )
        documents.append(doc_text)
        metadatas.append({
            "type": "specialty",
            "specialty": spec["name"],
            "source": "ICD-10 / Curated",
        })
        ids.append(f"spec-{spec['name'].lower().replace('/', '-').replace(' ', '-')}")

    # Batch add to ChromaDB
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"  Ingested {len(documents)} documents into medical knowledge store.")
    return len(documents)


def query_knowledge(query: str, n_results: int = 5) -> list[dict]:
    """
    Query the medical knowledge base for relevant conditions/specialties.
    Returns list of {document, metadata, distance}.
    """
    collection = get_knowledge_store()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
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
