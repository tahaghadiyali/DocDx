"""FastAPI app — thin internal API layer for health checks and optional REST access."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # Startup: nothing heavy here — DB and ChromaDB init happen via scripts
    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="RemedyRadar",
    description="Agentic health specialist finder API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/api/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "RemedyRadar",
        "llm_model": settings.llm_model,
        "embed_model": settings.embed_model,
    }
