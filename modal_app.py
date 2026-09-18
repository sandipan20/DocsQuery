"""
DocsQuery - Modal Deployment Adapter

Deploys the existing FastAPI application to Modal.

The actual application and Docker image remain unchanged.
Modal simply:
    1. Builds the existing Dockerfile.
    2. Provides production environment variables.
    3. Injects Gemini/Qdrant secrets.
    4. Exposes the existing FastAPI application.
"""

import modal

app = modal.App("docsquery-api")

# Reuse the Docker image we already tested locally.
image = modal.Image.from_dockerfile(
    "./Dockerfile",
    context_dir=".",
)

# Non-secret production configuration.
production_env = {
    "APP_ENV": "production",
    "APP_NAME": "DocsQuery",
    "APP_VERSION": "0.1.0",
    "DEBUG": "false",
    "GEMINI_MODEL": "gemini-3.5-flash-lite",
    "GEMINI_TEMPERATURE": "0.0",
    "GEMINI_MAX_TOKENS": "1000",
    "QDRANT_URL": (
        "https://a09f0871-bc56-4611-b6c4-e8f345aa0068."
        "eu-central-1-0.aws.cloud.qdrant.io"
    ),
    "QDRANT_COLLECTION": "docsquery_chunks",
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "TOP_K_DENSE": "20",
    "TOP_K_BM25": "20",
    "RERANKER_TOP_K": "5",
    "VECTOR_CONFIDENCE_THRESHOLD": "0.54",
    "API_CORS_ORIGINS": "[]",
}


@app.function(
    image=image,
    env=production_env,
    secrets=[
        modal.Secret.from_name("docsquery-production"),
    ],
    cpu=2,
    memory=4096,
    timeout=600,
    startup_timeout=600,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    """
    Expose the existing DocsQuery FastAPI application.

    The application itself is not duplicated here.
    """

    from app.main import app as fastapi_app

    return fastapi_app
