"""
DocsQuery - Main Application

This file creates the FastAPI application.

At this stage, we only have a health-check endpoint.
As the project grows, routers for documents, search,
queries, evaluation, etc. will be added here.
"""

from fastapi import FastAPI

# ------------------------------------------------------------
# Create the FastAPI application
# ------------------------------------------------------------
# FastAPI() creates the web application object.
#
# title:
#     Name shown in the automatic API documentation.
#
# description:
#     Short explanation of what our API does.
#
# version:
#     Current API/application version.
# ------------------------------------------------------------

app = FastAPI(
    title="DocsQuery",
    description="Production-oriented domain-specific RAG application",
    version="0.1.0",
)


# ------------------------------------------------------------
# Health Check Endpoint
# ------------------------------------------------------------
# GET /health
#
# This endpoint is used to verify that the application is
# running correctly.
#
# Later, this endpoint can also be expanded to check:
# - Database connectivity
# - Qdrant connectivity
# - Required services
# - Application dependencies
# ------------------------------------------------------------


@app.get("/health")
def health_check():
    """
    Return the current health status of the application.
    """

    return {"status": "ok"}
