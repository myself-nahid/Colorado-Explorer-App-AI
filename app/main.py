from fastapi import FastAPI
from app.api.v1.endpoints import guide

app = FastAPI(
    title="Colorado Explorer AI Guide API",
    description="API for generating personalized travel guides for Colorado.",
    version="1.0.0"
)

app.include_router(guide.router, prefix="/api/v1", tags=["AI Guide"])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Colorado Explorer AI Guide API"}