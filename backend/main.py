import os
import sys
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Configure logging so all modules output to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Ensure backend directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from database import init_db
from routers import transcribe


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    print("Database initialized.")
    yield
    # Shutdown
    print("Shutting down.")


app = FastAPI(
    title="Exhibit G Transcriber",
    description="SAG-AFTRA Exhibit G form transcription tool",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS - allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded/processed images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/uploads", StaticFiles(directory=os.path.join(BASE_DIR, "uploads")), name="uploads")
app.mount("/processed", StaticFiles(directory=os.path.join(BASE_DIR, "processed")), name="processed")

# Include routers
app.include_router(transcribe.router, prefix="/api/transcribe", tags=["transcribe"])


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "Exhibit G Transcriber"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
