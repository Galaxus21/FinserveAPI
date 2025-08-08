# main.py
# To run this code:
# 1. Create a .env file with your GENAI_KEY
# 2. Install dependencies from requirements.txt: pip install -r requirements.txt
# 3. Run the server: uvicorn main:app --reload

import os
from fastapi import FastAPI, Depends, HTTPException, status, APIRouter, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Import the RAG logic
from rag_pipeline import RAGPipeline

# --- Environment and Configuration ---
load_dotenv() # Load environment variables from a .env file

# Load secrets and configs from environment variables
# In a real app, use a more robust secrets management system.
SECURITY_TOKEN = os.getenv("SECURITY_TOKEN", "128c33fc16f4a70cab19dab48958d5bf246e7003a8bfd7eb0be2f617b48e662a")
GENAI_KEY = os.getenv("GENAI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# This dictionary will hold our initialized RAG pipeline instance.
# It's populated during the 'startup' event.
ml_models = {}

# --- Lifespan Management (for model loading) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan. The 'startup' part loads the ML model,
    and the 'shutdown' part can be used for cleanup.
    """
    print("--- Server starting up ---")
    # In main.py, inside the lifespan function
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    ml_models["rag_pipeline"] = RAGPipeline(openai_api_key=OPENAI_API_KEY)
    print("--- RAG Pipeline Initialized ---")
    yield
    # Clean up the ML models and release the resources
    print("--- Server shutting down ---")
    ml_models.clear()

# --- Pydantic Models for API Data Validation ---
class SubmissionRequest(BaseModel):
    documents: str = Field(..., example="https://hackrx.blob.core.windows.net/assets/policy.pdf?...")
    questions: List[str]

class SubmissionResponse(BaseModel):
    answers: List[str]

# --- Security Dependency ---
security_scheme = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    """Dependency to verify the Bearer token."""
    if credentials.scheme != "Bearer" or credentials.credentials != SECURITY_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        )
    print("✅ Team token loaded successfully")

# --- API Router and Endpoints ---
api_router = APIRouter(prefix="/api/v1")

@api_router.post(
    "/hackrx/run",
    response_model=SubmissionResponse,
    summary="Run Submissions against a Document",
    dependencies=[Depends(verify_token)] # Protect the endpoint
)
async def run_submission(request_data: SubmissionRequest):
    """
    This endpoint processes a document and answers questions about it.
    1. It takes a document URL and a list of questions.
    2. It processes the document to build the internal knowledge base (if not already processed).
    3. It uses the RAG pipeline to generate answers.
    """
    pipeline: RAGPipeline = ml_models["rag_pipeline"]
    
    try:
        # For simplicity, we re-process the document on each call.
        # In a more advanced setup, you might cache results based on the URL.
        pipeline.process_document(request_data.documents)
        
        # Generate answers to the questions
        answers = pipeline.answer_questions(request_data.questions)
        
        return SubmissionResponse(answers=answers)
        
    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred while processing the request: {str(e)}"
        )

# --- Main FastAPI App ---
app = FastAPI(
    title="Document Q&A API with RAG",
    description="An API that uses a Retrieval-Augmented Generation pipeline to answer questions about a document.",
    version="1.0.0",
    lifespan=lifespan # Use the lifespan manager for model loading
)

app.include_router(api_router)

@app.get("/", summary="Health Check")
def read_root():
    return {"status": "API is running."}

@app.get("/api/v1")
def read_welcome():
    return {"Greet": "Welcome to Bajaj Finserv"}
