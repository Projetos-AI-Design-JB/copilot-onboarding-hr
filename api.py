
import os
import time
import logging
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

from rag_engine import ask_copilot, setup_environment

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
API_TOKEN = os.environ.get("API_TOKEN", "default-secret-token")
API_KEY_NAME = "X-API-Key"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise Copilot HR Onboarding API")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    """
    Dependency to validate the API key from the request header.
    """
    if not api_key_header or api_key_header != API_TOKEN:
        logger.warning(f"Unauthorized access attempt. Provided token: {'[REDACTED]' if api_key_header else 'None'}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key_header

class ChatRequest(BaseModel):
    """
    Pydantic model for input validation. Enforces that the question is provided and not empty.
    """
    question: str = Field(..., min_length=1, description="The user's question. Cannot be empty.")

class ChatResponse(BaseModel):
    """
    Pydantic model for the response.
    """
    answer: str

@app.on_event("startup")
async def startup_event():
    """
    Initializes the RAG engine components (Vertex AI, Pinecone) when the API starts.
    """
    logger.info("Starting up API server and initializing RAG engine...")
    try:
        setup_environment()
        logger.info("RAG engine initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize RAG engine during startup: {e}")
        # Depending on the deployment environment, you might want to raise an exception 
        # here to prevent the pod from becoming healthy if initialization fails.

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log the start, completion status, and duration of every request.
    """
    start_time = time.time()
    logger.info(f"Received request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Completed request: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
    
    return response

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(get_api_key)])
async def chat_endpoint(request: ChatRequest):
    """
    Main endpoint to interact with the RAG Copilot.
    Requires a valid API key in the header.
    """
    logger.info(f"Processing chat request. Question: '{request.question}'")
    
    # Additional validation to ensure the string isn't just whitespaces
    if not request.question.strip():
        logger.warning("Received whitespace-only question. Rejecting request.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty or just whitespace.")

    try:
        # Call the core RAG engine logic
        answer = ask_copilot(request.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error during RAG processing: {e}", exc_info=True)
        # Return a generic 500 error to the client, masking internal details
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred while processing your request.")

if __name__ == "__main__":
    # For local development and testing
    uvicorn.run(app, host="0.0.0.0", port=8080)
