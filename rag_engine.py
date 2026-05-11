import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel, GenerationConfig
from pinecone import Pinecone, Index

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Google Cloud Project and Location for Vertex AI
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "projeto-jb-api-gcp") # e.g., "my-gcp-project"
GCP_LOCATION = os.environ.get("GCP_LOCATION", "southamerica-east1-a")

# Vertex AI Models
VERTEX_EMBEDDING_MODEL_NAME = "text-embedding-005"
VERTEX_GENERATIVE_MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_DIMENSION = 768  # Dimension for text-embedding-005

# Pinecone Configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_5SL7KB_PZQHtiRACbdzymyRAcXWenp41PuFpYdFDNCiAipJrwo3RSn13VzFJn7DtQxVVVR")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rh-copilot-index")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Instances (initialized once) ---
embedding_model: Optional[TextEmbeddingModel] = None
generative_model: Optional[GenerativeModel] = None
pinecone_index: Optional[Index] = None

# --- Utility Functions ---
def setup_environment():
    """
    Ensures all necessary environment variables are set and initializes global clients.
    """
    global embedding_model, generative_model, pinecone_index

    if not GCP_PROJECT_ID or GCP_PROJECT_ID == "your-gcp-project-id":
        logger.error("GCP_PROJECT_ID environment variable not set or is default. Please set it.")
        exit(1)
    if not PINECONE_API_KEY or PINECONE_API_KEY == "your-pinecone-api-key":
        logger.error("PINECONE_API_KEY environment variable not set or is default. Please set it.")
        exit(1)
    
    # Initialize Vertex AI SDK
    try:
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        logger.info(f"Vertex AI initialized for project '{GCP_PROJECT_ID}' in location '{GCP_LOCATION}'.")
        
        embedding_model = TextEmbeddingModel.from_pretrained(VERTEX_EMBEDDING_MODEL_NAME)
        logger.info(f"Vertex AI Embedding Model '{VERTEX_EMBEDDING_MODEL_NAME}' loaded.")
        
        generative_model = GenerativeModel(VERTEX_GENERATIVE_MODEL_NAME)
        logger.info(f"Vertex AI Generative Model '{VERTEX_GENERATIVE_MODEL_NAME}' loaded.")
        
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI clients: {e}")
        exit(1)

    # Initialize Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            logger.error(f"Pinecone index '{PINECONE_INDEX_NAME}' not found. Please ensure it's created via the ingestion pipeline.")
            exit(1)
        
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        logger.info(f"Pinecone initialized and connected to index '{PINECONE_INDEX_NAME}'.")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone or connect to index: {e}")
        exit(1)

def get_embedding_for_query_with_retry(text: str, model: TextEmbeddingModel, max_retries: int = 3, initial_backoff: float = 1.0) -> List[float]:
    """
    Generates an embedding for a single text query with retry logic.
    """
    retries = 0
    backoff_time = initial_backoff
    while retries < max_retries:
        try:
            logger.info(f"Attempt {retries + 1}/{max_retries} to get embedding for query: '{text[:50]}...'")
            inputs = [TextEmbeddingInput(text, task_type="RETRIEVAL_QUERY")]
            embeddings = model.get_embeddings(inputs)
            logger.info(f"Successfully generated embedding for query.")
            return embeddings[0].values
        except Exception as e:
            logger.warning(f"Error generating embedding: {e}. Retrying in {backoff_time:.2f} seconds...")
            retries += 1
            time.sleep(backoff_time)
            backoff_time *= 2
    logger.error(f"Failed to generate embedding after {max_retries} retries for query: '{text[:50]}...'")
    raise Exception(f"Failed to get embedding after {max_retries} retries.")

def retrieve_context_from_pinecone_with_retry(query_embedding: List[float], index: Index, top_k: int = 5, max_retries: int = 3, initial_backoff: float = 1.0) -> List[str]:
    """
    Retrieves relevant text contexts from Pinecone based on a query embedding with retry logic.
    """
    retries = 0
    backoff_time = initial_backoff
    while retries < max_retries:
        try:
            logger.info(f"Attempt {retries + 1}/{max_retries} to retrieve context from Pinecone (top_k={top_k}).")
            query_results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            contexts = [match["metadata"]["text"] for match in query_results["matches"] if "text" in match["metadata"]]
            logger.info(f"Successfully retrieved {len(contexts)} contexts from Pinecone.")
            return contexts
        except Exception as e:
            logger.warning(f"Error retrieving context from Pinecone: {e}. Retrying in {backoff_time:.2f} seconds...")
            retries += 1
            time.sleep(backoff_time)
            backoff_time *= 2
    logger.error(f"Failed to retrieve context from Pinecone after {max_retries} retries.")
    raise Exception(f"Failed to retrieve context from Pinecone after {max_retries} retries.")

def generate_response_with_retry(prompt: str, model: GenerativeModel, temperature: float = 0.1, max_retries: int = 3, initial_backoff: float = 1.0) -> str:
    """
    Generates a response using the Vertex AI generative model with retry logic and specific generation config.
    """
    retries = 0
    backoff_time = initial_backoff
    generation_config = GenerationConfig(temperature=temperature)

    while retries < max_retries:
        try:
            logger.info(f"Attempt {retries + 1}/{max_retries} to generate response with temperature={temperature}.")
            response = model.generate_content(prompt, generation_config=generation_config)
            generated_text = response.text
            logger.info(f"Successfully generated response. Length: {len(generated_text)} characters.")
            return generated_text
        except Exception as e:
            logger.warning(f"Error generating content: {e}. Retrying in {backoff_time:.2f} seconds...")
            retries += 1
            time.sleep(backoff_time)
            backoff_time *= 2
    logger.error(f"Failed to generate response after {max_retries} retries for prompt: '{prompt[:100]}...'")
    raise Exception(f"Failed to generate response after {max_retries} retries.")

def ask_copilot(question: str) -> str:
    """
    Main function for the RAG engine.
    Processes a user question, retrieves context, and generates an answer.
    """
    logger.info(f"Received question: '{question}'")

    if embedding_model is None or generative_model is None or pinecone_index is None:
        logger.error("RAG engine components not initialized. Please call setup_environment() first.")
        raise RuntimeError("RAG engine not properly initialized.")

    try:
        # 1. Vectorize the User Question
        query_embedding = get_embedding_for_query_with_retry(question, embedding_model)
        logger.info("Question vectorized.")

        # 2. Retrieve Context (Retrieval)
        retrieved_contexts = retrieve_context_from_pinecone_with_retry(query_embedding, pinecone_index)
        
        if not retrieved_contexts:
            logger.warning("No relevant context found in Pinecone. Responding with a default message.")
            return "I apologize, but I could not find relevant information to answer your question in my knowledge base. Please try rephrasing your question or refer to official documentation."

        # 3. Prompt Engineering and Generation
        context_str = "\\n".join([f"- {ctx}" for ctx in retrieved_contexts])
        
        # Role Prompting for strict adherence to context
        system_instruction = (
            "You are an Enterprise Copilot for HR onboarding. Your task is to answer user questions "
            "EXCLUSIVELY based on the provided context. Do NOT use any external knowledge. "
            "If the answer is not found in the context, state that you cannot answer from the provided information. "
            "Maintain a professional and helpful tone."
        )

        full_prompt = f"{system_instruction}\\n\\nContext:\\n{context_str}\\n\\nUser Question: {question}\\n\\nAnswer:"
        
        logger.info(f"Generated prompt (first 200 chars): '{full_prompt[:200]}...'")

        # 4. Generate Answer with Vertex AI (temperature 0.1)
        response_text = generate_response_with_retry(full_prompt, generative_model, temperature=0.1)
        logger.info(f"Response generated for question: '{question}'")
        
        return response_text

    except Exception as e:
        logger.critical(f"An unhandled error occurred in ask_copilot: {e}", exc_info=True)
        return "An internal error occurred while processing your request. Please try again later."

if __name__ == "__main__":
    setup_environment() # Initialize clients once at startup

    print("\\nEnter 'quit' or 'exit' to end the session.")
    while True:
        user_question = input("\\nAsk the Copilot: ")
        if user_question.lower() in ["quit", "exit"]:
            break
        
        try:
            answer = ask_copilot(user_question)
            print(f"\\nCopilot: {answer}")
        except RuntimeError as e:
            print(f"Error: {e}. Please ensure the environment is correctly set up.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
