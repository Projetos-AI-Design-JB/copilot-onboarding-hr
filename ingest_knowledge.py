import os
import json
import logging
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from pinecone import Pinecone, ServerlessSpec, PodSpec, AwsRegion

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Google Cloud Project and Location for Vertex AI
# It's highly recommended to set these as environment variables in production.
# For local development, you can set default values here.
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "projeto-jb-api-gcp") # e.g., "my-gcp-project"
GCP_LOCATION = os.environ.get("GCP_LOCATION", "southamerica-east1-a")

VERTEX_EMBEDDING_MODEL_NAME = "text-embedding-005"
EMBEDDING_DIMENSION = 768  # Dimension for text-embedding-005

# Pinecone Configuration
# It's highly recommended to set these as environment variables in production.
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_5SL7KB_PZQHtiRACbdzymyRAcXWenp41PuFpYdFDNCiAipJrwo3RSn13VzFJn7DtQxVVVR")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rh-copilot-index")

# Dataset Path
DATASET_PATH = "dataset/QA.json"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Utility Functions ---
def setup_environment():
    """Ensures all necessary environment variables are set."""
    if not GCP_PROJECT_ID or GCP_PROJECT_ID == "your-gcp-project-id":
        logger.error("GCP_PROJECT_ID environment variable not set or is default. Please set it.")
        exit(1)
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY environment variable not set. Please set it.")
        exit(1)
    
    # Initialize Vertex AI SDK
    try:
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        logger.info(f"Vertex AI initialized for project '{GCP_PROJECT_ID}' in location '{GCP_LOCATION}'.")
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI: {e}")
        exit(1)

def load_data(file_path: str) -> List[Dict[str, str]]:
    """
    Loads text data from a JSON file.
    The JSON file is expected to be a list of dictionaries, each with a 'text' key.
    """
    logger.info(f"Starting data extraction from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Dataset file not found at {file_path}")
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Successfully extracted {len(data)} items from {file_path}")
    return data

def clean_text(text: str) -> str:
    """
    Cleans the raw text by removing specific instruction tags.
    """
    text = text.replace("<s>[INST]", "").replace("[/INST]", "").replace("</s>", "").strip()
    return text

def get_embeddings_with_retry(texts: List[str], model: TextEmbeddingModel, max_retries: int = 5, initial_backoff: float = 1.0) -> List[List[float]]:
    """
    Generates embeddings for a list of texts with retry logic for API calls.
    """
    all_embeddings = []
    for i in range(0, len(texts), 5): # Process in smaller batches for robustness
        batch_texts = texts[i:i+5]
        
        # Prepare inputs for the embedding model, specifying task type
        inputs = [TextEmbeddingInput(text, task_type="RETRIEVAL_DOCUMENT") for text in batch_texts]

        retries = 0
        backoff_time = initial_backoff
        while retries < max_retries:
            try:
                logger.info(f"Attempt {retries + 1}/{max_retries} to get embeddings for batch starting with: '{batch_texts[0][:50]}...'")
                # The model expects a list of TextEmbeddingInput objects
                embeddings = model.get_embeddings(inputs)
                all_embeddings.extend([embedding.values for embedding in embeddings])
                logger.info(f"Successfully generated embeddings for a batch of {len(batch_texts)} texts.")
                break # Exit retry loop on success
            except Exception as e:
                logger.warning(f"Error generating embeddings: {e}. Retrying in {backoff_time:.2f} seconds...")
                retries += 1
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
        else:
            logger.error(f"Failed to generate embeddings after {max_retries} retries for batch starting with: '{batch_texts[0][:50]}...'")
            raise Exception(f"Failed to get embeddings after {max_retries} retries.")
    return all_embeddings


def initialize_pinecone(api_key: str, index_name: str, dimension: int):
    """
    Initializes Pinecone and creates the index if it does not exist.
    """
    logger.info("Initializing Pinecone...")
    try:
        pc = Pinecone(api_key=api_key)
        
        if index_name not in pc.list_indexes().names():
            logger.info(f"Pinecone index '{index_name}' not found. Creating a new one.")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',  # Common metric for embeddings
                spec=ServerlessSpec(
                    cloud=CloudProvider.AWS, # !!! IMPORTANT: Adjust if your Pinecone instance is on a different cloud provider (e.g., GCP, Azure)
                    region=AwsRegion.US_EAST_1 # !!! IMPORTANT: Adjust to your Pinecone instance's region
                )
            )
            logger.info(f"Pinecone index '{index_name}' created successfully.")
        else:
            logger.info(f"Pinecone index '{index_name}' already exists.")
        
        index = pc.Index(index_name)
        logger.info("Pinecone initialized and index connected.")
        return index
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone or connect to index: {e}")
        raise

def upsert_vectors_with_retry(index, vectors: List[Dict[str, Any]], max_retries: int = 5, initial_backoff: float = 1.0):
    """
    Upserts vectors to Pinecone with retry logic.
    """
    retries = 0
    backoff_time = initial_backoff
    while retries < max_retries:
        try:
            logger.info(f"Attempt {retries + 1}/{max_retries} to upsert {len(vectors)} vectors to Pinecone.")
            index.upsert(vectors=vectors)
            logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone.")
            break
        except Exception as e:
            logger.warning(f"Error upserting vectors to Pinecone: {e}. Retrying in {backoff_time:.2f} seconds...")
            retries += 1
            time.sleep(backoff_time)
            backoff_time *= 2  # Exponential backoff
    else:
        logger.error(f"Failed to upsert vectors after {max_retries} retries.")
        raise Exception(f"Failed to upsert vectors after {max_retries} retries.")

def main():
    logger.info("Starting data ingestion pipeline.")
    
    setup_environment()

    try:
        # 1. Extract data
        raw_data = load_data(DATASET_PATH)
        texts_to_embed = [clean_text(item["text"]) for item in raw_data]

        # 2. Vectorize data using Vertex AI
        embedding_model = TextEmbeddingModel.from_pretrained(VERTEX_EMBEDDING_MODEL_NAME)
        embeddings = get_embeddings_with_retry(texts_to_embed, embedding_model)

        # Ensure unique IDs for Pinecone upsert
        pinecone_vectors = []
        for i, (text, embedding) in enumerate(zip(texts_to_embed, embeddings)):
            pinecone_vectors.append({
                "id": f"doc-{i}",
                "values": embedding,
                "metadata": {"text": text} # Store original text as metadata
            })

        # 3. Store vectors in Pinecone
        pinecone_index = initialize_pinecone(PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION)
        upsert_vectors_with_retry(pinecone_index, pinecone_vectors)

        logger.info("Data ingestion pipeline completed successfully.")

    except FileNotFoundError as e:
        logger.error(f"A file error occurred: {e}")
    except Exception as e:
        logger.critical(f"An unhandled error occurred during pipeline execution: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
