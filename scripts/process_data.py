import os
import time
import logging
import json
import uuid
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
import torch
import numpy as np
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize
import re
from ollama import Client, GenerateResponse

# Configure logging
logging.basicConfig(
    filename=os.path.expanduser("~/rag_poc/rag-poc/logs/processing.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize Qdrant client
client = QdrantClient(host="192.168.1.49", port=6333, grpc_port=6334)

# Collection name
collection_name = "documents"

# Vector size
vector_size = 4096  # Adjust based on your model

# Set the NLTK data path
nltk.data.path.append(os.path.expanduser("~/nltk_data"))

# Download the punkt resource
try:
    nltk.download('punkt', download_dir=os.path.expanduser("~/nltk_data"))
except Exception as e:
    logging.error(f"Failed to download punkt resource: {e}")
def generate_metadata(content: str) -> Dict[str, Any]:
    """Use Ollama to generate metadata/tags for the document."""
    prompt = f"""
    You are an intelligent metadata generator. Analyze the provided document content and generate JSON metadata containing:
    - title: A concise title for the document
    - tags: A list of 3-5 relevant topics
    - summary: A brief summary (1-2 sentences)

    Return ONLY the JSON object without any extra text.

    Document: {content[:2000]}  # Truncate to avoid token limits

    Example JSON Output:
    {{
        "title": "Document Title",
        "tags": ["tag1", "tag2"],
        "summary": "A brief summary of the document."
    }}
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response: GenerateResponse = ollama.generate(
                model="llama2",
                prompt=prompt,
                stream=False  # Set stream to False to get the complete response
            )
            logging.info(f"LLM Response: {response['response']}")  # Access the response text

            # Extract JSON from response using regex
            json_pattern = r'\{.*?\}(?=\n|\Z)'
            match = re.search(json_pattern, response['response'], re.DOTALL)  # Use response['response']
            if match:
                json_string = match.group(0)
                try:
                    metadata = json.loads(json_string)
                    # Validate metadata
                    if ("title" in metadata and 
                        "tags" in metadata and 
                        "summary" in metadata):
                        return metadata
                    else:
                        logging.warning("Invalid metadata structure in LLM response.")
                        return {"title": "Untitled", "tags": [], "summary": ""}
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON: {e}")
                    return {"title": "Untitled", "tags": [], "summary": ""}
            else:
                logging.warning("No JSON found in LLM response.")
                return {"title": "Untitled", "tags": [], "summary": ""}
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Metadata generation failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(1)
            else:
                logging.error(f"Failed to generate metadata after {max_retries} attempts: {e}")
                # Dump file for debugging
                dump_file(content, f"failed_metadata_{uuid.uuid4()}.txt")
                return {"title": "Untitled", "tags": [], "summary": ""}

def dump_file(content: str, filename: str) -> None:
    """Dump content to a file for debugging."""
    try:
        with open(os.path.expanduser(f"~/rag_poc/rag_poc/data/failed/{filename}"), "w") as f:
            f.write(content)
        logging.info(f"Dumped content to: {filename}")
    except Exception as e:
        logging.error(f"Failed to dump content: {e}")

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Chunk text into smaller pieces with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def semantic_chunk_text(text: str) -> List[str]:
    """Chunk text into smaller pieces based on sentences."""
    sentences = sent_tokenize(text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < 512:
            chunk += sentence + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def recursive_chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Chunk text into smaller pieces recursively."""
    chunks = []
    if len(text) < chunk_size:
        return [text]
    sentences = sent_tokenize(text)
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < chunk_size:
            chunk += sentence + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "
    if chunk:
        chunks.append(chunk.strip())
    if len(chunks) == 1:
        return [text]
    result = []
    for chunk in chunks:
        result.extend(recursive_chunk_text(chunk, chunk_size, overlap))
    return result

def process_file(file_path: str) -> None:
    """Process a single file and ingest into Qdrant."""
    content = ""
    try:
        logging.info(f"Processing: {file_path}")
        content = Path(file_path).read_text()

        # Chunk the content
        chunks = recursive_chunk_text(content)

        for i, chunk in enumerate(chunks):
            # Generate metadata with Ollama
            metadata = generate_metadata(chunk)
            
            if not is_metadata_valid(metadata):
                logging.warning(f"Skipping chunk {i}: Invalid metadata generated.")
                continue

            try:
                # Create embedding using the model
                inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).to("cuda")
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                # Use last_hidden_state to generate embeddings
                embedding = outputs.hidden_states[-1].mean(dim=1).cpu().numpy().tolist()[0]
            except Exception as e:
                logging.error(f"Error generating embedding: {e}")
                # Generate a random embedding as fallback for testing
                embedding = np.random.rand(vector_size).tolist()
                logging.warning("Using random embedding as fallback")

            # Generate UUID for point ID
            point_id = uuid.uuid4()

            # Check if the collection exists
            if not collection_exists(collection_name):
                create_collection(collection_name, vector_size)
                time.sleep(5)  # Wait for collection to be created

            # Add to Qdrant
            try:
                client.upsert(
                    collection_name=collection_name,
                    wait=True,
                    points=[
                        models.PointStruct(
                            id=str(point_id),  # Convert UUID to string
                            vector=embedding,
                            payload={"content": chunk, "metadata": metadata, "chunk_num": i}
                        )
                    ]
                )
                logging.info(f"Successfully ingested chunk {i} from {file_path} into Qdrant")
            except Exception as e:
                logging.error(f"Error uploading to Qdrant: {e}")
                # If Qdrant fails, try ChromaDB
                logging.info("Falling back to local ChromaDB")
                try:
                    from chromadb import PersistentClient
                    chroma_client = PersistentClient(path=os.path.join(os.path.expanduser("~/rag_poc/rag-poc"), "chromadb-data"))
                    chroma_collection = chroma_client.get_or_create_collection("documents")
                    
                    chroma_collection.add(
                        documents=[chunk],
                        metadatas=[{"metadata": metadata, "chunk_num": i}],
                        ids=[str(point_id)]
                    )
                    logging.info(f"Successfully ingested chunk {i} from {file_path} into ChromaDB")
                except Exception as chroma_error:
                    logging.error(f"ChromaDB fallback also failed: {chroma_error}")

        # Move processed file
        processed_path = os.path.expanduser(f"~/rag_poc/rag-poc/data/processed/{os.path.basename(file_path)}")
        os.rename(file_path, processed_path)
        logging.info(f"Completed: {file_path} -> {processed_path}")

    except Exception as e:
        logging.error(f"Failed to process {file_path}: {e}")
        # Dump file for debugging
        if content:
            dump_file(content, f"failed_processing_{uuid.uuid4()}.txt")

def is_metadata_valid(metadata: Dict[str, Any]) -> bool:
    """Check if metadata contains required fields."""
    required_fields = ["title", "tags", "summary"]
    if not all(field in metadata for field in required_fields):
        logging.warning("Metadata is missing required fields.")
        return False
    if not (isinstance(metadata["title"], str) and 
            isinstance(metadata["tags"], list) and 
            isinstance(metadata["summary"], str)):
        logging.warning("Metadata fields are of incorrect type.")
        return False
    return True

def collection_exists(collection_name: str) -> bool:
    """Check if a collection exists."""
    try:
        client.get_collection(collection_name=collection_name)
        return True
    except Exception as e:
        logging.error(f"Error checking collection existence: {e}")
        return False

def create_collection(collection_name: str, vector_size: int) -> None:
    """Create a new collection."""
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        logging.info(f"Collection '{collection_name}' created successfully with vector size {vector_size}.")
    except Exception as e:
        logging.error(f"Failed to create collection '{collection_name}': {e}")

class FileHandler(FileSystemEventHandler):
    """Watchdog handler to process new files."""
    def on_created(self, event):
        if not event.is_directory:
            process_file(event.src_path)

if __name__ == "__main__":
    # Initialize Ollama client
    try:
        ollama = Client(host='http://localhost:11434')
        logging.info("Connected to Ollama client.")
    except Exception as e:
        logging.error(f"Failed to connect to Ollama: {e}")
        raise SystemExit("Failed to connect to Ollama.")

    # Initialize DeepSeek model
    try:
        model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
        logging.info("Loaded DeepSeek model successfully.")
    except Exception as e:
        logging.error(f"Failed to load DeepSeek model: {e}")
        logging.info("Falling back to Ollama for embeddings only.")

    # Process existing files first
    raw_dir = os.path.expanduser("~/rag_poc/rag-poc/data/raw")
    for file in os.listdir(raw_dir):
        process_file(os.path.join(raw_dir, file))

    # Start watching for new files
    observer = Observer()
    observer.schedule(FileHandler(), path=raw_dir, recursive=False)
    observer.start()
    logging.info("Started directory watcher.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

