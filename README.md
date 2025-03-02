# RAG-POC: Retrieval-Augmented Generation Proof of Concept

A self-hosted RAG environment with document processing, vector database integration, and a web interface.

## Features

- **Document Processing**: Automatic document ingestion with sophisticated chunking strategies
- **Vector Search**: Fast semantic search using either Qdrant or ChromaDB vector databases
- **LLM Integration**: Generate responses using OpenAI, Anthropic, or Ollama
- **Web Interface**: Streamlit UI for document search and management
- **Chat Interface**: ChatGPT-like conversational interface for document queries
- **API**: FastAPI backend for programmatic access and integration
- **Multi-format Support**: Process TXT, PDF, DOCX, XLSX, CSV, and more file formats
- **Monitoring**: Track system performance and document processing metrics

## Architecture

The system is organized into several core components:

1. **Data Ingestion**: Watches for new documents, chunks them, and processes them for storage
2. **Vector Database**: Stores document chunks with embeddings for semantic search
3. **Embedding Models**: Generates vector embeddings for documents and queries
4. **RAG Engine**: Combines document retrieval with LLM generation
5. **Web Interface**: Provides a user-friendly search interface and document management
6. **Chat Interface**: Provides a conversational interface to query documents with conversation context
7. **API**: Exposes the core functionality for integration with other systems

For detailed architecture diagrams, see the [mermaid](./mermaid) directory.

## Getting Started

### Prerequisites

- Python 3.12+
- Ollama with llama2 model (can run locally or on a remote server) or API keys for OpenAI/Anthropic
- Qdrant or ChromaDB vector database

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-poc.git
   cd rag-poc
   ```

2. Set up the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. Configure the environment:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

### Configuration

Edit the `.env` file to customize:
- Vector database connection
- Embedding model settings
- LLM integration (OpenAI, Anthropic, Ollama)
- Chunking parameters
- Web server configuration

### Running the System

Use the management script to start all components:

```bash
# Start everything (API, UI, ingestion)
./scripts/manage_system.sh start

# Start specific components
./scripts/manage_system.sh start api
./scripts/manage_system.sh start ui
./scripts/manage_system.sh start chat  # Start the chat interface
./scripts/manage_system.sh start ingest watch  # Start ingestion with file watching

# Stop all services
./scripts/manage_system.sh stop

# Check system status
./scripts/manage_system.sh status
```

Alternatively, you can use the simplified run script:

```bash
# Start everything
python run.py

# Start specific components
python run.py web  # Start the web interface
python run.py ingest  # Start the ingestion process
```

## Data Ingestion with Enhanced Logging

To process documents with detailed logging to diagnose issues:

```bash
# Process existing files with debug logging
python scripts/run_ingestion.py --debug

# Process a specific file with detailed logging
python scripts/run_ingestion.py --file /path/to/file.docx --debug

# Watch directory for new files
python scripts/run_ingestion.py --watch
```

All processed files are moved to `data/processed` when successfully processed, or `data/failed` if processing fails.

### Troubleshooting Ingestion Issues

If ingestion is stalling or failing:

1. Check the logs for detailed information:
   - `/logs/ragstack.log` - General application logs
   - `/logs/processing.log` - Document processing logs

2. Look for these common issues:
   - Connection failures to embedding model server
   - Vector database connectivity issues
   - File format parsing errors
   - NLTK resources missing (for text chunking)

3. Running with `--debug` provides detailed step-by-step logging of:
   - File detection and identification
   - Text extraction progress
   - Chunking operations
   - Embedding generation
   - Vector database operations

4. For binary file formats (DOCX, XLSX, PDF), ensure required packages are installed.

## Supported File Formats

RAGStack supports processing these document formats:

- **Text files**: .txt, .md, .json, .csv
- **Microsoft Word**: .docx, .doc
- **PDF documents**: .pdf
- **Spreadsheets**: .xlsx, .xls
- **Other formats**: Supported via the unstructured library

## Using the Chat Interface

The chat interface provides a ChatGPT-like experience for interacting with your documents. It maintains conversation history and provides context-aware responses based on your document collection.

### Starting the Chat Interface

```bash
# Using the management script
./scripts/manage_system.sh start chat

# Or using the script directly
python scripts/run_chat.py
```

The chat interface will be available at http://localhost:8502

### Features

- **Conversational Interface**: Maintains chat history and context
- **Vector Search Integration**: Retrieves relevant document snippets based on your queries
- **LLM Integration**: Can process the retrieved context with a language model for more coherent responses
- **Source References**: Shows which documents were used to generate responses
- **Collection Switching**: Change document collections during the conversation
- **Raw Context Viewing**: Option to view the raw context used to generate responses

### Settings

The sidebar provides several configuration options:

- **Collection Selection**: Choose which document collection to query
- **Number of Results**: Adjust how many documents to retrieve per query
- **LLM Toggle**: Enable/disable LLM processing (will just show raw context when disabled)
- **Show Raw Context**: Toggle whether to show the raw context snippets

## Using the API

The system exposes a REST API for programmatic access:

- `POST /search` - Search for documents
- `POST /upload` - Upload new documents
- `GET /collections` - List available collections
- `GET /stats` - Get statistics about collections

Example search request:
```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={"query": "What is RAG?", "top_k": 5}
)
results = response.json()["results"]
```

## Development

### Project Structure

```
ragstack/
   config/         # Configuration settings
   core/           # Core RAG functionality
   data_ingestion/ # Document processing
   models/         # Embedding and metadata models
   vector_db/      # Vector database clients
   web/            # Web interface and API
   utils/          # Utility functions
   tests/          # Test suite
```

### Adding New Features

- **New Vector Database**: Implement the `VectorDB` interface in `vector_db/base.py`
- **New Embedding Model**: Extend the `EmbeddingModel` class in `models/embeddings.py`
- **Custom Chunking Strategy**: Add methods to `TextChunker` in `data_ingestion/text_chunker.py`

## License

MIT License