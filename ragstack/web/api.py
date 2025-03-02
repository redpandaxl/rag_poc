"""
FastAPI backend for the RAG application.
"""
import os
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ragstack.config.settings import settings
from ragstack.core.rag import RAGEngine
from ragstack.data_ingestion.document_processor import DocumentProcessor
from ragstack.utils.logging import setup_logger


logger = setup_logger("ragstack.web.api")

# Initialize FastAPI app
app = FastAPI(
    title="RAGStack API",
    description="API for RAG document processing and retrieval",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (lazy loading)
rag_engine = None
document_processor = None


def get_rag_engine():
    """Get the RAG engine instance."""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine


def get_document_processor():
    """Get the document processor instance."""
    global document_processor
    if document_processor is None:
        document_processor = DocumentProcessor()
    return document_processor


# Define API models
class SearchQuery(BaseModel):
    """Search query model."""
    query: str
    top_k: Optional[int] = 5


class SearchResult(BaseModel):
    """Search result model."""
    results: List[Dict[str, Any]]


class UploadResponse(BaseModel):
    """Upload response model."""
    filename: str
    status: str
    message: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to RAGStack API",
        "endpoints": {
            "GET /": "This help message",
            "POST /search": "Search for documents based on a query",
            "POST /upload": "Upload a new document for processing",
            "GET /collections": "List all collections in the vector database",
            "GET /stats": "Get statistics about a collection",
            "GET /documents": "List processed documents",
            "GET /health": "Check API health status"
        }
    }


@app.post("/search", response_model=SearchResult)
async def search(query: SearchQuery):
    """
    Search for documents based on a query.
    
    Args:
        query: Search query with optional parameters
        
    Returns:
        Search results
    """
    try:
        engine = get_rag_engine()
        results = engine.search(query.query, query.top_k)
        return SearchResult(results=results)
    
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file for processing.
    
    Args:
        file: File to upload
        
    Returns:
        Upload status
    """
    try:
        # Validate file extension
        supported_extensions = [
            '.txt', '.md', '.json', '.csv', '.docx', '.doc', 
            '.pdf', '.xlsx', '.xls'
        ]
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in supported_extensions:
            logger.warning(f"Unsupported file extension: {file_ext}")
            # Try to process it anyway using unstructured, but warn the user
            message = f"File type {file_ext} may not be fully supported, but we'll try to process it"
        else:
            message = "File uploaded successfully and queued for processing"
        
        # Save file to raw data directory
        raw_path = os.path.join(settings.raw_data_dir, file.filename)
        
        # Write file content
        with open(raw_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"File saved to {raw_path}")
        
        # Process file (async to avoid blocking the response)
        # This will be picked up by the file watcher
        
        return UploadResponse(
            filename=file.filename,
            status="success",
            message=message
        )
    
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/collections")
async def list_collections():
    """List all collections in the vector database."""
    try:
        engine = get_rag_engine()
        collections = engine.vector_db.list_collections()
        return {"collections": collections}
    
    except Exception as e:
        logger.error(f"List collections error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


@app.get("/stats")
async def get_stats(collection: Optional[str] = None):
    """
    Get statistics about a collection.
    
    Args:
        collection: Collection name (optional)
        
    Returns:
        Collection statistics
    """
    try:
        engine = get_rag_engine()
        
        # Use provided collection or default
        collection_name = collection or engine.collection_name
        
        stats = engine.vector_db.get_collection_info(collection_name)
        return {"collection": collection_name, "stats": stats}
    
    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/documents")
async def list_documents():
    """
    List all processed documents.
    
    Returns:
        List of document paths with their processing status
    """
    try:
        # Get list of documents in raw, processed, and failed directories
        raw_docs = [f for f in os.listdir(settings.raw_data_dir) 
                   if os.path.isfile(os.path.join(settings.raw_data_dir, f))]
        
        processed_docs = [f for f in os.listdir(settings.processed_data_dir) 
                         if os.path.isfile(os.path.join(settings.processed_data_dir, f))]
        
        failed_docs = [f for f in os.listdir(settings.failed_data_dir) 
                      if os.path.isfile(os.path.join(settings.failed_data_dir, f))]
        
        return {
            "pending": raw_docs,
            "processed": processed_docs,
            "failed": failed_docs,
            "counts": {
                "pending": len(raw_docs),
                "processed": len(processed_docs),
                "failed": len(failed_docs),
                "total": len(raw_docs) + len(processed_docs) + len(failed_docs)
            }
        }
    
    except Exception as e:
        logger.error(f"List documents error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Check the health status of the API and its components.
    
    Returns:
        Health status of the API and its components
    """
    health_info = {
        "status": "ok",
        "components": {}
    }
    
    # Check vector database connection
    try:
        engine = get_rag_engine()
        collections = engine.vector_db.list_collections()
        health_info["components"]["vector_db"] = {
            "status": "ok",
            "type": engine.vector_db.__class__.__name__,
            "collections": collections
        }
    except Exception as e:
        health_info["status"] = "degraded"
        health_info["components"]["vector_db"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check document processor
    try:
        doc_processor = get_document_processor()
        health_info["components"]["document_processor"] = {
            "status": "ok"
        }
    except Exception as e:
        health_info["status"] = "degraded"
        health_info["components"]["document_processor"] = {
            "status": "error",
            "error": str(e)
        }
    
    return health_info


def start_api():
    """Start the FastAPI server with uvicorn."""
    import uvicorn
    uvicorn.run(
        "ragstack.web.api:app", 
        host=settings.api_host, 
        port=settings.api_port,
        reload=settings.debug
    )