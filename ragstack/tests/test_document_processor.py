"""
Tests for the document processor module.
"""
import os
import unittest
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path
import shutil

import pytest

from ragstack.data_ingestion.document_processor import DocumentProcessor
from ragstack.data_ingestion.text_chunker import TextChunker
from ragstack.models.embeddings import EmbeddingModel
from ragstack.vector_db.base import VectorDB as VectorDatabase


class TestDocumentProcessor(unittest.TestCase):
    """Test the DocumentProcessor class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directories for test files
        self.test_dir = Path(tempfile.mkdtemp())
        self.raw_dir = self.test_dir / "raw"
        self.processed_dir = self.test_dir / "processed"
        self.failed_dir = self.test_dir / "failed"
        
        # Create the directories
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.failed_dir.mkdir(exist_ok=True)
        
        # Create mock dependencies
        self.chunker = MagicMock(spec=TextChunker)
        self.embedding_model = MagicMock(spec=EmbeddingModel)
        self.vector_db = MagicMock(spec=VectorDatabase)
        
        # Create sample test files
        self.create_test_files()
        
        # Create the document processor
        self.processor = DocumentProcessor(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            failed_dir=self.failed_dir,
            chunker=self.chunker,
            embedding_model=self.embedding_model,
            vector_db=self.vector_db
        )
    
    def tearDown(self):
        """Clean up after the tests."""
        # Remove temporary directory and all its contents
        shutil.rmtree(self.test_dir)
    
    def create_test_files(self):
        """Create test files for document processing."""
        # Create a text file
        text_file = self.raw_dir / "test.txt"
        with open(text_file, "w") as f:
            f.write("This is a test document for processing.")
        
        # Create an empty file to test error handling
        empty_file = self.raw_dir / "empty.txt"
        with open(empty_file, "w") as f:
            f.write("")
    
    def test_process_text_file(self):
        """Test processing a text file."""
        # Set up mocks
        self.chunker.chunk_text.return_value = ["Chunk 1", "Chunk 2"]
        self.embedding_model.get_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        # Process the file
        file_path = self.raw_dir / "test.txt"
        result = self.processor.process_file(file_path)
        
        # Verify the file was processed correctly
        self.assertTrue(result)
        self.chunker.chunk_text.assert_called_once()
        self.embedding_model.get_embeddings.assert_called_once()
        self.vector_db.add_texts.assert_called_once()
        
        # Verify the file was moved to the processed directory
        self.assertFalse(file_path.exists())
        self.assertTrue((self.processed_dir / "test.txt").exists())
    
    def test_process_empty_file(self):
        """Test processing an empty file."""
        # Process the file
        file_path = self.raw_dir / "empty.txt"
        result = self.processor.process_file(file_path)
        
        # Verify the file failed processing
        self.assertFalse(result)
        
        # Verify the file was moved to the failed directory
        self.assertFalse(file_path.exists())
        self.assertTrue((self.failed_dir / "empty.txt").exists())
    
    def test_process_nonexistent_file(self):
        """Test processing a file that doesn't exist."""
        # Process a non-existent file
        file_path = self.raw_dir / "nonexistent.txt"
        result = self.processor.process_file(file_path)
        
        # Verify the processing failed
        self.assertFalse(result)
    
    def test_chunker_error_handling(self):
        """Test handling errors from the chunker."""
        # Set up the chunker to raise an exception
        self.chunker.chunk_text.side_effect = Exception("Chunking error")
        
        # Process the file
        file_path = self.raw_dir / "test.txt"
        result = self.processor.process_file(file_path)
        
        # Verify the processing failed
        self.assertFalse(result)
        
        # Verify the file was moved to the failed directory
        self.assertFalse(file_path.exists())
        self.assertTrue((self.failed_dir / "test.txt").exists())
    
    def test_embedding_error_handling(self):
        """Test handling errors from the embedding model."""
        # Set up mocks
        self.chunker.chunk_text.return_value = ["Chunk 1", "Chunk 2"]
        self.embedding_model.get_embeddings.side_effect = Exception("Embedding error")
        
        # Process the file
        file_path = self.raw_dir / "test.txt"
        result = self.processor.process_file(file_path)
        
        # Verify the processing failed
        self.assertFalse(result)
        
        # Verify the file was moved to the failed directory
        self.assertFalse(file_path.exists())
        self.assertTrue((self.failed_dir / "test.txt").exists())
    
    def test_vectordb_error_handling(self):
        """Test handling errors from the vector database."""
        # Set up mocks
        self.chunker.chunk_text.return_value = ["Chunk 1", "Chunk 2"]
        self.embedding_model.get_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
        self.vector_db.add_texts.side_effect = Exception("Vector DB error")
        
        # Process the file
        file_path = self.raw_dir / "test.txt"
        result = self.processor.process_file(file_path)
        
        # Verify the processing failed
        self.assertFalse(result)
        
        # Verify the file was moved to the failed directory
        self.assertFalse(file_path.exists())
        self.assertTrue((self.failed_dir / "test.txt").exists())


if __name__ == "__main__":
    unittest.main()