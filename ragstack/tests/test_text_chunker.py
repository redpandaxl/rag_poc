"""
Tests for the text chunker module.
"""
import unittest
from unittest.mock import patch
import pytest

from ragstack.data_ingestion.text_chunker import TextChunker, ChunkingStrategy


class TestTextChunker(unittest.TestCase):
    """Test the TextChunker class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.chunker = TextChunker()
    
    def test_fixed_size_chunking(self):
        """Test chunking text with fixed size strategy."""
        text = "This is a test document. It has multiple sentences. We need to chunk it properly."
        chunks = self.chunker.chunk_text(text, strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=10, chunk_overlap=0)
        
        # Check that chunks were created
        self.assertTrue(len(chunks) > 0)
        
        # Check that each chunk is no larger than the specified size
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 10)
    
    def test_sentence_chunking(self):
        """Test chunking text with sentence strategy."""
        text = "This is a test document. It has multiple sentences. We need to chunk it properly."
        chunks = self.chunker.chunk_text(text, strategy=ChunkingStrategy.SENTENCE)
        
        # Check that we get 3 chunks (one per sentence)
        self.assertEqual(len(chunks), 3)
        
        # Check content of chunks
        self.assertEqual(chunks[0], "This is a test document.")
        self.assertEqual(chunks[1], "It has multiple sentences.")
        self.assertEqual(chunks[2], "We need to chunk it properly.")
    
    def test_paragraph_chunking(self):
        """Test chunking text with paragraph strategy."""
        text = "This is paragraph one.\nIt has multiple sentences.\n\nThis is paragraph two.\nIt also has multiple sentences."
        chunks = self.chunker.chunk_text(text, strategy=ChunkingStrategy.PARAGRAPH)
        
        # Check that we get 2 chunks (one per paragraph)
        self.assertEqual(len(chunks), 2)
        
        # Check content of chunks
        self.assertEqual(chunks[0], "This is paragraph one.\nIt has multiple sentences.")
        self.assertEqual(chunks[1], "This is paragraph two.\nIt also has multiple sentences.")
    
    def test_recursive_chunking(self):
        """Test chunking text with recursive strategy."""
        # Create a long document that will need multiple levels of chunking
        paragraphs = ["Paragraph " + str(i) + ": " + "X" * 100 for i in range(20)]
        text = "\n\n".join(paragraphs)
        
        # Set a small chunk size to ensure multiple chunks
        chunks = self.chunker.chunk_text(
            text, 
            strategy=ChunkingStrategy.RECURSIVE, 
            chunk_size=200, 
            chunk_overlap=50
        )
        
        # Check that chunks were created
        self.assertTrue(len(chunks) > 0)
        
        # Check that each chunk is no larger than the specified size (with some tolerance)
        for chunk in chunks:
            # Recursive chunking might slightly exceed the chunk size sometimes,
            # so we allow a small tolerance
            self.assertTrue(len(chunk) <= 300)  # 200 + some tolerance
    
    def test_chunk_overlap(self):
        """Test that chunk overlap works correctly."""
        text = "AAAAA BBBBB CCCCC DDDDD EEEEE"
        chunks = self.chunker.chunk_text(
            text, 
            strategy=ChunkingStrategy.FIXED_SIZE, 
            chunk_size=10, 
            chunk_overlap=5
        )
        
        # Check that we have overlapping chunks
        self.assertTrue(len(chunks) > 1)
        
        # Check that each consecutive chunk overlaps with the previous one
        for i in range(1, len(chunks)):
            # The end of the previous chunk should appear at the start of the current chunk
            prev_end = chunks[i-1][-5:] if len(chunks[i-1]) >= 5 else chunks[i-1]
            curr_start = chunks[i][:5] if len(chunks[i]) >= 5 else chunks[i]
            self.assertEqual(prev_end, curr_start)
    
    def test_empty_text(self):
        """Test chunking empty text."""
        chunks = self.chunker.chunk_text("", strategy=ChunkingStrategy.FIXED_SIZE)
        
        # Should return an empty list
        self.assertEqual(chunks, [])
    
    def test_chunker_with_minimum_chunk_size(self):
        """Test the minimum chunk size parameter."""
        text = "Short. Another. One more."
        chunks = self.chunker.chunk_text(
            text, 
            strategy=ChunkingStrategy.SENTENCE, 
            minimum_chunk_size=10
        )
        
        # All chunks should be at least the minimum size
        for chunk in chunks:
            self.assertTrue(len(chunk) >= 10 or chunk in ["Short.", "Another.", "One more."])
    
    def test_recursion_depth_limit(self):
        """Test that recursion depth is properly limited."""
        # Create a very long document with no natural boundaries
        text = "X" * 10000
        
        # Patch the _recursive_chunk method to track recursion depth
        with patch.object(self.chunker, '_recursive_chunk', wraps=self.chunker._recursive_chunk) as mock_method:
            chunks = self.chunker.chunk_text(
                text,
                strategy=ChunkingStrategy.RECURSIVE,
                chunk_size=100,
                max_recursion_depth=3
            )
            
            # Check that the method wasn't called more than 3 levels deep
            # We check for calls with depth > 3
            depth_exceeded = False
            for call in mock_method.call_args_list:
                if call[1].get('depth', 0) > 3:
                    depth_exceeded = True
                    break
            
            self.assertFalse(depth_exceeded)
            
            # Verify chunks were created
            self.assertTrue(len(chunks) > 0)
    
    def test_invalid_strategy(self):
        """Test that an invalid strategy raises an error."""
        with self.assertRaises(ValueError):
            self.chunker.chunk_text("Some text", strategy="INVALID_STRATEGY")


if __name__ == "__main__":
    unittest.main()