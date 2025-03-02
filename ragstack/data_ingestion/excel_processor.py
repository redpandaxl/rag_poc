"""
Specialized Excel document processor for product management templates.
Handles multi-sheet workbooks, tables, and structured data formats.
"""
import os
import pandas as pd
import uuid
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ragstack.utils.logging import setup_logger
from ragstack.data_ingestion.text_chunker import TextChunker, ChunkingStrategy
from ragstack.utils.monitoring.metrics import metrics_collector, ProcessingStage

# Initialize logger
logger = setup_logger("ragstack.data_ingestion.excel_processor")

class ExcelProcessor:
    """
    Specialized processor for Excel files, particularly those used in
    product management like templates, tables, and structured data.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        include_metadata: bool = True,
        include_headers: bool = True,
        include_sheet_names: bool = True,
        max_rows_per_chunk: int = 100,
        strategy: str = "table",
    ):
        """
        Initialize the Excel processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
            include_metadata: Whether to include file metadata
            include_headers: Whether to include column headers with each chunk
            include_sheet_names: Whether to include sheet names with chunks
            max_rows_per_chunk: Maximum number of rows per chunk for large tables
            strategy: Chunking strategy - 'table', 'sheet', or 'generic'
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_metadata = include_metadata
        self.include_headers = include_headers
        self.include_sheet_names = include_sheet_names
        self.max_rows_per_chunk = max_rows_per_chunk
        self.strategy = strategy
        self.text_chunker = TextChunker()
        
        logger.info(f"Excel Processor initialized with strategy {strategy}")
    
    def process_excel_file(self, file_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process an Excel file and return chunks of text with metadata.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Tuple of (text chunks, chunk metadata)
        """
        logger.info(f"Processing Excel file: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Excel file not found: {file_path}")
            
        # Generate document ID for tracking
        document_id = str(uuid.uuid4())
        document_name = file_path.name
        
        # Start metrics tracking if available
        try:
            metrics_collector.start_document_processing(document_id, document_name, "xlsx")
            has_metrics = True
        except (ImportError, AttributeError):
            has_metrics = False
        
        try:
            start_time = __import__('time').time()
            
            # Read the Excel file
            logger.info(f"Reading Excel file: {file_path}")
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            logger.info(f"Excel file has {len(sheet_names)} sheets: {', '.join(sheet_names)}")
            
            extraction_time = __import__('time').time() - start_time
            if has_metrics:
                metrics_collector.record_stage_timing(
                    document_id, 
                    ProcessingStage.EXTRACTION, 
                    extraction_time,
                    metadata={"sheet_count": len(sheet_names)}
                )
            
            # Process based on strategy
            if self.strategy == "table":
                chunks, chunk_metadata = self._process_as_tables(excel_file, file_path, document_id)
            elif self.strategy == "sheet":
                chunks, chunk_metadata = self._process_as_sheets(excel_file, file_path, document_id)
            else:  # generic strategy
                chunks, chunk_metadata = self._process_generic(excel_file, file_path, document_id)
                
            logger.info(f"Generated {len(chunks)} chunks from Excel file")
            
            # Record completion
            if has_metrics:
                metrics_collector.complete_document_processing(document_id, True, len(chunks))
                
            return chunks, chunk_metadata
                
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {str(e)}", exc_info=True)
            if has_metrics:
                metrics_collector.complete_document_processing(document_id, False, 0, str(e))
            raise
    
    def _process_as_tables(
        self, excel_file: pd.ExcelFile, file_path: Path, document_id: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process Excel file treating each worksheet as a collection of tables.
        Good for product management templates with defined tables.
        
        Args:
            excel_file: The pandas ExcelFile object
            file_path: Path to the Excel file
            document_id: Unique ID for the document
            
        Returns:
            Tuple of (text chunks, chunk metadata)
        """
        start_time = __import__('time').time()
        chunks = []
        chunk_metadata = []
        
        for sheet_name in excel_file.sheet_names:
            logger.info(f"Processing sheet: {sheet_name}")
            
            try:
                # Read the sheet
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Skip empty sheets
                if df.empty:
                    logger.info(f"Sheet {sheet_name} is empty, skipping")
                    continue
                    
                # Look for tables within the sheet by identifying header rows
                # This heuristic searches for rows with string headers followed by data
                potential_tables = self._identify_tables_in_dataframe(df)
                
                if potential_tables:
                    logger.info(f"Found {len(potential_tables)} potential tables in sheet {sheet_name}")
                    
                    # Process each identified table
                    for table_idx, (start_row, end_row) in enumerate(potential_tables):
                        table_df = df.iloc[start_row:end_row]
                        
                        # Get table headers as the first row
                        try:
                            headers = table_df.iloc[0].tolist()
                            table_data = table_df.iloc[1:]
                        except:
                            headers = table_df.columns.tolist()
                            table_data = table_df
                        
                        # Process this table in chunks if it's large
                        table_chunks, table_metadata = self._chunk_dataframe(
                            table_data, 
                            f"{sheet_name}_Table{table_idx+1}", 
                            headers, 
                            document_id,
                            file_path
                        )
                        
                        chunks.extend(table_chunks)
                        chunk_metadata.extend(table_metadata)
                        
                else:
                    # No tables identified, process the sheet as one table
                    logger.info(f"No distinct tables found in {sheet_name}, processing as one table")
                    sheet_chunks, sheet_metadata = self._chunk_dataframe(
                        df, 
                        sheet_name, 
                        df.columns.tolist(), 
                        document_id,
                        file_path
                    )
                    
                    chunks.extend(sheet_chunks)
                    chunk_metadata.extend(sheet_metadata)
                    
            except Exception as e:
                logger.error(f"Error processing sheet {sheet_name}: {str(e)}", exc_info=True)
                # Continue with other sheets despite errors
                
        chunking_time = __import__('time').time() - start_time
        
        # Record metrics if available
        try:
            metrics_collector.record_stage_timing(
                document_id, 
                ProcessingStage.CHUNKING, 
                chunking_time,
                success=True,
                metadata={"chunks_count": len(chunks), "chunking_strategy": "table"}
            )
        except (ImportError, AttributeError):
            pass
            
        return chunks, chunk_metadata
    
    def _process_as_sheets(
        self, excel_file: pd.ExcelFile, file_path: Path, document_id: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process Excel file treating each worksheet as a separate document.
        Simple approach that maintains sheet context.
        
        Args:
            excel_file: The pandas ExcelFile object
            file_path: Path to the Excel file
            document_id: Unique ID for the document
            
        Returns:
            Tuple of (text chunks, chunk metadata)
        """
        start_time = __import__('time').time()
        chunks = []
        chunk_metadata = []
        
        for sheet_name in excel_file.sheet_names:
            logger.info(f"Processing sheet: {sheet_name}")
            
            try:
                # Read the sheet
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Skip empty sheets
                if df.empty:
                    logger.info(f"Sheet {sheet_name} is empty, skipping")
                    continue
                
                # Convert sheet to text representation
                sheet_text = f"Sheet: {sheet_name}\n\n"
                sheet_text += df.to_string(index=False)
                
                # Chunk the sheet text
                sheet_chunks = self.text_chunker.chunk_text(
                    sheet_text,
                    strategy=ChunkingStrategy.PARAGRAPH,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                
                # Create metadata for each chunk
                for i, chunk in enumerate(sheet_chunks):
                    metadata = {
                        "document_id": document_id,
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "sheet_name": sheet_name,
                        "chunk_index": i,
                        "total_chunks": len(sheet_chunks),
                        "chunk_type": "sheet_text",
                    }
                    
                    chunks.append(chunk)
                    chunk_metadata.append(metadata)
                    
                logger.info(f"Generated {len(sheet_chunks)} chunks from sheet {sheet_name}")
                
            except Exception as e:
                logger.error(f"Error processing sheet {sheet_name}: {str(e)}", exc_info=True)
                # Continue with other sheets despite errors
        
        chunking_time = __import__('time').time() - start_time
        
        # Record metrics if available
        try:
            metrics_collector.record_stage_timing(
                document_id, 
                ProcessingStage.CHUNKING, 
                chunking_time,
                success=True,
                metadata={"chunks_count": len(chunks), "chunking_strategy": "sheet"}
            )
        except (ImportError, AttributeError):
            pass
            
        return chunks, chunk_metadata
    
    def _process_generic(
        self, excel_file: pd.ExcelFile, file_path: Path, document_id: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process Excel file in a generic way suitable for any Excel document.
        Combines all sheets into one document and chunks it.
        
        Args:
            excel_file: The pandas ExcelFile object
            file_path: Path to the Excel file
            document_id: Unique ID for the document
            
        Returns:
            Tuple of (text chunks, chunk metadata)
        """
        start_time = __import__('time').time()
        
        # Combine all sheets into one text document
        all_text = f"Excel File: {file_path.name}\n\n"
        
        for sheet_name in excel_file.sheet_names:
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if not df.empty:
                    sheet_text = f"Sheet: {sheet_name}\n"
                    sheet_text += df.to_string(index=False) + "\n\n"
                    all_text += sheet_text
            except Exception as e:
                logger.error(f"Error reading sheet {sheet_name}: {str(e)}")
                
        # Chunk the combined text
        chunking_strategy = ChunkingStrategy.RECURSIVE
        chunks = self.text_chunker.chunk_text(
            all_text,
            strategy=chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            max_recursion_depth=3
        )
        
        # Create metadata for each chunk
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "document_id": document_id,
                "file_name": file_path.name,
                "file_path": str(file_path),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_type": "generic_excel",
            }
            chunk_metadata.append(metadata)
            
        chunking_time = __import__('time').time() - start_time
        
        # Record metrics if available
        try:
            metrics_collector.record_stage_timing(
                document_id, 
                ProcessingStage.CHUNKING, 
                chunking_time,
                success=True,
                metadata={"chunks_count": len(chunks), "chunking_strategy": "generic"}
            )
        except (ImportError, AttributeError):
            pass
        
        return chunks, chunk_metadata
    
    def _identify_tables_in_dataframe(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Identify potential tables within a DataFrame by looking for header rows.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of (start_row, end_row) tuples for potential tables
        """
        potential_tables = []
        current_table_start = None
        empty_row_count = 0
        
        # Look for patterns of headers followed by data
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Check if row is empty (all NaN or empty strings)
            is_empty = row.isna().all() or (row.astype(str).str.strip() == '').all()
            
            if is_empty:
                empty_row_count += 1
                # If we've seen multiple empty rows and have a table start,
                # this could be the end of a table
                if empty_row_count >= 2 and current_table_start is not None:
                    potential_tables.append((current_table_start, i - empty_row_count))
                    current_table_start = None
                    empty_row_count = 0
            else:
                empty_row_count = 0
                
                # Check if this looks like a header row (strings, not numbers)
                is_header = isinstance(row.iloc[0], str) if len(row) > 0 else False
                
                # If no table in progress and this might be a header, start a new table
                if current_table_start is None and is_header:
                    current_table_start = i
        
        # Don't forget a potential table at the end
        if current_table_start is not None:
            potential_tables.append((current_table_start, len(df)))
            
        # If no tables found, consider the whole sheet as one table
        if not potential_tables:
            potential_tables.append((0, len(df)))
            
        return potential_tables
    
    def _chunk_dataframe(
        self, 
        df: pd.DataFrame, 
        context_name: str, 
        headers: List[str],
        document_id: str,
        file_path: Path
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Chunk a DataFrame into manageable text pieces.
        
        Args:
            df: DataFrame to chunk
            context_name: Name for context (sheet or table name)
            headers: Column headers
            document_id: Unique ID for the document
            file_path: Path to the Excel file
            
        Returns:
            Tuple of (text chunks, chunk metadata)
        """
        chunks = []
        chunk_metadata = []
        
        # Skip empty dataframes
        if df.empty:
            return chunks, chunk_metadata
            
        # For large tables, chunk by rows
        total_rows = len(df)
        
        if total_rows <= self.max_rows_per_chunk:
            # Small enough to process as one chunk
            chunk_text = self._format_dataframe_chunk(df, headers, context_name)
            
            metadata = {
                "document_id": document_id,
                "file_name": file_path.name,
                "file_path": str(file_path),
                "context_name": context_name,
                "rows": total_rows,
                "columns": len(headers),
                "chunk_index": 0,
                "total_chunks": 1,
                "row_start": 0,
                "row_end": total_rows,
                "chunk_type": "excel_table",
            }
            
            chunks.append(chunk_text)
            chunk_metadata.append(metadata)
            
        else:
            # Break into chunks by rows
            for i in range(0, total_rows, self.max_rows_per_chunk):
                end_idx = min(i + self.max_rows_per_chunk, total_rows)
                chunk_df = df.iloc[i:end_idx]
                
                chunk_text = self._format_dataframe_chunk(chunk_df, headers, context_name)
                
                metadata = {
                    "document_id": document_id,
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "context_name": context_name,
                    "rows": len(chunk_df),
                    "columns": len(headers),
                    "chunk_index": len(chunks),
                    "row_start": i,
                    "row_end": end_idx,
                    "chunk_type": "excel_table",
                }
                
                chunks.append(chunk_text)
                chunk_metadata.append(metadata)
                
        return chunks, chunk_metadata
    
    def _format_dataframe_chunk(
        self, df: pd.DataFrame, headers: List[str], context_name: str
    ) -> str:
        """
        Format a DataFrame chunk as text.
        
        Args:
            df: DataFrame chunk
            headers: Column headers
            context_name: Name for context (sheet or table name)
            
        Returns:
            Formatted text representation of the DataFrame
        """
        # Start with context
        text = f"Context: {context_name}\n\n"
        
        # Add headers if requested
        if self.include_headers:
            text += "Headers: " + " | ".join(str(h) for h in headers) + "\n\n"
        
        # Convert data to text
        # Use to_string for better formatting of tabular data
        text += df.to_string(index=False)
        
        return text