"""
Metrics collection for monitoring the RAG pipeline.
"""
import time
import threading
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import os
import json

from ragstack.config.settings import settings
from ragstack.utils.logging import setup_logger

# Initialize logger
logger = setup_logger("ragstack.utils.monitoring")

class ProcessingState(str, Enum):
    """Processing states for documents in the pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStage(str, Enum):
    """Processing stages in the document pipeline."""
    EXTRACTION = "extraction"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    VECTORDB = "vectordb"
    METADATA = "metadata"


class MetricsCollector:
    """
    Collects and stores metrics about the RAG pipeline processing.
    Uses SQLite for persistent storage of metrics.
    """
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        """Singleton pattern to ensure only one metrics collector exists."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MetricsCollector, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Initialize the metrics collector and database."""
        with self._lock:
            if self._initialized:
                return
                
            self._db_path = settings.logs_dir / "metrics.db"
            self._ensure_db_exists()
            
            # In-memory cache for recent metrics to avoid DB overhead
            self._document_states: Dict[str, ProcessingState] = {}
            self._stage_timings: Dict[str, Dict[ProcessingStage, float]] = {}
            self._error_counts: Dict[str, int] = {}
            
            # Start time for tracking uptime
            self._start_time = time.time()
            self._initialized = True
            logger.info(f"Metrics collector initialized with database at {self._db_path}")

    def _ensure_db_exists(self) -> None:
        """Create the metrics database and tables if they don't exist."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()
            
            # Create document processing table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS document_processing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT,
                    document_name TEXT,
                    document_type TEXT,
                    state TEXT,
                    error_message TEXT,
                    start_time REAL,
                    end_time REAL,
                    total_chunks INTEGER,
                    timestamp REAL
                )
            ''')
            
            # Create stage metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stage_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT,
                    stage TEXT,
                    start_time REAL,
                    end_time REAL,
                    duration REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    metadata TEXT,
                    timestamp REAL
                )
            ''')
            
            # Create system metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.debug("Metrics database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing metrics database: {e}")

    def start_document_processing(self, document_id: str, document_name: str, document_type: str) -> None:
        """
        Record the start of document processing.
        
        Args:
            document_id: Unique identifier for the document
            document_name: Name of the document
            document_type: Type/format of the document (e.g., pdf, docx)
        """
        with self._lock:
            try:
                self._document_states[document_id] = ProcessingState.PROCESSING
                start_time = time.time()
                
                conn = sqlite3.connect(str(self._db_path))
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO document_processing 
                    (document_id, document_name, document_type, state, start_time, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''',
                    (document_id, document_name, document_type, 
                     ProcessingState.PROCESSING, start_time, start_time)
                )
                conn.commit()
                conn.close()
                logger.debug(f"Started processing document: {document_name} ({document_id})")
            except Exception as e:
                logger.error(f"Error recording document processing start: {e}")

    def complete_document_processing(
        self, 
        document_id: str, 
        success: bool, 
        total_chunks: int = 0, 
        error_message: Optional[str] = None
    ) -> None:
        """
        Record the completion of document processing.
        
        Args:
            document_id: Unique identifier for the document
            success: Whether processing was successful
            total_chunks: Total number of chunks created
            error_message: Error message if processing failed
        """
        with self._lock:
            try:
                state = ProcessingState.COMPLETED if success else ProcessingState.FAILED
                self._document_states[document_id] = state
                end_time = time.time()
                
                conn = sqlite3.connect(str(self._db_path))
                cursor = conn.cursor()
                
                # Find the document record to update
                cursor.execute(
                    "SELECT start_time FROM document_processing WHERE document_id = ? ORDER BY id DESC LIMIT 1", 
                    (document_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    start_time = result[0]
                    cursor.execute(
                        '''
                        UPDATE document_processing
                        SET state = ?, end_time = ?, total_chunks = ?, error_message = ?
                        WHERE document_id = ? AND start_time = ?
                        ''',
                        (state, end_time, total_chunks, error_message, document_id, start_time)
                    )
                    conn.commit()
                    
                    if not success and error_message:
                        self._increment_error_count(error_message[:50])  # Use first 50 chars as key
                    
                    logger.debug(
                        f"Completed processing document {document_id}: success={success}, "
                        f"chunks={total_chunks}, duration={end_time - start_time:.2f}s"
                    )
                else:
                    logger.warning(f"No start record found for document {document_id}")
                
                conn.close()
            except Exception as e:
                logger.error(f"Error recording document processing completion: {e}")

    def record_stage_timing(
        self, 
        document_id: str, 
        stage: ProcessingStage, 
        duration: float,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record timing for a processing stage.
        
        Args:
            document_id: Unique identifier for the document
            stage: Processing stage
            duration: Time taken for the stage in seconds
            success: Whether the stage was successful
            error_message: Error message if stage failed
            metadata: Additional metadata about the stage processing
        """
        with self._lock:
            try:
                # Cache the timing in memory
                if document_id not in self._stage_timings:
                    self._stage_timings[document_id] = {}
                self._stage_timings[document_id][stage] = duration
                
                # Store in database
                timestamp = time.time()
                end_time = timestamp
                start_time = end_time - duration
                
                conn = sqlite3.connect(str(self._db_path))
                cursor = conn.cursor()
                
                metadata_json = json.dumps(metadata) if metadata else None
                
                cursor.execute(
                    '''
                    INSERT INTO stage_metrics
                    (document_id, stage, start_time, end_time, duration, success, error_message, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (document_id, stage, start_time, end_time, duration, 
                     1 if success else 0, error_message, metadata_json, timestamp)
                )
                
                conn.commit()
                conn.close()
                
                logger.debug(
                    f"Recorded {stage} timing for {document_id}: {duration:.2f}s, "
                    f"success={success}"
                )
            except Exception as e:
                logger.error(f"Error recording stage timing: {e}")

    def _increment_error_count(self, error_key: str) -> None:
        """Increment the count for a specific error type."""
        if error_key in self._error_counts:
            self._error_counts[error_key] += 1
        else:
            self._error_counts[error_key] = 1

    def record_system_metric(self, metric_name: str, metric_value: float) -> None:
        """
        Record a system-level metric.
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
        """
        try:
            timestamp = time.time()
            
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO system_metrics
                (metric_name, metric_value, timestamp)
                VALUES (?, ?, ?)
                ''',
                (metric_name, metric_value, timestamp)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error recording system metric: {e}")

    def get_processing_stats(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Get processing statistics for the given time window.
        
        Args:
            time_window: Time window in seconds, or None for all time
            
        Returns:
            Dictionary of processing statistics
        """
        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()
            
            timestamp_filter = ""
            params = []
            
            if time_window is not None:
                cutoff_time = time.time() - time_window
                timestamp_filter = "WHERE timestamp > ?"
                params = [cutoff_time]
            
            # Count documents by state
            cursor.execute(
                f"""
                SELECT state, COUNT(*) 
                FROM document_processing 
                {timestamp_filter}
                GROUP BY state
                """, 
                params
            )
            state_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get average processing time for completed documents
            cursor.execute(
                f"""
                SELECT AVG(end_time - start_time) 
                FROM document_processing 
                WHERE state = ? {timestamp_filter.replace('WHERE', 'AND') if timestamp_filter else ''}
                """, 
                [ProcessingState.COMPLETED] + params
            )
            avg_processing_time = cursor.fetchone()[0] or 0
            
            # Count documents by type
            cursor.execute(
                f"""
                SELECT document_type, COUNT(*) 
                FROM document_processing 
                {timestamp_filter}
                GROUP BY document_type
                """, 
                params
            )
            type_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get average stage timings
            cursor.execute(
                f"""
                SELECT stage, AVG(duration) 
                FROM stage_metrics 
                {timestamp_filter}
                GROUP BY stage
                """, 
                params
            )
            avg_stage_timings = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get error counts by stage
            cursor.execute(
                f"""
                SELECT stage, COUNT(*) 
                FROM stage_metrics 
                WHERE success = 0 {timestamp_filter.replace('WHERE', 'AND') if timestamp_filter else ''}
                GROUP BY stage
                """, 
                params
            )
            stage_error_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get most recent documents
            cursor.execute(
                f"""
                SELECT document_id, document_name, document_type, state, start_time, end_time, total_chunks, error_message
                FROM document_processing
                {timestamp_filter}
                ORDER BY timestamp DESC
                LIMIT 10
                """,
                params
            )
            
            recent_documents = []
            for row in cursor.fetchall():
                doc = {
                    "document_id": row[0],
                    "document_name": row[1],
                    "document_type": row[2],
                    "state": row[3],
                    "start_time": row[4],
                    "end_time": row[5],
                    "total_chunks": row[6],
                    "error_message": row[7],
                }
                if doc["start_time"] and doc["end_time"]:
                    doc["duration"] = doc["end_time"] - doc["start_time"]
                recent_documents.append(doc)
            
            conn.close()
            
            # Calculate uptime
            uptime = time.time() - self._start_time
            
            return {
                "document_counts": {
                    "total": sum(state_counts.values()),
                    "by_state": state_counts,
                    "by_type": type_counts
                },
                "processing_times": {
                    "average_total": avg_processing_time,
                    "average_by_stage": avg_stage_timings
                },
                "errors": {
                    "by_stage": stage_error_counts,
                    "total": sum(stage_error_counts.values() if stage_error_counts else [0])
                },
                "recent_documents": recent_documents,
                "system": {
                    "uptime": uptime
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {}

    def get_document_details(self, document_id: str) -> Dict[str, Any]:
        """
        Get detailed metrics for a specific document.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            Dictionary of document processing details
        """
        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()
            
            # Get document info
            cursor.execute(
                """
                SELECT document_id, document_name, document_type, state, 
                       start_time, end_time, total_chunks, error_message
                FROM document_processing
                WHERE document_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (document_id,)
            )
            
            doc_row = cursor.fetchone()
            if not doc_row:
                conn.close()
                return {"error": f"Document {document_id} not found"}
            
            document = {
                "document_id": doc_row[0],
                "document_name": doc_row[1],
                "document_type": doc_row[2],
                "state": doc_row[3],
                "start_time": doc_row[4],
                "end_time": doc_row[5],
                "total_chunks": doc_row[6],
                "error_message": doc_row[7],
            }
            
            if document["start_time"] and document["end_time"]:
                document["duration"] = document["end_time"] - document["start_time"]
            
            # Get stage metrics
            cursor.execute(
                """
                SELECT stage, start_time, end_time, duration, success, error_message, metadata
                FROM stage_metrics
                WHERE document_id = ?
                ORDER BY start_time
                """,
                (document_id,)
            )
            
            stages = []
            for row in cursor.fetchall():
                stage_data = {
                    "stage": row[0],
                    "start_time": row[1],
                    "end_time": row[2],
                    "duration": row[3],
                    "success": bool(row[4]),
                    "error_message": row[5],
                }
                
                if row[6]:  # metadata JSON
                    try:
                        stage_data["metadata"] = json.loads(row[6])
                    except:
                        stage_data["metadata"] = {}
                
                stages.append(stage_data)
            
            document["stages"] = stages
            
            conn.close()
            return document
            
        except Exception as e:
            logger.error(f"Error getting document details: {e}")
            return {"error": str(e)}

# Singleton instance
metrics_collector = MetricsCollector()