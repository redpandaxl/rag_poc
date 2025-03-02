"""
Streamlit dashboard for monitoring the RAG pipeline.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import threading
import json
from typing import Dict, List, Optional, Any, Tuple
import os
from pathlib import Path

from ragstack.utils.monitoring.metrics import metrics_collector
from ragstack.utils.monitoring.system_monitor import system_monitor
from ragstack.config.settings import settings
from ragstack.utils.logging import setup_logger

# Initialize logger
logger = setup_logger("ragstack.web.dashboard")

# Constants for refresh rates
REFRESH_RATES = {
    "Live (5s)": 5,
    "10 seconds": 10,
    "30 seconds": 30,
    "1 minute": 60,
    "5 minutes": 300,
    "Manual": None
}

# Time windows for metrics display
TIME_WINDOWS = {
    "Last hour": 3600,
    "Last 6 hours": 21600,
    "Last 12 hours": 43200,
    "Last 24 hours": 86400,
    "All time": None
}

class DashboardState:
    """Manage global dashboard state to persist between Streamlit reruns."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DashboardState, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Initialize dashboard state
        self.last_update_time = None
        self.system_metrics_history = {
            "timestamp": [],
            "cpu.total_percent": [],
            "memory.percent": [],
            "disk.read_mbps": [],
            "disk.write_mbps": [],
            "process.memory_mb": []
        }
        
        # Dashboard settings
        self.auto_refresh = True
        self.refresh_rate = 5  # seconds
        self.time_window = 3600  # 1 hour
        
        # Start system monitoring
        system_monitor.start_monitoring()
        system_monitor.register_callback(self._record_system_metrics)
        
        self._initialized = True
        logger.info("Dashboard state initialized")
        
    def _record_system_metrics(self, metrics: Dict[str, float]) -> None:
        """Record system metrics for the dashboard charts."""
        # Add timestamp
        timestamp = time.time()
        self.system_metrics_history["timestamp"].append(timestamp)
        
        # Record each metric we care about for the dashboard
        for key in self.system_metrics_history:
            if key != "timestamp" and key in metrics:
                self.system_metrics_history[key].append(metrics[key])
            elif key != "timestamp":
                # If metric isn't in the current batch, append None
                self.system_metrics_history[key].append(None)
        
        # Limit history size to avoid memory issues
        max_history = 1000
        if len(self.system_metrics_history["timestamp"]) > max_history:
            for key in self.system_metrics_history:
                self.system_metrics_history[key] = self.system_metrics_history[key][-max_history:]
    
    def get_system_metrics_df(self, window_seconds: Optional[int] = None) -> pd.DataFrame:
        """
        Get system metrics as a DataFrame for the specified time window.
        
        Args:
            window_seconds: Time window in seconds, or None for all history
            
        Returns:
            DataFrame of system metrics history
        """
        df = pd.DataFrame(self.system_metrics_history)
        
        # Filter by time window if specified
        if window_seconds and not df.empty:
            cutoff_time = time.time() - window_seconds
            df = df[df["timestamp"] >= cutoff_time]
            
        # Convert timestamp to datetime for better plotting
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            
        return df

# Global dashboard state
dashboard_state = DashboardState()

def format_timedelta(seconds: float) -> str:
    """Format seconds as a human-readable time duration."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    else:
        days = seconds / 86400
        return f"{days:.1f} days"

def create_progress_dashboard():
    """Main function to render the Streamlit dashboard."""
    st.set_page_config(
        page_title="RAG Pipeline Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 RAG Pipeline Dashboard")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Dashboard Settings")
        
        # Refresh settings
        st.subheader("Refresh Settings")
        dashboard_state.auto_refresh = st.toggle("Auto-refresh", value=dashboard_state.auto_refresh)
        
        refresh_rate_option = st.selectbox(
            "Refresh Rate", 
            options=list(REFRESH_RATES.keys()),
            index=0 if dashboard_state.refresh_rate == 5 else list(REFRESH_RATES.values()).index(dashboard_state.refresh_rate)
        )
        dashboard_state.refresh_rate = REFRESH_RATES[refresh_rate_option]
        
        if not dashboard_state.auto_refresh:
            st.button("Refresh Now", use_container_width=True, on_click=lambda: setattr(dashboard_state, "last_update_time", time.time()))
        
        # Time window selection
        st.subheader("Time Window")
        time_window_option = st.selectbox(
            "Display data from", 
            options=list(TIME_WINDOWS.keys()),
            index=0  # Default to "Last hour"
        )
        dashboard_state.time_window = TIME_WINDOWS[time_window_option]
        
        # System info
        st.subheader("System Information")
        system_info = system_monitor.get_system_info()
        st.write(f"OS: {system_info['os']} ({system_info['platform']})")
        st.write(f"CPU: {system_info['processor']}")
        st.write(f"CPU Cores: {system_info['physical_cpu_count']} physical, {system_info['cpu_count']} logical")
        st.write(f"Python: {system_info['python_version']}")
    
    # Auto-refresh logic
    if dashboard_state.auto_refresh and dashboard_state.refresh_rate:
        if (dashboard_state.last_update_time is None or 
            (time.time() - dashboard_state.last_update_time) >= dashboard_state.refresh_rate):
            dashboard_state.last_update_time = time.time()
            st.empty()  # Force a rerun
    
    # Get processing statistics
    stats = metrics_collector.get_processing_stats(dashboard_state.time_window)
    
    # Top-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_docs = stats.get("document_counts", {}).get("total", 0)
        st.metric("Total Documents", total_docs)
    
    with col2:
        completed_docs = stats.get("document_counts", {}).get("by_state", {}).get("completed", 0)
        st.metric("Completed Documents", completed_docs)
    
    with col3:
        failed_docs = stats.get("document_counts", {}).get("by_state", {}).get("failed", 0)
        st.metric("Failed Documents", failed_docs)
    
    with col4:
        avg_time = stats.get("processing_times", {}).get("average_total", 0)
        st.metric("Avg. Processing Time", format_timedelta(avg_time))
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Document Processing", "Pipeline Performance", "System Resources", "Logs"])
    
    # Document Processing Tab
    with tab1:
        st.header("Document Processing Status")
        
        # Document status summary
        state_counts = stats.get("document_counts", {}).get("by_state", {})
        
        # Create status summary chart
        if state_counts:
            status_df = pd.DataFrame({
                "Status": list(state_counts.keys()),
                "Count": list(state_counts.values())
            })
            
            fig = px.pie(
                status_df, 
                values="Count", 
                names="Status", 
                title="Document Status",
                color="Status",
                color_discrete_map={
                    "completed": "green",
                    "failed": "red",
                    "processing": "blue",
                    "pending": "gray"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No document processing data available.")
        
        # Document type distribution
        type_counts = stats.get("document_counts", {}).get("by_type", {})
        if type_counts:
            st.subheader("Documents by Type")
            type_df = pd.DataFrame({
                "Document Type": list(type_counts.keys()),
                "Count": list(type_counts.values())
            })
            
            fig = px.bar(
                type_df, 
                x="Document Type", 
                y="Count",
                title="Document Count by Type",
                color="Document Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent documents table
        st.subheader("Recent Documents")
        recent_docs = stats.get("recent_documents", [])
        
        if recent_docs:
            docs_df = pd.DataFrame(recent_docs)
            
            # Format timestamps
            if "start_time" in docs_df.columns:
                docs_df["start_time"] = pd.to_datetime(docs_df["start_time"], unit="s")
            if "end_time" in docs_df.columns:
                docs_df["end_time"] = pd.to_datetime(docs_df["end_time"], unit="s")
            
            # Format duration
            if "duration" in docs_df.columns:
                docs_df["duration"] = docs_df["duration"].apply(lambda x: f"{x:.2f}s" if x else "")
            
            # Select and rename columns for display
            display_cols = ["document_name", "document_type", "state", "start_time", "duration", "total_chunks"]
            rename_map = {
                "document_name": "Document",
                "document_type": "Type",
                "state": "Status",
                "start_time": "Started",
                "duration": "Duration",
                "total_chunks": "Chunks"
            }
            
            display_df = docs_df[display_cols].rename(columns=rename_map)
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No recent document processing activity.")
    
    # Pipeline Performance Tab
    with tab2:
        st.header("Pipeline Performance")
        
        # Stage timing charts
        avg_stage_timings = stats.get("processing_times", {}).get("average_by_stage", {})
        stage_error_counts = stats.get("errors", {}).get("by_stage", {})
        
        if avg_stage_timings:
            # Prepare data
            stage_df = pd.DataFrame({
                "Stage": list(avg_stage_timings.keys()),
                "Average Time (s)": list(avg_stage_timings.values())
            })
            
            # Add error counts if available
            if stage_error_counts:
                stage_df["Error Count"] = stage_df["Stage"].map(
                    lambda x: stage_error_counts.get(x, 0)
                )
            
            # Plot stage timing bar chart
            st.subheader("Average Processing Time by Stage")
            fig = px.bar(
                stage_df,
                x="Stage",
                y="Average Time (s)",
                title="Average Time per Processing Stage",
                color="Stage"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot error counts by stage if available
            if "Error Count" in stage_df.columns and stage_df["Error Count"].sum() > 0:
                st.subheader("Errors by Processing Stage")
                fig = px.bar(
                    stage_df,
                    x="Stage",
                    y="Error Count",
                    title="Error Count by Processing Stage",
                    color="Stage"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pipeline performance data available.")
        
        # Processing time distribution over time
        # This would require additional metrics tracking over time
        st.subheader("Processing Pipeline Visualization")
        
        # Create a Sankey diagram of the document flow
        labels = ["Raw Files", "Extraction", "Chunking", "Embedding", "Vector DB"]
        
        # Example values - in a real implementation these would come from the metrics
        # The values represent the flow between stages
        source = [0, 1, 2, 3]  # From nodes
        target = [1, 2, 3, 4]  # To nodes
        
        # Get counts for each stage (this is example data since we don't track this yet)
        # In a real implementation, these would be actual counts from the metrics
        raw_count = stats.get("document_counts", {}).get("total", 100)
        extraction_success = raw_count - stats.get("errors", {}).get("by_stage", {}).get("extraction", 0)
        chunking_success = extraction_success - stats.get("errors", {}).get("by_stage", {}).get("chunking", 0)
        embedding_success = chunking_success - stats.get("errors", {}).get("by_stage", {}).get("embedding", 0)
        
        value = [raw_count, extraction_success, chunking_success, embedding_success]
        
        # Create the Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["blue", "green", "green", "green", "green"]
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])
        
        fig.update_layout(title_text="Document Processing Flow", font_size=10)
        st.plotly_chart(fig, use_container_width=True)
    
    # System Resources Tab
    with tab3:
        st.header("System Resource Usage")
        
        # Get system metrics from history
        system_df = dashboard_state.get_system_metrics_df(dashboard_state.time_window)
        
        if not system_df.empty:
            # CPU usage over time
            st.subheader("CPU Usage")
            fig = px.line(
                system_df, 
                x="datetime", 
                y="cpu.total_percent",
                title="CPU Usage Over Time (%)",
                labels={"cpu.total_percent": "CPU Usage (%)", "datetime": "Time"}
            )
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
            
            # Memory usage over time
            st.subheader("Memory Usage")
            fig = px.line(
                system_df, 
                x="datetime", 
                y="memory.percent",
                title="Memory Usage Over Time (%)",
                labels={"memory.percent": "Memory Usage (%)", "datetime": "Time"}
            )
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
            
            # Process memory usage
            st.subheader("RAG Process Memory Usage")
            fig = px.line(
                system_df, 
                x="datetime", 
                y="process.memory_mb",
                title="Process Memory Usage Over Time (MB)",
                labels={"process.memory_mb": "Memory (MB)", "datetime": "Time"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Disk I/O
            st.subheader("Disk I/O")
            disk_df = system_df[["datetime", "disk.read_mbps", "disk.write_mbps"]].copy()
            disk_df = pd.melt(
                disk_df,
                id_vars=["datetime"],
                value_vars=["disk.read_mbps", "disk.write_mbps"],
                var_name="Metric",
                value_name="Value"
            )
            disk_df["Metric"] = disk_df["Metric"].map({
                "disk.read_mbps": "Read (MB/s)",
                "disk.write_mbps": "Write (MB/s)"
            })
            
            fig = px.line(
                disk_df,
                x="datetime",
                y="Value",
                color="Metric",
                title="Disk I/O Over Time",
                labels={"Value": "MB/s", "datetime": "Time"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No system resource metrics available yet. They will appear as they are collected.")
    
    # Logs Tab
    with tab4:
        st.header("Processing Logs")
        
        # Show processing log files
        log_file = settings.logs_dir / "processing.log"
        
        if log_file.exists():
            # Read the last N lines of the log file
            max_lines = 100
            log_lines = []
            
            with open(log_file, 'r') as f:
                for line in f:
                    log_lines.append(line.strip())
                    if len(log_lines) > max_lines:
                        log_lines.pop(0)
            
            # Display logs with most recent at the top
            log_lines.reverse()
            st.code("\n".join(log_lines), language="bash")
        else:
            st.info(f"No processing log file found at {log_file}")
    
    # Footer with update time
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set up auto-refresh if enabled
    if dashboard_state.auto_refresh and dashboard_state.refresh_rate:
        st.empty()  # Force rerun after the refresh rate interval

if __name__ == "__main__":
    create_progress_dashboard()