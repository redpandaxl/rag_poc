#!/usr/bin/env python3
"""
Simple script to run the RAGStack application components directly.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_api():
    """Run the API server."""
    print("Starting API server...")
    cmd = [sys.executable, "-m", "uvicorn", "ragstack.web.api:app", "--host", "0.0.0.0", "--port", "8000"]
    return subprocess.Popen(cmd)

def run_streamlit():
    """Run the Streamlit UI."""
    print("Starting Streamlit UI...")
    cmd = [sys.executable, "-m", "streamlit", "run", "ragstack/web/streamlit_app.py", 
           "--server.address", "localhost", "--server.port", "8501"]
    return subprocess.Popen(cmd)

def run_ingestion(watch=True):
    """Run the document ingestion process."""
    print("Starting document ingestion...")
    from ragstack.data_ingestion.file_watcher import FileWatcher
    file_watcher = FileWatcher()
    
    if watch:
        file_watcher.run_forever()
    else:
        file_watcher.process_existing_files()

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "api":
            process = run_api()
            process.wait()
        elif command == "ui":
            process = run_streamlit()
            process.wait()
        elif command == "ingest":
            watch = "--watch" in sys.argv
            run_ingestion(watch=watch)
        else:
            print(f"Unknown command: {command}")
            return 1
    else:
        # Run both API and UI
        api_process = run_api()
        time.sleep(1)  # Give the API a moment to start
        ui_process = run_streamlit()
        
        print("\nRAGStack is now running!")
        print("• API: http://localhost:8000")
        print("• Web UI: http://localhost:8501")
        print("\nPress Ctrl+C to stop...\n")
        
        try:
            # Keep the script running until interrupted
            ui_process.wait()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            # Clean up processes
            if api_process.poll() is None:
                api_process.terminate()
            if ui_process.poll() is None:
                ui_process.terminate()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())