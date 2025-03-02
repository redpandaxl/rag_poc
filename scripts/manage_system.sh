#!/bin/bash

# Define colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists and isn't already activated
if [ -d "$PROJECT_ROOT/venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Function to start the system
start_system() {
    echo -e "${BLUE}Starting RAGStack system...${NC}"
    
    # Choose which components to start
    if [ "$1" == "all" ] || [ "$1" == "" ]; then
        echo -e "${BLUE}Starting all components...${NC}"
        # Start main components
        python run.py &
        echo $! > /tmp/ragstack_main.pid
        
        # Also start chat interface
        echo -e "${BLUE}Starting Chat Interface...${NC}"
        python scripts/run_chat.py &
        echo $! > /tmp/ragstack_chat.pid
        
        # Start file watcher for automatic document ingestion
        echo -e "${BLUE}Starting file watcher for document ingestion...${NC}"
        python scripts/run_ingestion.py --watch --debug &
        echo $! > /tmp/ragstack_ingest.pid
        
        echo -e "${GREEN}✓ System started successfully!${NC}"
        echo -e "${BLUE}• API: http://localhost:8000${NC}"
        echo -e "${BLUE}• UI: http://localhost:8501${NC}"
        echo -e "${BLUE}• Chat: http://localhost:8502${NC}"
        echo -e "${BLUE}• File watcher: Monitoring ${PROJECT_ROOT}/data/raw/ for new files${NC}"
    elif [ "$1" == "api" ]; then
        echo -e "${BLUE}Starting API server only...${NC}"
        python run.py api &
        echo $! > /tmp/ragstack_api.pid
        echo -e "${GREEN}✓ API started successfully at http://localhost:8000${NC}"
    elif [ "$1" == "ui" ]; then
        echo -e "${BLUE}Starting UI only...${NC}"
        python run.py ui &
        echo $! > /tmp/ragstack_ui.pid
        echo -e "${GREEN}✓ UI started successfully at http://localhost:8501${NC}"
    elif [ "$1" == "chat" ]; then
        echo -e "${BLUE}Starting Chat Interface...${NC}"
        python scripts/run_chat.py &
        echo $! > /tmp/ragstack_chat.pid
        echo -e "${GREEN}✓ Chat Interface started successfully at http://localhost:8502${NC}"
    elif [ "$1" == "ingest" ]; then
        WATCH_FLAG=""
        if [ "$2" == "watch" ]; then
            WATCH_FLAG="--watch"
            echo -e "${BLUE}Starting document ingestion with file watching...${NC}"
        else
            echo -e "${BLUE}Running one-time document processing...${NC}"
        fi
        python scripts/run_ingestion.py $WATCH_FLAG --debug &
        echo $! > /tmp/ragstack_ingest.pid
        echo -e "${GREEN}✓ Ingestion process started successfully${NC}"
    else
        echo -e "${RED}Unknown component: $1${NC}"
        echo "Usage: $0 start [all|api|ui|ingest [watch]]"
        return 1
    fi
}

# Function to stop the system
stop_system() {
    echo -e "${BLUE}Stopping RAGStack system...${NC}"
    
    # Kill processes by PID file if available
    for pid_file in /tmp/ragstack_*.pid; do
        if [ -f "$pid_file" ]; then
            PID=$(cat "$pid_file")
            COMPONENT=$(basename "$pid_file" | sed 's/ragstack_//;s/\.pid//')
            if ps -p $PID > /dev/null; then
                echo -e "${YELLOW}Stopping $COMPONENT (PID: $PID)...${NC}"
                kill $PID
                rm "$pid_file"
            fi
        fi
    done
    
    # Also try to kill by process name patterns to catch any missed processes
    echo -e "${YELLOW}Cleaning up any remaining processes...${NC}"
    pkill -f "python.*run.py" 2>/dev/null || true
    pkill -f "python.*ragstack" 2>/dev/null || true
    pkill -f "streamlit run" 2>/dev/null || true
    pkill -f "uvicorn" 2>/dev/null || true
    pkill -f "run_ingestion.py" 2>/dev/null || true
    pkill -f "run_chat.py" 2>/dev/null || true
    
    echo -e "${GREEN}✓ All RAGStack processes stopped${NC}"
}

# Function to check system status
check_status() {
    echo -e "${BLUE}Checking RAGStack system status...${NC}"
    
    # Check if any RAGStack processes are running
    RUNNING_PROCESSES=$(ps aux | grep -E "python.*ragstack|streamlit run|uvicorn.*ragstack" | grep -v grep)
    
    if [ -z "$RUNNING_PROCESSES" ]; then
        echo -e "${RED}No RAGStack processes are currently running${NC}"
        return 1
    else
        echo -e "${GREEN}RAGStack system is running:${NC}"
        echo "$RUNNING_PROCESSES" | awk '{print "• PID: " $2 " - " $11 " " $12}'
        
        # Check if API and UI are responding
        if curl -s http://localhost:8000/health > /dev/null; then
            echo -e "${GREEN}✓ API is accessible at http://localhost:8000${NC}"
        else
            echo -e "${RED}✗ API is not responding at http://localhost:8000${NC}"
        fi
        
        if curl -s http://localhost:8501 > /dev/null; then
            echo -e "${GREEN}✓ UI is accessible at http://localhost:8501${NC}"
        else
            echo -e "${RED}✗ UI is not responding at http://localhost:8501${NC}"
        fi
        
        if curl -s http://localhost:8502 > /dev/null; then
            echo -e "${GREEN}✓ Chat Interface is accessible at http://localhost:8502${NC}"
        else
            echo -e "${RED}✗ Chat Interface is not responding at http://localhost:8502${NC}"
        fi
    fi
}

# Main script logic
case "$1" in
    start)
        stop_system  # Stop any existing processes first
        sleep 1
        start_system "${@:2}"
        ;;
    stop)
        stop_system
        ;;
    restart)
        stop_system
        sleep 2
        start_system "${@:2}"
        ;;
    status)
        check_status
        ;;
    *)
        echo -e "${YELLOW}Usage: $0 {start|stop|restart|status}${NC}"
        echo -e "  ${BLUE}start [all|api|ui|chat|ingest [watch]]${NC} - Start all or specific components"
        echo -e "  ${BLUE}stop${NC} - Stop all components"
        echo -e "  ${BLUE}restart [all|api|ui|chat|ingest [watch]]${NC} - Restart all or specific components"
        echo -e "  ${BLUE}status${NC} - Check system status"
        exit 1
        ;;
esac

exit 0