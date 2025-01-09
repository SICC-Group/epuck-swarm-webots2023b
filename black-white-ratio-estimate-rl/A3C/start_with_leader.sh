#!/bin/bash
PYTHON_PATH="/usr/bin/python3"
NUM_WORKERS=6
N=$((NUM_WORKERS / 2))
SCRIPT_PATH=$(realpath "$BASH_SOURCE")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
SERVER_FILE="$SCRIPT_DIR/server.py"
WORKER_FILE="$SCRIPT_DIR/worker.py"

SESSION_NAME="workers"

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
  tmux kill-session -t $SESSION_NAME
fi

tmux new-session -d -s $SESSION_NAME -n Workers
tmux rename-window -t $SESSION_NAME:0 'Workers'
tmux select-window -t $SESSION_NAME:0

if (( NUM_WORKERS % 2 == 1 )); then
    # Odd number of workers
    create_odd_panes() {
        local panes=$1
        local n=$((panes / 2))
        for ((i=0; i<n; i++)); do
            tmux split-window -h
            tmux select-pane -t $((2*i))
            tmux split-window -v
            tmux select-pane -t $((2*i + 1))
        done
        tmux select-pane -t $((panes - 1))
    }
    create_odd_panes $NUM_WORKERS
else
    create_even_panes() {
        local panes=$1
        local n=$((panes / 2))        
        tmux split-window -h
        for ((i=0; i<n-1; i++)); do
            tmux split-window -v -t $i
            tmux split-window -v -t $((2 * (i + 1)))
        done
        tmux select-window -t $((panes - 1))
        tmux select-layout tiled
    }
    create_even_panes $NUM_WORKERS
fi

tmux select-layout tiled
for ((i=0; i<NUM_WORKERS; i++)); do
    tmux select-pane -t $i -T "worker$i" -P 'fg=yellow'
    tmux send-keys -t $i "$PYTHON_PATH $WORKER_FILE --id=$i" C-m
done
tmux attach-session -t workers