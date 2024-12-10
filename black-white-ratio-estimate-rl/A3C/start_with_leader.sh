#!/bin/bash
PYTHON_PATH="/usr/bin/python-deepbots"
NUM_WORKERS=6
N=$((NUM_WORKERS / 2))
SCRIPT_PATH=$(realpath "$BASH_SOURCE")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
SERVER_FILE="$SCRIPT_DIR/server.py"
WORKER_FILE="$SCRIPT_DIR/worker.py"

# # Create a new GNOME terminal window
# gnome-terminal -- bash -c "$PYTHON_PATH $WORKER_FILE --id=0; exec bash"

# if (( (NUM_WORKERS - 1) % 2 == 1 )); then
#     # Odd number of workers
#     gnome-terminal --geometry=150x40+100+100 -- bash -c "
#     create_odd_panes() {
#         local panes=\$1
#         local n=\$((panes / 2))
#         tmux new-session -d -s workers
#         tmux rename-window -t workers:0 'Workers'
#         tmux select-window -t workers:0

#         for ((i=0; i<n; i++)); do
#             tmux split-window -h
#             tmux select-pane -t \$((2*i))
#             tmux split-window -v
#             tmux select-pane -t \$((2*i + 1))
#         done
#         tmux select-pane -t \$((panes - 1))
#         tmux select-layout tiled
#     }
#     create_odd_panes $((NUM_WORKERS - 1))
#     tmux attach-session -t workers
#     exec bash
#     "
# else
#     # Even number of workers
#     gnome-terminal --geometry=150x40+100+100 -- bash -c "
#     create_even_panes() {
#         local panes=\$1
#         local n=\$((panes / 2))
#         tmux new-session -d -s workers
#         tmux rename-window -t workers:0 'Workers'
#         tmux select-window -t workers:0 
        
#         tmux split-window -h
#         for ((i=0; i<n-1; i++)); do
#             tmux split-window -v -t \$i
#             tmux split-window -v -t \$((2 * (i + 1)))
#         done
#         tmux select-window -t \$((panes - 1))
#         tmux select-layout tiled
#     }
#     create_even_panes $((NUM_WORKERS - 1))
#     tmux attach-session -t workers
#     exec bash
#     "
# fi

# for ((i=0; i<NUM_WORKERS-1; i++)); do
#     tmux select-pane -t $i -T "worker$i" -P 'fg=yellow'
#     tmux send-keys -t $i "$PYTHON_PATH $WORKER_FILE --id=$((i + 1))" C-m
# done


# Create a new GNOME terminal window
# gnome-terminal -- bash -c "$PYTHON_PATH $WORKER_FILE --id=0; exec bash"

if (( NUM_WORKERS % 2 == 1 )); then
    # Odd number of workers
    gnome-terminal --geometry=150x40+100+100 -- bash -c "
    create_odd_panes() {
        local panes=\$1
        local n=\$((panes / 2))
        tmux new-session -d -s workers
        tmux rename-window -t workers:0 'Workers'
        tmux select-window -t workers:0

        for ((i=0; i<n; i++)); do
            tmux split-window -h
            tmux select-pane -t \$((2*i))
            tmux split-window -v
            tmux select-pane -t \$((2*i + 1))
        done
        tmux select-pane -t \$((panes - 1))
        tmux select-layout tiled
    }
    create_odd_panes $NUM_WORKERS
    tmux attach-session -t workers
    exec bash
    "
else
    # Even number of workers
    gnome-terminal --geometry=150x40+100+100 -- bash -c "
    create_even_panes() {
        local panes=\$1
        local n=\$((panes / 2))
        tmux new-session -d -s workers
        tmux rename-window -t workers:0 'Workers'
        tmux select-window -t workers:0 
        
        tmux split-window -h
        for ((i=0; i<n-1; i++)); do
            tmux split-window -v -t \$i
            tmux split-window -v -t \$((2 * (i + 1)))
        done
        tmux select-window -t \$((panes - 1))
        tmux select-layout tiled
    }
    create_even_panes $NUM_WORKERS
    tmux attach-session -t workers
    exec bash
    "
fi

for ((i=0; i<NUM_WORKERS; i++)); do
    tmux select-pane -t $i -T "worker$i" -P 'fg=yellow'
    tmux send-keys -t $i "$PYTHON_PATH $WORKER_FILE --id=$i" C-m
done