#!/usr/bin/env bash
# ==============================================================================
# Script to launch AST training inside a persistent tmux session on HPC
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SESSION_NAME="ast_train"
CLIPS_CSV="${1:-$PROJECT_ROOT/outputs/data/final/csv/clips_metadata.csv}"
OUTPUT_DIR="${2:-$PROJECT_ROOT/outputs/models}"
BATCH_SIZE="${3:-16}"
NUM_WORKERS="${4:-4}"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "[!] tmux could not be found. Please install tmux or run train.py directly."
    exit 1
fi

# Check if session already exists
tmux has-session -t "$SESSION_NAME" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "[-] tmux session '$SESSION_NAME' already exists."
    echo "    Attach with:  tmux attach -t $SESSION_NAME"
    exit 0
fi

echo "[+] Starting new tmux session: '$SESSION_NAME'..."

# Start detached session
tmux new-session -d -s "$SESSION_NAME"

# Send training command into tmux session
CMD="cd '$PROJECT_ROOT' && python3 '$SCRIPT_DIR/train.py' --clips_csv '$CLIPS_CSV' --output_dir '$OUTPUT_DIR' --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS"
tmux send-keys -t "$SESSION_NAME" "$CMD" C-m

echo "[✓] Training process initiated inside tmux session '$SESSION_NAME'!"
echo ""
echo "Useful Commands:"
echo "  1. Attach to session:      tmux attach -t $SESSION_NAME"
echo "  2. Detach from session:    Press [Ctrl + b], then release and press [d]"
echo "  3. Stream live log:        tail -f '$OUTPUT_DIR/train.log'"
echo "  4. Monitor GPU:            watch -n 1 nvidia-smi"
echo "  5. Kill session (abort):   tmux kill-session -t $SESSION_NAME"
echo ""
