#!/bin/bash
# Test script for running RICL with LeRobot retrieval on Libero
# This runs 1 task with 2 repetitions

set -e

# Change to script directory
cd "$(dirname "$0")"

# Configuration
CHECKPOINT_CONFIG="pi0_fast_droid_ricl"
CHECKPOINT_DIR="checkpoints/pi0_fast_droid_ricl_checkpoint"  # Downloaded from HuggingFace
DATASET_DIR="/Volumes/T7 4/datasets/meural/libero_10_image"
TEXT_INDEX="/Volumes/T7 4/datasets/meural/libero_10_image/indexes/text_index.faiss"
IMAGE_INDEX="/Volumes/T7 4/datasets/meural/libero_10_image/indexes/image_index.faiss"
PORT=8000

echo "Starting RICL policy server with LeRobot retrieval..."
echo "Dataset: $DATASET_DIR"
echo "Port: $PORT"

# Activate .venv environment
source .venv/bin/activate

# Install missing dependencies if needed (using uv pip)
echo "Checking dependencies..."
python -c "import autofaiss" 2>/dev/null || uv pip install autofaiss
python -c "import libero" 2>/dev/null || uv pip install libero
# Check if third_party/libero exists (for PYTHONPATH)
if [ -d "third_party/libero" ]; then
    export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
fi

# Start policy server in background
python scripts/serve_policy_ricl.py \
  --port "$PORT" \
  policy:checkpoint \
  --policy.config="$CHECKPOINT_CONFIG" \
  --policy.dir="$CHECKPOINT_DIR" \
  --policy.use-lerobot-retrieval \
  --policy.lerobot-dataset-dir="$DATASET_DIR" \
  --policy.lerobot-text-index-path="$TEXT_INDEX" \
  --policy.lerobot-image-index-path="$IMAGE_INDEX" \
  > server.log 2>&1 &

SERVER_PID=$!
echo "Policy server started (PID: $SERVER_PID)"

# Wait for server to start
echo "Waiting for server to start..."
sleep 10

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "✓ Policy server is running (PID: $SERVER_PID)"
    echo "Server logs:"
    tail -20 server.log 2>/dev/null || echo "No server.log found"
    echo ""
    echo "Server should be ready at ws://localhost:$PORT"
    echo "Press Ctrl+C to stop the server"
    # Keep server running
    wait $SERVER_PID
else
    echo "✗ Policy server failed to start. Check server.log for errors:"
    cat server.log 2>/dev/null || echo "No server.log found"
    exit 1
fi

# Commented out libero evaluation for now - focus on getting server running first
echo "Starting Libero evaluation (1 task, 2 repetitions)..."
# Run libero evaluation (using .venv, ensure PYTHONPATH is set)
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero:$PWD/src
python examples/libero/main.py \
    --args.host=localhost \
    --args.port=$PORT \
    --args.task_suite_name=libero_10 \
    --args.num_trials_per_task=2 \
    --args.max_tasks=1 \
    --args.video_out_path=data/libero/test_videos

echo "Evaluation complete. Stopping server..."
kill $SERVER_PID 2>/dev/null || true

echo "Done!"
