#!/usr/bin/env bash
# Set up and train CSWinUNETR on a RunPod (or any Linux GPU) machine.
#
# With Google Drive connected through rclone (remote named "gdrive", see HOWTO below):
#   bash runpod_train.sh               downloads the dataset from Drive, trains, and copies the
#                                      run folder (best.pt, last.pt, metrics) back to Drive every
#                                      30 minutes and when training ends
# Without rclone:
#   bash runpod_train.sh "<Google Drive share link to ffhq_wrinkle_data.zip>"
#
# Rerun the same command any time: it resumes from last.pt (from /workspace, or from Drive on a new pod).
# Other seed:   SEED=43 bash runpod_train.sh
# Benchmark:    BENCHMARK=1 bash runpod_train.sh   (trains 1 epoch into a test folder, then stops)
#
# Connecting Google Drive (once per pod):
#   On your Mac:  brew install rclone && rclone authorize "drive"     -> copy the token it prints
#   On the pod:   rclone config   -> n (new remote) -> name: gdrive -> storage: drive
#                 -> leave client_id/secret empty -> scope: 1 (full access) -> Enter through the rest
#                 -> "Use web browser to automatically authenticate?": n -> paste the token -> no shared drive -> y
#   When finished with RunPod: delete the pod, and remove "rclone" under
#   myaccount.google.com -> Security -> Third-party access.
set -euo pipefail

WS=${WS:-/workspace}
REPO=$WS/CSWinUNETR
DATA=$WS/data
SEED=${SEED:-42}
RUN_NAME=${RUN_NAME:-baseline-s$SEED}
RUN_DIR=$WS/runs/$RUN_NAME
LOG=$WS/runs/$RUN_NAME.log   # outside RUN_DIR: train.py refuses to start a new run in a non-empty folder

# Google Drive locations (used only if an rclone remote with this name exists).
REMOTE=${REMOTE:-gdrive}
DRIVE_ZIP=${DRIVE_ZIP:-ffhq_wrinkle_data.zip}          # path inside My Drive
DRIVE_RUNS=${DRIVE_RUNS:-cswinunetr_runs/runpod}        # run folders are copied here
SYNC_MINUTES=${SYNC_MINUTES:-30}

# Keep these identical for every run in the study (resume refuses if they change).
BATCH_SIZE=2
WORKERS=8
EPOCHS=300

drive_ok() { command -v rclone > /dev/null && rclone listremotes 2> /dev/null | grep -qx "$REMOTE:"; }
upload_run() {
    # Upload a frozen snapshot, not the live folder: train.py replaces last.pt every epoch, and a
    # file that changes mid-upload fails rclone's checksum check. Hard links keep the old checkpoint
    # intact because train.py writes a new file and renames it; small files are copied outright.
    local snap="$WS/runs/.upload-$RUN_NAME" status=0
    rm -rf "$snap"
    cp -al "$RUN_DIR" "$snap" || return 1
    find "$snap" -type f -name '*.tmp' -delete
    find "$snap" -type f ! -name '*.pt' -exec sh -c 'cp "$1" "$1.copy" && mv "$1.copy" "$1"' _ {} \;
    cp "$LOG" "$snap.log" 2> /dev/null || : > "$snap.log"
    rclone copy "$snap" "$REMOTE:$DRIVE_RUNS/$RUN_NAME" && \
        rclone copyto "$snap.log" "$REMOTE:$DRIVE_RUNS/$RUN_NAME.log" || status=1
    rm -rf "$snap" "$snap.log"
    [ "$status" = 0 ] && echo "[$(date '+%F %T')] copied $RUN_NAME to Drive: $DRIVE_RUNS/$RUN_NAME"
    return "$status"
}

# Background worker: train, and copy the run folder to Drive while training and at the end.
if [ "${1:-}" = "__worker" ]; then
    shift
    cd "$REPO"
    python cswinunetr/train.py "$@" &
    pid=$!
    if drive_ok; then
        waited=0
        while kill -0 "$pid" 2> /dev/null; do
            sleep "${POLL_SECONDS:-60}"
            waited=$((waited + ${POLL_SECONDS:-60}))
            if [ "$waited" -ge $((SYNC_MINUTES * 60)) ] && kill -0 "$pid" 2> /dev/null; then
                waited=0
                upload_run || echo "Drive upload failed; will retry."
            fi
        done
    fi
    status=0
    wait "$pid" || status=$?
    echo "[$(date '+%F %T')] training process exited with status $status"
    if drive_ok; then upload_run || echo "Final Drive upload failed; files are still in $RUN_DIR"; fi
    exit "$status"
fi

echo "== 1/4 Code"
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only
else
    git clone https://github.com/tommy-mnan/CSWinUNETR.git "$REPO"
fi
cd "$REPO"

echo "== 2/4 Install (first run downloads PyTorch 2.5, a few minutes)"
pip install -q -e '.[train,eval]'

echo "== 3/4 Dataset"
if drive_ok; then echo "Google Drive connected (rclone remote '$REMOTE')."; else echo "Google Drive not connected (no rclone remote '$REMOTE')."; fi
if [ ! -d "$DATA/masked_face_images" ]; then
    if [ $# -ge 1 ]; then
        pip install -q gdown
        gdown --fuzzy "$1" -O "$WS/data.zip"
    elif drive_ok; then
        echo "Downloading $DRIVE_ZIP from Drive"
        rclone copyto "$REMOTE:$DRIVE_ZIP" "$WS/data.zip" --progress
    else
        echo "No dataset yet. Connect Google Drive with rclone (see the top of this script),"
        echo "or run: bash runpod_train.sh \"<Google Drive share link to ffhq_wrinkle_data.zip>\""
        exit 1
    fi
    mkdir -p "$DATA"
    unzip -q -o "$WS/data.zip" -d "$DATA"
    # If the zip had one top-level folder, move its contents up.
    nested=$(find "$DATA" -mindepth 2 -maxdepth 2 -type d -name masked_face_images | head -n 1)
    if [ -n "$nested" ] && [ ! -d "$DATA/masked_face_images" ]; then
        top=$(dirname "$nested")
        mv "$top"/* "$DATA"/ && rmdir "$top"
    fi
    rm -f "$WS/data.zip"
    [ $# -ge 1 ] && echo "Dataset ready. Set the Google Drive link back to 'Restricted' now."
fi
for d in masked_face_images weak_wrinkle_masks manual_wrinkle_masks; do
    printf '  %-22s %s PNG files\n' "$d" "$(find "$DATA/$d" -name '*.png' | wc -l)"
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ARGS=(--data-root "$DATA" --device cuda --batch-size "$BATCH_SIZE" --workers "$WORKERS"
      --use-checkpoint --seed "$SEED")

if [ "${BENCHMARK:-0}" = 1 ]; then
    echo "== 4/4 Benchmark: 1 epoch into $WS/runs/benchmark"
    rm -rf "$WS/runs/benchmark"
    python cswinunetr/train.py "${ARGS[@]}" --output-dir "$WS/runs/benchmark" --epochs 1
    python -c "import json; r=json.loads(open('$WS/runs/benchmark/metrics.jsonl').readline()); print('Epoch time:', round(r['epoch_seconds']), 's =', round(r['epoch_seconds']/60, 1), 'min ->', round(r['epoch_seconds']*$EPOCHS/3600, 1), 'h for $EPOCHS epochs')"
    exit 0
fi

echo "== 4/4 Train: $RUN_DIR"
if pgrep -f "train.py.*--output-dir $RUN_DIR( |$)" > /dev/null; then
    echo "Already training into $RUN_DIR. Watch it with: tail -f $LOG"
    exit 0
fi
mkdir -p "$WS/runs"
# New pod: fetch the run from Drive so training resumes instead of starting over.
if [ ! -f "$RUN_DIR/last.pt" ] && drive_ok && rclone lsf "$REMOTE:$DRIVE_RUNS/$RUN_NAME/last.pt" > /dev/null 2>&1; then
    echo "Found $RUN_NAME on Drive -> downloading it to resume"
    rclone copy "$REMOTE:$DRIVE_RUNS/$RUN_NAME" "$RUN_DIR"
    rclone copyto "$REMOTE:$DRIVE_RUNS/$RUN_NAME.log" "$LOG" 2> /dev/null || true
fi
if [ -f "$RUN_DIR/last.pt" ]; then
    echo "Found last.pt -> resuming"
    ARGS+=(--resume)
elif [ -d "$RUN_DIR" ] && [ -n "$(ls -A "$RUN_DIR")" ]; then
    echo "$RUN_DIR has files but no last.pt (the first epoch never finished). Delete it and rerun."
    exit 1
fi
nohup bash "$REPO/runpod_train.sh" __worker "${ARGS[@]}" --output-dir "$RUN_DIR" --epochs "$EPOCHS" \
    --no-progress >> "$LOG" 2>&1 &
echo "Training started in the background (PID $!). It keeps running if you close the browser."
drive_ok && echo "  The run folder is copied to Drive ($DRIVE_RUNS/$RUN_NAME) every $SYNC_MINUTES min and at the end."
echo "  Progress:  tail -f $LOG      (one line per finished epoch; Ctrl+C stops watching, not training)"
echo "  GPU:       nvidia-smi"
echo "  Stop:      pkill -f train.py               (resume later by running this script again)"
