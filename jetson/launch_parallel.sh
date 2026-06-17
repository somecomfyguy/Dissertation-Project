#!/bin/bash
# =============================================================================
# launch_parallel.sh
# Launch GNSS-SDR and the ML interference classifier in parallel on Jetson Nano
#
# Architecture:
#   PlutoSDR → demo_inference.py (owns the SDR, writes IQ to FIFO + classifies)
#                    │
#                    ├──→ TensorRT inference → console + CSV log
#                    │
#                    └──→ FIFO pipe ──→ GNSS-SDR (File_Signal_Source)
#                                           │
#                                           └──→ PVT + NMEA + RINEX
#
# Usage:
#   ./launch_parallel.sh --trt model.trt [--uri ip:192.168.2.1] [--csv-out log.csv]
#
# All arguments after the script flags are forwarded to demo_inference.py.
# The script handles:
#   1. Creating the named FIFO pipe
#   2. Starting GNSS-SDR in the background (reading from the FIFO)
#   3. Starting demo_inference.py with --fifo pointing to the pipe
#   4. Cleaning up both processes and the FIFO on Ctrl+C or exit
#
# Prerequisites:
#   - gnss-sdr must be on PATH (built with ENABLE_FMCOMMS2=ON)
#   - demo_inference.py must support the --fifo flag
#   - PlutoSDR must be reachable (default: ip:192.168.2.1)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIFO_PATH="/tmp/gnss_iq_fifo"
GNSS_SDR_CONF="gnss_sdr_fifo.conf"
GNSS_SDR_LOG="gnss_sdr_output.log"
GNSS_SDR_PID=""
CLASSIFIER_PID=""

# ---------------------------------------------------------------------------
# Cleanup handler — runs on EXIT, INT, TERM
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "[launcher] Shutting down..."

    # Kill classifier first (it owns the PlutoSDR and writes to the FIFO)
    if [ -n "$CLASSIFIER_PID" ] && kill -0 "$CLASSIFIER_PID" 2>/dev/null; then
        echo "[launcher] Stopping classifier (PID $CLASSIFIER_PID)..."
        kill -INT "$CLASSIFIER_PID" 2>/dev/null || true
        # Give it a moment to flush CSV and release CUDA context
        sleep 1
        kill -0 "$CLASSIFIER_PID" 2>/dev/null && kill -KILL "$CLASSIFIER_PID" 2>/dev/null || true
    fi

    # Kill GNSS-SDR (it will get EOF on the FIFO once the writer exits)
    if [ -n "$GNSS_SDR_PID" ] && kill -0 "$GNSS_SDR_PID" 2>/dev/null; then
        echo "[launcher] Stopping GNSS-SDR (PID $GNSS_SDR_PID)..."
        kill -INT "$GNSS_SDR_PID" 2>/dev/null || true
        sleep 1
        kill -0 "$GNSS_SDR_PID" 2>/dev/null && kill -KILL "$GNSS_SDR_PID" 2>/dev/null || true
    fi

    # Remove FIFO
    if [ -p "$FIFO_PATH" ]; then
        echo "[launcher] Removing FIFO: $FIFO_PATH"
        rm -f "$FIFO_PATH"
    fi

    echo "[launcher] Done."
}

trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  GNSS Interference Classifier + GNSS-SDR Parallel Launcher"
echo "============================================================"
echo ""

# Check gnss-sdr is available
if ! command -v gnss-sdr &>/dev/null; then
    echo "[FAIL] gnss-sdr not found on PATH."
    echo "       Build it with: cmake .. -DENABLE_FMCOMMS2=ON && make -j3 && sudo make install"
    exit 1
fi

# Check config file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/$GNSS_SDR_CONF" ]; then
    # Also check current directory
    if [ ! -f "$GNSS_SDR_CONF" ]; then
        echo "[FAIL] GNSS-SDR config not found: $GNSS_SDR_CONF"
        echo "       Expected alongside this script or in the current directory."
        exit 1
    fi
    GNSS_SDR_CONF_PATH="$GNSS_SDR_CONF"
else
    GNSS_SDR_CONF_PATH="$SCRIPT_DIR/$GNSS_SDR_CONF"
fi

# Check demo_inference.py
if [ ! -f "$SCRIPT_DIR/demo_inference.py" ] && [ ! -f "demo_inference.py" ]; then
    echo "[FAIL] demo_inference.py not found."
    exit 1
fi
DEMO_SCRIPT="demo_inference.py"
[ -f "$SCRIPT_DIR/demo_inference.py" ] && DEMO_SCRIPT="$SCRIPT_DIR/demo_inference.py"

# ---------------------------------------------------------------------------
# Step 1: Create FIFO
# ---------------------------------------------------------------------------
if [ -p "$FIFO_PATH" ]; then
    echo "[INFO] FIFO already exists: $FIFO_PATH (reusing)"
elif [ -e "$FIFO_PATH" ]; then
    echo "[WARN] $FIFO_PATH exists but is not a FIFO — removing and recreating"
    rm -f "$FIFO_PATH"
    mkfifo "$FIFO_PATH"
    echo "[INFO] Created FIFO: $FIFO_PATH"
else
    mkfifo "$FIFO_PATH"
    echo "[INFO] Created FIFO: $FIFO_PATH"
fi

# ---------------------------------------------------------------------------
# Step 2: Start GNSS-SDR in the background
# ---------------------------------------------------------------------------
# GNSS-SDR will block on opening the FIFO until demo_inference.py starts
# writing. This is expected — it will sit waiting and spring to life once
# IQ data flows.
echo "[INFO] Starting GNSS-SDR (will block until IQ data arrives)..."
echo "       Config: $GNSS_SDR_CONF_PATH"
echo "       Log:    $GNSS_SDR_LOG"

gnss-sdr --config_file="$GNSS_SDR_CONF_PATH" > "$GNSS_SDR_LOG" 2>&1 &
GNSS_SDR_PID=$!
echo "[INFO] GNSS-SDR started (PID $GNSS_SDR_PID)"

# Brief pause to let GNSS-SDR initialize and open the FIFO for reading
sleep 2

# Verify it's still alive (it should be blocking on the FIFO open)
if ! kill -0 "$GNSS_SDR_PID" 2>/dev/null; then
    echo "[FAIL] GNSS-SDR exited immediately. Check $GNSS_SDR_LOG:"
    tail -20 "$GNSS_SDR_LOG"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: Start the classifier with --fifo
# ---------------------------------------------------------------------------
echo "[INFO] Starting classifier..."
echo "       FIFO:   $FIFO_PATH"
echo "       Args:   $@"
echo ""

python3 "$DEMO_SCRIPT" --fifo "$FIFO_PATH" "$@" &
CLASSIFIER_PID=$!
echo "[INFO] Classifier started (PID $CLASSIFIER_PID)"
echo ""
echo "============================================================"
echo "  Both processes running. Press Ctrl+C to stop."
echo "  Classifier output appears below."
echo "  GNSS-SDR output logged to: $GNSS_SDR_LOG"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Wait for either process to exit
# ---------------------------------------------------------------------------
# If the classifier exits (Ctrl+C, error, or --snap mode finishing),
# the cleanup handler will kill GNSS-SDR. If GNSS-SDR crashes, we
# let the classifier continue (it doesn't depend on GNSS-SDR).
wait "$CLASSIFIER_PID" 2>/dev/null || true
echo "[INFO] Classifier exited."

# Check if GNSS-SDR is still running
if kill -0 "$GNSS_SDR_PID" 2>/dev/null; then
    echo "[INFO] GNSS-SDR still running — stopping..."
fi
# cleanup() handles the rest via the EXIT trap
