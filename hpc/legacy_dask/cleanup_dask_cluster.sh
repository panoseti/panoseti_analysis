#!/bin/bash

# cleanup_dask_cluster.sh - Manual cleanup utility for orphaned Dask processes
#
# This script can be run manually to clean up any orphaned Dask scheduler/worker
# processes that may be left running if the pipeline crashes unexpectedly.

set -e

echo "================================================"
echo "Dask Cluster Cleanup Utility"
echo "================================================"
echo ""

# Check for running Dask processes
echo "Checking for Dask processes..."
echo ""

SCHEDULER_PIDS=$(pgrep -f "distributed.cli.dask_scheduler" 2>/dev/null || true)
WORKER_PIDS=$(pgrep -f "distributed.cli.dask_worker" 2>/dev/null || true)
NANNY_PIDS=$(pgrep -f "distributed.nanny" 2>/dev/null || true)

if [ -z "$SCHEDULER_PIDS" ] && [ -z "$WORKER_PIDS" ] && [ -z "$NANNY_PIDS" ]; then
    echo "✓ No Dask processes found"
    echo ""
    exit 0
fi

# Show what will be killed
if [ ! -z "$SCHEDULER_PIDS" ]; then
    echo "Found Dask scheduler processes:"
    ps -fp $SCHEDULER_PIDS
    echo ""
fi

if [ ! -z "$WORKER_PIDS" ]; then
    echo "Found Dask worker processes:"
    ps -fp $WORKER_PIDS
    echo ""
fi

if [ ! -z "$NANNY_PIDS" ]; then
    echo "Found Dask nanny processes:"
    ps -fp $NANNY_PIDS
    echo ""
fi

# Confirm before killing
echo "================================================"
read -p "Kill these processes? [y/N] " -n 1 -r
echo ""
echo "================================================"

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled by user"
    exit 0
fi

# Kill processes
echo ""
echo "Terminating Dask processes..."

if [ ! -z "$SCHEDULER_PIDS" ]; then
    echo "  Killing schedulers..."
    kill -TERM $SCHEDULER_PIDS 2>/dev/null || true
    sleep 2
    # Force kill if still running
    for pid in $SCHEDULER_PIDS; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid 2>/dev/null || true
        fi
    done
fi

if [ ! -z "$WORKER_PIDS" ]; then
    echo "  Killing workers..."
    kill -TERM $WORKER_PIDS 2>/dev/null || true
    sleep 2
    # Force kill if still running
    for pid in $WORKER_PIDS; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid 2>/dev/null || true
        fi
    done
fi

if [ ! -z "$NANNY_PIDS" ]; then
    echo "  Killing nannies..."
    kill -TERM $NANNY_PIDS 2>/dev/null || true
    sleep 2
    # Force kill if still running
    for pid in $NANNY_PIDS; do
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid 2>/dev/null || true
        fi
    done
fi

# Clean up any temp files
echo "  Cleaning up temp files..."
rm -f /tmp/dask_scheduler_*.txt 2>/dev/null || true

# Verify cleanup
sleep 1
REMAINING=$(pgrep -f "distributed.cli" 2>/dev/null || true)

if [ -z "$REMAINING" ]; then
    echo ""
    echo "✓ All Dask processes cleaned up successfully"
else
    echo ""
    echo "Warning: Some processes may still be running:"
    ps -fp $REMAINING
fi

echo ""
echo "================================================"
echo "Cleanup complete"
echo "================================================"
