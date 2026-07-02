# Hardware-in-the-Loop (HITL) Remote Observatory Streaming

This document describes how to route a real-time data stream from a remote, GPU-less observatory to the UC Berkeley Radio Astro Lab (RAL) cluster for real-time Machine Learning inference.

Because the ML logic (Ray Serve) is strictly decoupled from the DAQ ingestion logic via gRPC, the remote observatory is completely unaware of the downstream ML frameworks.

## Prerequisites

- **Remote Observatory (No GPUs)**: 4 telescopes running Hashpipe.
- **RAL Cluster (88-cores, 2x A6000 GPUs)**: The `digilab-transmit` and `digilab-receiver` nodes.

## Step 1: Run the gRPC broker on the remote observatory

On the remote observatory's head node, run the lightweight `panoseti_grpc` broker. This reads the Hashpipe shared memory and broadcasts it over gRPC (port 50051). It requires no ML dependencies and no GPUs.

```bash
# On the remote observatory
pseti-grpc server --profile daq_node
```

## Step 2: Port-forward to the RAL cluster

To securely feed the data to RAL over the internet, set up an SSH tunnel that forwards port `50051` from the remote observatory to your RAL head node.

```bash
# On the RAL head node
ssh -N -L 50051:localhost:50051 user@<remote-observatory-ip>
```

## Step 3: Spin up the real-time inference pipeline on RAL

On the RAL cluster, ensure the persistent Ray cluster is running (via `cluster/ral_up.sh`). Then, start the streaming pipeline. It will automatically connect to the forwarded port, spin up the `CloudInferDeployment` Ray Serve actors on the A6000 GPUs, and start routing the frames.

```bash
# On the RAL head node
pa-stream-cloud \
    --model-path assets/models/cloud_detector_v1.pt \
    --grpc-host localhost
```

**Architecture note:** The `StreamConsumer` will automatically multiplex the incoming frames into 4 separate `FrameAccumulator` Ray actors (one for each telescope module). They maintain a rolling-median calibration buffer and emit inference requests to the GPU asynchronously.

## Step 4: View real-time inferences

The `CloudInferDeployment` actors emit their predictions back to the gRPC broker. You can view these live scores as they arrive.

### Option A: Python CLI

Subscribe to the live prediction stream via a quick Python script:

```python
from panoseti_grpc.ml_inference.client import MLInferenceClient

with MLInferenceClient(host="localhost") as c:
    for p in c.stream_predictions():
        print(f"Module: {p.module_id} | Score: {p.cloud_score:.4f}")
```

### Option B: Grafana Dashboard

Because the `FrameAccumulator` emits telemetry (`log_flexible`), you can open your local RAL Grafana instance and visualize the per-telescope cloud scores graphing in real time as the data streams in from the remote observatory.
