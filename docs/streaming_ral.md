# Real-time Streaming Pipeline (RAL — attach mode)

The streaming pipeline consumes the live `DaqData.StreamImages` gRPC feed and emits
cloud-detection scores in real time. It runs **alongside** the batch pipeline on the
same Ray cluster (attach mode, externally owned). See [ML Architecture](ml_architecture.md)
for the component diagram, progressive-calibration design, and equivalence keystone tests.

Ray Serve is a **persistent substrate** here — the scoped exception to the
transient-cluster rule. The cluster is owned externally (user ran `ray up`); the
pipeline deploys onto it and tears down its own deployment on exit, but never the
cluster itself.

## Steps

```bash
# 1. Start the panoseti_grpc unified server with DaqData + MLInference enabled:
#    Edit grpc/src/panoseti_grpc/config/server.toml:  services.ml_inference = true
pseti-grpc server                                   # or: pseti-grpc server --config custom.toml

# 2. (Optional) For replay from archived PFF, configure simulate_daq_cfg in server.toml
#    and point movie_pff_path to a file under /mnt/beegfs/data/L0/

# 3. Run the streaming pipeline (gaming GPU on digilab-transmit, 60s cadence):
pa-stream-cloud \
    --model-path assets/models/cloud_detector_v1.pt \
    --recipe recipes/ml/stream_cloud_v1.yml \
    --grpc-host localhost \
    --archive-dir /mnt/beegfs/streams/

# 4. Subscribe to live predictions (from another terminal):
python -c "
from panoseti_grpc.ml_inference.client import MLInferenceClient
with MLInferenceClient() as c:
    for p in c.stream_predictions():
        print(p.module_id, p.cloud_score, p.cloud_label)
"
```

## GPU placement

| Node               | GPU                                      | Role                                            |
| ------------------ | ---------------------------------------- | ----------------------------------------------- |
| `digilab-transmit` | 2× consumer RTX (`accelerator_type:RTX`) | Ray Serve inference replica (serving)           |
| `digilab-receiver` | 2× RTX A6000 (`accelerator_type:G`)      | Training — **reserved, do not use for serving** |

`pa-stream-cloud` pins `CloudInferDeployment` to `digilab-transmit` with
`ray_actor_options={"num_gpus": 1, "resources": {"accelerator_type:G": 0.001}}`.

## Shutdown behaviour

`pa-stream-cloud` shuts down `CloudInferDeployment` on Ctrl-C but does **not** shut
down the Ray cluster. The cluster persists for training jobs.

## Testing with archived PFF (replay)

Set `simulate_daq_cfg` in `server.toml` and point `movie_pff_path` to a `.pffd` under
`/mnt/beegfs/data/L0/`. The gRPC server replays frames through the same `StreamImages`
interface — no code changes required in the streaming pipeline.
