"""
panoseti_analysis.adapters.stream — Real-time ML streaming pipeline.

This package bridges the panoseti_grpc DaqData.StreamImages feed to the
Layer A inference kernels via Ray actors and Ray Serve.

Components:
  consumer.py       — async DaqData client wrapper; routes frames to accumulators
  accumulator.py    — @ray.remote FrameAccumulator; progressive calibration + window emission
  serve_app.py      — @serve.deployment CloudInferDeployment; loads model once, infers per window
  cli.py            — pa-stream-cloud typer CLI; ties everything together in attach mode

Layer A kernels (algorithms/) are NOT imported here — they live in the
accumulator and serve_app.  This __init__ is import-side-effect-free.
"""
