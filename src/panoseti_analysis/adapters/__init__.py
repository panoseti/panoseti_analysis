"""Layer B — thin Typer CLIs invoked by Nextflow processes.

Each adapter reads paths/params from argv, opens inputs via ``panoseti_analysis.io``,
calls exactly one Layer A kernel, writes outputs via ``io``, and emits a
``*.lineage.json`` artifact. No science logic lives here.
"""
