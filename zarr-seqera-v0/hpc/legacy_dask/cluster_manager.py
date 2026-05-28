#!/usr/bin/env python3

"""
cluster_manager.py - Extensible Dask cluster management with plugin architecture

This module provides a factory-based system for creating different types of Dask clusters.
New cluster types can be added by implementing the ClusterBackend interface.
"""

import os
import asyncio
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from dask.distributed import Client

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


class ClusterBackend(ABC):
    """Abstract base class for cluster backends"""

    @abstractmethod
    async def create_cluster(self, config: Dict[str, Any]) -> Tuple[Optional[Client], Optional[Any]]:
        """
        Create a cluster and return (client, cluster) tuple.

        Returns:
            Tuple of (Client, Cluster object) or (None, None) if disabled
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the name of this backend"""
        pass


class SSHClusterBackend(ClusterBackend):
    """Backend for SSH-based distributed clusters"""

    def get_backend_name(self) -> str:
        return "ssh"

    async def create_cluster(self, config: Dict[str, Any]) -> Tuple[Optional[Client], Optional[Any]]:
        from dask.distributed import SSHCluster

        hosts = config.get('hosts', ['localhost'])
        if isinstance(hosts, str):
            hosts = [hosts]
        if not hosts:
            raise ValueError("ssh.hosts must contain at least one entry")

        workers_per_host = max(int(config.get('workers_per_host', 1)), 1)
        threads_per_worker = config.get('threads_per_worker', 16)
        memory_per_worker = config.get('memory_per_worker', '16GB')
        scheduler_port = config.get('scheduler_port', 0)
        dashboard_port = config.get('dashboard_port', 8797)
        connect_timeout = config.get('connect_timeout', 60)

        scheduler_host = hosts[0]
        configured_worker_hosts = hosts[1:] if len(hosts) > 1 else []

        if not configured_worker_hosts:
            configured_worker_hosts = [scheduler_host]

        worker_hosts = []
        for host in configured_worker_hosts:
            worker_hosts.extend([host] * workers_per_host)

        hosts_for_cluster = [scheduler_host] + worker_hosts
        expected_workers = len(worker_hosts)

        print(f"\nCreating Dask SSH Cluster:")
        print(f"  Scheduler host: {scheduler_host}")
        print(f"  Worker hosts: {configured_worker_hosts}")
        print(f"  Workers per host: {workers_per_host}")
        print(f"  Threads per worker: {threads_per_worker}")
        print(f"  Memory per worker: {memory_per_worker}")

        preload_command = "import os; os.umask(0o000)"

        cluster = await SSHCluster(
            hosts=hosts_for_cluster,
            connect_options={"known_hosts": None},
            worker_options={
                "nthreads": threads_per_worker,
                "preload": [preload_command],
                "local_directory": config.get('local_directory', os.environ.get('SCRATCH_DIR')),
            },
            scheduler_options={
                "port": scheduler_port,
                "dashboard_address": f":{dashboard_port}",
                "preload": [preload_command],
            },
            asynchronous=True
        )

        client = await Client(cluster, asynchronous=True)

        # Wait for workers
        print(f"\n  Waiting for all {expected_workers} workers to connect...")
        start_time = time.time()
        while time.time() - start_time < connect_timeout:
            info = client.scheduler_info()
            num_workers = len(info['workers'])
            if num_workers >= expected_workers:
                break
            print(f"  {num_workers}/{expected_workers} workers connected...", flush=True)
            await asyncio.sleep(2)

        info = client.scheduler_info()
        num_workers = len(info['workers'])
        total_cores = sum(w['nthreads'] for w in info['workers'].values())

        print(f"\n✓ Dask cluster ready!")
        print(f"  Dashboard: {client.dashboard_link}")
        print(f"  Scheduler: {client.scheduler.address}")
        print(f"  Workers: {num_workers}/{expected_workers}")
        print(f"  Total cores: {total_cores}")

        if num_workers < expected_workers:
            print(f"  ⚠ Warning: Only {num_workers}/{expected_workers} workers connected")
        print()

        return client, cluster


class LocalClusterBackend(ClusterBackend):
    """Backend for local multi-process clusters"""

    def get_backend_name(self) -> str:
        return "local"

    async def create_cluster(self, config: Dict[str, Any]) -> Tuple[Optional[Client], Optional[Any]]:
        from dask.distributed import LocalCluster

        n_workers = config.get('n_workers', 4)
        threads_per_worker = config.get('threads_per_worker', 2)
        memory_per_worker = config.get('memory_per_worker', '4GB')
        dashboard_port = config.get('dashboard_port', 8787)

        print(f"\nCreating Dask Local Cluster:")
        print(f"  Workers: {n_workers}")
        print(f"  Threads per worker: {threads_per_worker}")
        print(f"  Memory per worker: {memory_per_worker}")

        cluster = await LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_per_worker,
            dashboard_address=f":{dashboard_port}",
            asynchronous=True
        )

        client = await Client(cluster, asynchronous=True)

        info = client.scheduler_info()
        num_workers = len(info['workers'])
        total_cores = sum(w['nthreads'] for w in info['workers'].values())

        print(f"\n✓ Dask cluster ready!")
        print(f"  Dashboard: {client.dashboard_link}")
        print(f"  Workers: {num_workers}")
        print(f"  Total cores: {total_cores}")
        print()

        return client, cluster



# class SLURMClusterBackend(ClusterBackend):
#     """Backend for SLURM-managed clusters"""
#
#     def get_backend_name(self) -> str:
#         return "slurm"
#
#     async def create_cluster(self, config: Dict[str, Any]) -> Tuple[Optional[Client], Optional[Any]]:
#         try:
#             from dask_jobqueue import SLURMCluster
#         except ImportError:
#             raise ImportError(
#                 "dask_jobqueue is required for SLURM clusters. "
#                 "Install with: pip install dask-jobqueue"
#             )
#
#         queue = config.get('queue', 'normal')
#         account = config.get('account')
#         cores = config.get('cores', 16)
#         memory = config.get('memory', '32GB')
#         walltime = config.get('walltime', '04:00:00')
#         n_workers = config.get('n_workers', 4)
#         job_extra_directives = config.get('job_extra_directives', [])
#
#         print(f"\nCreating Dask SLURM Cluster:")
#         print(f"  Queue: {queue}")
#         print(f"  Cores per job: {cores}")
#         print(f"  Memory per job: {memory}")
#         print(f"  Walltime: {walltime}")
#
#         cluster = SLURMCluster(
#             queue=queue,
#             account=account,
#             cores=cores,
#             memory=memory,
#             walltime=walltime,
#             job_extra_directives=job_extra_directives,
#             asynchronous=True
#         )
#
#         cluster.scale(n_workers)
#         client = await Client(cluster, asynchronous=True)
#
#         print(f"\n✓ SLURM cluster created!")
#         print(f"  Dashboard: {client.dashboard_link}")
#         print()
#
#         return client, cluster
#
#
# class KubernetesClusterBackend(ClusterBackend):
#     """Backend for Kubernetes-based clusters"""
#
#     def get_backend_name(self) -> str:
#         return "kubernetes"
#
#     async def create_cluster(self, config: Dict[str, Any]) -> Tuple[Optional[Client], Optional[Any]]:
#         try:
#             from dask_kubernetes import KubeCluster
#         except ImportError:
#             raise ImportError(
#                 "dask_kubernetes is required for Kubernetes clusters. "
#                 "Install with: pip install dask-kubernetes"
#             )
#
#         namespace = config.get('namespace', 'dask')
#         image = config.get('image', 'daskdev/dask:latest')
#         n_workers = config.get('n_workers', 4)
#         memory_limit = config.get('memory_limit', '8GB')
#         memory_request = config.get('memory_request', '4GB')
#         cpu_limit = config.get('cpu_limit', 2.0)
#         cpu_request = config.get('cpu_request', 1.0)
#
#         print(f"\nCreating Dask Kubernetes Cluster:")
#         print(f"  Namespace: {namespace}")
#         print(f"  Image: {image}")
#
#         cluster = await KubeCluster(
#             namespace=namespace,
#             image=image,
#             memory_limit=memory_limit,
#             memory_request=memory_request,
#             cpu_limit=cpu_limit,
#             cpu_request=cpu_request,
#             asynchronous=True
#         )
#
#         cluster.scale(n_workers)
#         client = await Client(cluster, asynchronous=True)
#
#         print(f"\n✓ Kubernetes cluster created!")
#         print(f"  Dashboard: {client.dashboard_link}")
#         print()
#
#         return client, cluster


class ClusterFactory:
    """Factory for creating cluster backends"""

    _backends: Dict[str, ClusterBackend] = {}

    @classmethod
    def register_backend(cls, backend: ClusterBackend):
        """Register a new cluster backend"""
        name = backend.get_backend_name()
        cls._backends[name] = backend

    @classmethod
    def get_backend(cls, cluster_type: str) -> ClusterBackend:
        """Get a backend by type name"""
        if cluster_type not in cls._backends:
            available = ', '.join(cls._backends.keys())
            raise ValueError(
                f"Unknown cluster type: {cluster_type}. "
                f"Available types: {available}"
            )
        return cls._backends[cluster_type]

    @classmethod
    def list_backends(cls) -> list:
        """List all registered backend names"""
        return list(cls._backends.keys())


# Register built-in backends
ClusterFactory.register_backend(SSHClusterBackend())
ClusterFactory.register_backend(LocalClusterBackend())
# ClusterFactory.register_backend(SLURMClusterBackend())
# ClusterFactory.register_backend(KubernetesClusterBackend())


def load_cluster_config(config_path: str = "config.toml") -> dict:
    """Load cluster configuration from TOML file"""
    config = {
        'use_cluster': False,
        'use_dask': False,  # Legacy support
        'type': 'local',
    }

    if config_path and os.path.exists(config_path):
        if tomllib is None:
            print(f"Warning: Cannot read {config_path} - tomli/tomllib not installed")
            return config

        try:
            with open(config_path, 'rb') as f:
                toml_config = tomllib.load(f)

            if 'cluster' in toml_config:
                cluster_config = toml_config['cluster']
                if 'use_cluster' in cluster_config:
                    config['use_cluster'] = cluster_config['use_cluster']
                    config['use_dask'] = cluster_config['use_cluster']
                elif 'use_dask' in cluster_config:
                    config['use_dask'] = cluster_config['use_dask']
                    config['use_cluster'] = cluster_config['use_dask']
                config['type'] = cluster_config.get('type', 'local')

                # Load type-specific configuration
                cluster_type = config['type']
                if cluster_type in cluster_config:
                    config['backend_config'] = cluster_config[cluster_type]
                else:
                    config['backend_config'] = {}

            print(f"Loaded cluster configuration from {config_path}")
            print(f"  Cluster type: {config['type']}")
            print(f"  Cluster enabled: {config['use_cluster']}")
        except Exception as e:
            print(f"Warning: Could not read config file: {e}")

    return config


async def create_cluster(config: dict) -> Tuple[Optional[Client], Optional[Any]]:
    """
    Create a cluster based on configuration.

    Returns:
        Tuple of (Client, Cluster) or (None, None) if clustering is disabled
    """
    use_cluster = config.get('use_cluster', False)

    if not use_cluster:
        print("Cluster disabled - operations will use local processing")
        return None, None

    cluster_type = config.get('type', 'local')
    backend_config = config.get('backend_config', {})

    backend = ClusterFactory.get_backend(cluster_type)
    return await backend.create_cluster(backend_config)


async def connect_to_cluster(scheduler_address: str) -> Client:
    """Connect to an existing Dask scheduler"""
    print(f"Connecting to existing Dask scheduler: {scheduler_address}")
    client = await Client(scheduler_address, asynchronous=True)
    info = client.scheduler_info()
    print(f"  ✓ Connected! Workers: {len(info['workers'])}")
    return client


async def shutdown_cluster(client: Optional[Client], cluster: Optional[Any]):
    """Gracefully shutdown Dask cluster"""
    if client:
        print("\nShutting down Dask client...")
        await client.close()

    if cluster:
        print("Shutting down Dask cluster...")
        await cluster.close()

    if client or cluster:
        print("✓ Cluster shutdown complete")


def get_scheduler_address_from_client(client: Optional[Client]) -> str:
    """Extract scheduler address from client"""
    if client:
        return client.scheduler.address
    return ""
