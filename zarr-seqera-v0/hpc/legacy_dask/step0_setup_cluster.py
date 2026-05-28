#!/usr/bin/env python3
"""
Step 0: Setup Dask cluster and write scheduler address to file

This script initializes the Dask cluster and outputs the scheduler address
to a file, which subsequent processing steps can read to connect.

Usage:
    python step0_setup_cluster.py [config.toml] [scheduler_output_file]
    
Outputs:
    - Writes scheduler address to specified file
    - Keeps cluster alive until interrupted
    - Handles cleanup on exit
"""

import sys
import os
import asyncio
import signal
from pathlib import Path

# Import cluster management utilities
from cluster_manager import load_cluster_config, create_cluster, shutdown_cluster


class DaskClusterSetup:
    """Manages persistent Dask cluster for pipeline operations"""
    
    def __init__(self, config_path: str, scheduler_file: str):
        self.config_path = config_path
        self.scheduler_file = scheduler_file
        self.client = None
        self.cluster = None
        self.cleanup_done = False
        self.shutdown_event = None
        
    def bind_event_loop(self, loop):
        """Bind to asyncio event loop for signal handling"""
        self.loop = loop
        self.shutdown_event = asyncio.Event()
        
    def request_shutdown(self):
        """Signal cluster to shutdown gracefully"""
        if self.shutdown_event and not self.shutdown_event.is_set():
            print("\nShutdown requested...", file=sys.stderr)
            self.shutdown_event.set()
            
    async def setup_cluster(self):
        """Initialize Dask cluster and write scheduler address"""
        try:
            print(f"Loading cluster configuration from: {self.config_path}")
            config = load_cluster_config(self.config_path)
            
            # Create the cluster
            print("\nInitializing Dask cluster...")
            self.client, self.cluster = await create_cluster(config)
            
            if not self.client:
                # Dask disabled - write empty file
                print("Dask disabled in config - no cluster created")
                with open(self.scheduler_file, 'w') as f:
                    f.write('')
                return False
                
            # Write scheduler address for other processes to use
            scheduler_address = self.client.scheduler.address
            print(f"\n✓ Cluster ready!")
            print(f"  Scheduler: {scheduler_address}")
            print(f"  Dashboard: {self.client.dashboard_link}")
            
            with open(self.scheduler_file, 'w') as f:
                f.write(scheduler_address)
                
            print(f"  Scheduler address written to: {self.scheduler_file}")
            print("\nCluster is now available for pipeline operations.")
            print("Press Ctrl+C to shutdown cluster.")
            
            return True
            
        except Exception as e:
            print(f"\nERROR: Failed to setup cluster: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            await self.cleanup()
            return False
            
    async def wait_for_shutdown_signal(self):
        """Wait indefinitely until shutdown is requested"""
        try:
            await self.shutdown_event.wait()
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("\nShutdown signal received")
            
    async def cleanup(self):
        """Cleanup cluster resources"""
        if self.cleanup_done:
            return
        self.cleanup_done = True
        
        print("\n" + "="*60)
        print("Shutting down Dask cluster...")
        print("="*60)
        
        try:
            await shutdown_cluster(self.client, self.cluster)
            print("✓ Cluster shutdown complete")
        except Exception as e:
            print(f"Warning during cleanup: {e}")
            
        # Remove scheduler file
        try:
            if os.path.exists(self.scheduler_file):
                os.remove(self.scheduler_file)
        except:
            pass
            
        print("="*60)
        
    async def run(self):
        """Main execution: setup, wait, cleanup"""
        try:
            # Setup cluster
            success = await self.setup_cluster()
            if not success:
                return 1
                
            # Wait for shutdown signal
            await self.wait_for_shutdown_signal()
            return 0
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return 0
        except Exception as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1
        finally:
            # Always cleanup
            await self.cleanup()


async def main():
    """Entry point"""
    if len(sys.argv) < 3:
        print("Usage: step0_setup_cluster.py [config.toml] [scheduler_output_file]")
        print()
        print("Arguments:")
        print("  config.toml           - Configuration file with [cluster] section")
        print("  scheduler_output_file - File to write scheduler address")
        print()
        print("Example:")
        print("  python step0_setup_cluster.py config.toml /tmp/dask_scheduler.txt")
        return 1
        
    config_path = sys.argv[1]
    scheduler_file = sys.argv[2]
    
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return 1
        
    # Setup cluster manager
    manager = DaskClusterSetup(config_path, scheduler_file)
    
    # Setup signal handlers
    loop = asyncio.get_running_loop()
    manager.bind_event_loop(loop)
    
    def signal_handler(signum, frame):
        loop.call_soon_threadsafe(manager.request_shutdown)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run cluster
    return await manager.run()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nForced exit")
        sys.exit(0)

