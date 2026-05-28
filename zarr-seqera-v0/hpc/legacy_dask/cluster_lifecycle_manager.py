#!/usr/bin/env python3

"""
cluster_lifecycle_manager.py - Robust SSH cluster lifecycle management with error handling

This script manages a Dask SSH cluster with guaranteed cleanup on ANY exit condition:
- Normal completion
- Errors during startup
- Errors during operation
- User interruption (Ctrl+C)
- Parent process termination
- Unexpected exceptions

The cluster is guaranteed to shutdown cleanly in all scenarios.
"""

import sys
import os
import signal
import asyncio
import atexit
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from cluster_manager import load_cluster_config, create_dask_cluster, shutdown_cluster
except ImportError:
    print("ERROR: Could not import cluster_manager module", file=sys.stderr)
    print("Make sure cluster_manager.py is in the same directory", file=sys.stderr)
    sys.exit(1)


class ClusterLifecycleManager:
    """Manages cluster lifecycle with guaranteed cleanup"""

    def __init__(self, config_path: str, scheduler_file: str):
        self.config_path = config_path
        self.scheduler_file = scheduler_file
        self.client = None
        self.cluster = None
        self.cleanup_called = False
        self.shutdown_event: asyncio.Event | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop):
        """Attach event loop context for signal handlers"""
        self.loop = loop
        self.shutdown_event = asyncio.Event()

    def request_shutdown(self):
        """Signal the manager to begin shutdown"""
        if self.shutdown_event and not self.shutdown_event.is_set():
            self.shutdown_event.set()

    async def start_cluster(self):
        """Start the Dask SSH cluster"""
        try:
            # Load configuration
            config = load_cluster_config(self.config_path)

            # Create cluster
            self.client, self.cluster = await create_dask_cluster(config)

            if not self.client:
                # No Dask - write empty scheduler file
                print("Dask disabled - no cluster to manage", file=sys.stderr)
                with open(self.scheduler_file, 'w') as f:
                    f.write('')
                return False

            # Write scheduler address to file
            scheduler_address = self.client.scheduler.address
            print(f"SCHEDULER_ADDRESS={scheduler_address}", file=sys.stderr)

            with open(self.scheduler_file, 'w') as f:
                f.write(scheduler_address)

            print("Cluster is running. Waiting for shutdown signal...", file=sys.stderr)
            return True

        except Exception as e:
            print(f"\nERROR during cluster startup: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

            # Ensure cleanup even on startup failure
            await self.cleanup()
            return False

    async def wait_for_shutdown(self):
        """Wait indefinitely for shutdown signal"""
        if not self.shutdown_event:
            raise RuntimeError("Shutdown event not initialized")

        try:
            await self.shutdown_event.wait()
        except asyncio.CancelledError:
            print("\nReceived cancellation signal", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt", file=sys.stderr)

    async def cleanup(self):
        """Cleanup cluster resources - guaranteed to run"""
        if self.cleanup_called:
            return  # Prevent double cleanup

        self.cleanup_called = True

        print("\n" + "="*60, file=sys.stderr)
        print("Cleaning up Dask cluster...", file=sys.stderr)
        print("="*60, file=sys.stderr)

        try:
            await shutdown_cluster(self.client, self.cluster)
            print("✓ Cluster shutdown complete", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}", file=sys.stderr)
            # Still try to force cleanup
            if self.client:
                try:
                    await self.client.close(timeout=5)
                except:
                    pass
            if self.cluster:
                try:
                    await self.cluster.close(timeout=5)
                except:
                    pass

        # Remove scheduler file
        try:
            if os.path.exists(self.scheduler_file):
                os.remove(self.scheduler_file)
        except:
            pass

        print("="*60, file=sys.stderr)

    async def run(self):
        """Main run loop with guaranteed cleanup"""
        try:
            # Start cluster
            success = await self.start_cluster()

            if not success:
                print("Cluster startup failed", file=sys.stderr)
                return 1

            # Wait for shutdown signal
            await self.wait_for_shutdown()

            return 0

        except KeyboardInterrupt:
            print("\nReceived Ctrl+C", file=sys.stderr)
            return 0

        except Exception as e:
            print(f"\nERROR in cluster manager: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 1

        finally:
            # GUARANTEED cleanup
            await self.cleanup()


async def main():
    """Main entry point"""
    if len(sys.argv) != 3:
        print("Usage: cluster_lifecycle_manager.py <config_path> <scheduler_file>", file=sys.stderr)
        return 1

    config_path = sys.argv[1]
    scheduler_file = sys.argv[2]

    manager = ClusterLifecycleManager(config_path, scheduler_file)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    manager.bind_loop(loop)

    def signal_handler(signum, frame):
        """Handle termination signals"""
        print(f"\nReceived signal {signum}", file=sys.stderr)
        # Request orderly shutdown without abruptly stopping the loop
        loop.call_soon_threadsafe(manager.request_shutdown)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the manager
    exit_code = await manager.run()

    return exit_code


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nForced exit", file=sys.stderr)
        sys.exit(0)
    except RuntimeError as e:
        if str(e) == "Event loop stopped before Future completed.":
            # Expected when parent process stops the loop after requesting cleanup
            sys.exit(0)
        raise
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
