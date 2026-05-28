# worker_preload.py - Set umask for Dask workers
"""
This script runs when each Dask worker starts up.
It sets the umask to 0o000 to allow full file permissions.
"""
import os

# Set umask to allow all permissions for newly created files/directories
original_umask = os.umask(0o000)
print(f"Worker preload: Set umask from {oct(original_umask)} to 0o000")
