"""Execute a notebook in place with nbclient: uv run python notebooks/run_walkthrough.py notebooks/pipeline_walkthrough.ipynb"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1])
nb = nbformat.read(path, as_version=4)
t = time.time()
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
nbformat.write(nb, path)
print(f"executed in {time.time() - t:.0f} s; size {path.stat().st_size / 1e6:.2f} MB")
