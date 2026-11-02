"""Execute a notebook in place with nbclient.
    uv run python notebooks/run_walkthrough.py notebooks/pipeline_walkthrough.ipynb          # 600 s per cell
    uv run python notebooks/run_walkthrough.py notebooks/demo_inference.ipynb 0            # no cell time limit (~45 min on GPU)
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1])
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 600
nb = nbformat.read(path, as_version=4)
t = time.time()
NotebookClient(nb, timeout=timeout if timeout > 0 else None, kernel_name="python3",
               resources={"metadata": {"path": str(path.parent)}}).execute()
nbformat.write(nb, path)
print(f"executed in {time.time() - t:.0f} s; size {path.stat().st_size / 1e6:.2f} MB")
