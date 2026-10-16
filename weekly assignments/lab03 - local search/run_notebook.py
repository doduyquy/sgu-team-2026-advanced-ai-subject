import sys
import os
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import time

def execute_notebook(nb_path, timeout=1800):
    nb_path = Path(nb_path).resolve()
    print(f"\n========================================================")
    print(f"Executing: {nb_path.name}")
    print(f"Working dir: {nb_path.parent}")
    print(f"Timeout: {timeout} seconds")
    print(f"========================================================")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    start_time = time.time()

    try:
        ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
        elapsed = time.time() - start_time
        print(f"SUCCESS: {nb_path.name} executed in {elapsed:.2f} seconds.")

        # Save executed notebook with all outputs
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"Saved executed notebook to {nb_path.name}")
        return True, elapsed, None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"ERROR in {nb_path.name} after {elapsed:.2f} seconds: {e}")
        return False, elapsed, str(e)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
        success, elapsed, err = execute_notebook(target, timeout=timeout)
        if not success:
            sys.exit(1)
    else:
        print("Usage: python run_notebook.py <path_to_notebook.ipynb> [timeout]")
