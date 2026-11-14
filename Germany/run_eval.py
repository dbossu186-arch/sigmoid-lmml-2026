"""Run data/evaluation_script.py under `python -I` (isolated mode drops the
script directory from sys.path, so `from harness import ...` needs this)."""
import os, runpy, sys
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
sys.path.insert(0, DATA)
sys.argv = [os.path.join(DATA, "evaluation_script.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
