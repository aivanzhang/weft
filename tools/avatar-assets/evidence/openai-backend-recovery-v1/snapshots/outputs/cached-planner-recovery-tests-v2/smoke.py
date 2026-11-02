import sys,unittest
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root),str(root/'tests')]
unittest.main(module='tests.test_pipeline',verbosity=2)
