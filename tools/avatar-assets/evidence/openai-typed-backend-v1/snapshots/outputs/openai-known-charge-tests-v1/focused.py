import sys,unittest
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root),str(root/'tests')]
suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_openai_runner','tests.test_openai_studio_integration'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
