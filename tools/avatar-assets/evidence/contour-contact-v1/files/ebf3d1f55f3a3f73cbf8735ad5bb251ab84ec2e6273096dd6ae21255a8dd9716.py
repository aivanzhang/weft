from pathlib import Path
import runpy
root=Path(__file__).resolve().parent
runpy.run_path(str(root/'check.py'),run_name='__main__')
runpy.run_path(str(root/'check_identity.py'),run_name='__main__')
