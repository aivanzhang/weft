"""Prepare one authored native scene without rendering or final-delivery claims."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from scripts.blender_finish_generated import finish
args=sys.argv[sys.argv.index('--')+1:]
if len(args)!=2: raise ValueError('Require config and new output')
finish(args[0],args[1],prepare_only=True,preserve_skin_source=True)
