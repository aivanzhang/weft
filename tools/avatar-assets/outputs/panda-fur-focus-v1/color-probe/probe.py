import bpy,json,sys
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1])
i=bpy.data.images.load(str(root/'gray.png'),check_existing=False)
(root/'result.json').write_text(json.dumps({'loaded_pixel':list(i.pixels)[:4],'colorspace':i.colorspace_settings.name,'blender':bpy.app.version_string},indent=2)+'\n')
