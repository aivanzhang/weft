import json
from importlib.resources import files
from types import MappingProxyType


# Load once; render loops consume the same palette without filesystem work.
COLORS = MappingProxyType(json.loads(files("weft").joinpath("assets/palette.json").read_text())["colors"])
