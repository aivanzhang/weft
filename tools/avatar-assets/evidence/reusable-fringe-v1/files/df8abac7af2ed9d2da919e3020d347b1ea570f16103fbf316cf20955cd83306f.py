"""Exercise only the actual new inventory filter with analytic objects; no Blender launch."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
tree = ast.parse((HERE/'build_component.py').read_text())
main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
nodes = [node for node in main.body if isinstance(node, ast.Assign)
         and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('rigs', 'helpers', 'meshes')]
assert len(nodes) == 3
fragment = compile(ast.Module(body=nodes, type_ignores=[]), str(HERE/'build_component.py'), 'exec')


class Object:
    def __init__(self, name, kind):
        self.name, self.type = name, kind
        self.pose = SimpleNamespace(bones=[])


body, rig, shape = Object('body', 'MESH'), Object('rig', 'ARMATURE'), Object('arbitrary-display-name', 'MESH')
rig.pose.bones = [SimpleNamespace(custom_shape=shape), SimpleNamespace(custom_shape=None)]


def run(objects):
    state = {'bpy': SimpleNamespace(context=SimpleNamespace(scene=SimpleNamespace(objects=objects)))}
    exec(fragment, state)
    return state


valid = run([body, rig, shape])
assert valid['meshes'] == [body] and valid['rigs'] == [rig] and valid['helpers'] == {shape}
unknown = Object('Icosphere', 'MESH')
unknown.hide_render = True
extra = run([body, rig, shape, unknown])
assert extra['meshes'] == [body, unknown], 'An unrelated hidden mesh must not be silently excluded'
shape.name = 'body-looking-name'
assert run([shape, body, rig])['meshes'] == [body], 'Filter must follow identity, not names'
no_rig = run([body, unknown])
assert not no_rig['rigs'] and no_rig['meshes'] == [body, unknown]
for name in ('build_component.py', 'execute.py', 'audit_export.py', 'postprocess_export.py'):
    ast.parse((HERE/name).read_text())
report = {'status': 'four_analytic_inventory_controls_passed', 'native_executed': False,
          'actual_build_script_sha256': hashlib.sha256((HERE/'build_component.py').read_bytes()).hexdigest(),
          'checks': ['Exact custom_shape identity excluded', 'Unrelated hidden/name-matching mesh retained for rejection',
                     'Renamed helper still excluded', 'Missing rig never invents a helper exclusion'],
          'scope': 'Actual new filter AST only; actual scene inventory remains unverified until the separate released native attempt.'}
with (HERE/'recipe-check.json').open('x') as stream:
    json.dump(report, stream, indent=2, sort_keys=True); stream.write('\n')
print(json.dumps(report))
