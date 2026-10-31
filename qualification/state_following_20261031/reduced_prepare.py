"""Reuse audited preparation, restricted to the reduced IDs and isolated metadata."""
from pathlib import Path
import socket,sys,ast,json
assert socket.getfqdn()=='oracle.sns.it'
R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030');D=R/'reduced_validation'
name=sys.argv[1];ids={x['id'] for x in json.loads((R/'reduced_scope.json').read_text())['primary_states']}|{'B039'}
for sub in ['prepared','assignment']:(D/sub).mkdir(parents=True,exist_ok=True)
for filename in ['protocol.json','spectral_protocol.json']:
 link=D/'assignment'/filename
 if not link.exists():link.symlink_to(P/'blind/mouse'/filename)
old=P/'blind/prepared'/name;new=D/'prepared'/name
if old.exists() and not new.exists():new.symlink_to(old,target_is_directory=True)
sys.path.insert(0,str(P));import blind_mouse as original
original.ROOT=D/'assignment'
tree=ast.parse((P/'blind_mouse.py').read_text());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
block=worker.body[-1];assert isinstance(block,ast.With)
cut=next(i for i,n in enumerate(block.body) if isinstance(n,ast.ImportFrom) and n.module=='matrix_gfn2.relative_pair_roks')
tail=block.body[cut-1];assert 'full.run(case)' in ast.unparse(tail)
text=ast.unparse(tail).replace('full.run(case)',"parent, xyz = state_from(case['targets'][0])\ns = parent.reference_state\ncalc = GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)\nctx = prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)\nspectral['calibrate'](case,s,ctx)")
block.body=block.body[:cut-1]+ast.parse(text).body
class Isolate(ast.NodeTransformer):
 def visit_Assign(self,node):
  if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
   name=node.targets[0].id
   if name=='prepared':node.value=ast.parse(repr(str(D/'prepared')),mode='eval').body;node.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[node.value],keywords=[])
   if name=='items':return [node,ast.parse("items=[x for x in items if x['id'] in REDUCED_IDS]").body[0]]
   if name=='case' and "json.loads" in ast.unparse(node.value):return [node,ast.parse("case['targets']=[x for x in case['targets'] if x['id'] in REDUCED_IDS]").body[0]]
  return self.generic_visit(node)
worker=Isolate().visit(worker);ast.fix_missing_locations(worker);original.__dict__['REDUCED_IDS']=ids
exec(compile(ast.Module(body=[worker],type_ignores=[]),str(P/'blind_mouse.py'),'exec'),original.__dict__)
original.worker(name)
