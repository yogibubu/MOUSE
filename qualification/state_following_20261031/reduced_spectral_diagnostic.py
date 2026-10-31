"""Oracle diagnostic: test the exact Q operator beyond its canonical-order precheck.
No reference excitation energies, parameter changes or production gate changes.
"""
from pathlib import Path
import socket,os,sys,json,hashlib,ast
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MOUSE_VALIDATION_ROOT='/home/enzo/MOUSE-state-following-20261031')
R=Path('/home/enzo/MOUSE-state-following-20261031');BASE=P=Path('/home/enzo/MOUSE-publication-validation-20261030');ROOT=R/'reduced_validation/rejected_spectral';p=P/'blind/mouse/spectral_protocol.json'
sys.path.insert(0,str(R/'source/qualification'));import remote_bootstrap
import numpy as np
from scipy.linalg import solve_sylvester
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
HOST=socket.getfqdn()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def write(path,v):
 path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item())+'\n');t.replace(path)
class Diagnostic(ast.NodeTransformer):
 def visit_Constant(self,node):
  if node.value=='orbital_energies_hartree':return ast.copy_location(ast.Constant(value='orbital_energies'),node)
  return node
 def visit_If(self,node):
  if any(isinstance(n,ast.Constant) and n.value=='spectral transfer loses the native occupied/virtual gap' for n in ast.walk(node)):
   return ast.copy_location(ast.parse("write(out/'canonical_ordering.json',dict(gap_hartree=float(np.min(levels[s.alpha_occupations==0])-np.max(levels[occupied])),levels=levels,native_levels=s.orbital_energies_hartree,diagnostic_only=True,model_parameters_unchanged=True))").body[0],node)
  return self.generic_visit(node)
ROOT.mkdir(parents=True,exist_ok=True)
name=sys.argv[1];case=json.loads((R/'reduced_validation/prepared'/name/'case.json').read_text());cp=R/'reduced_validation/prepared'/name/'ground.npz'
with np.load(cp) as f:xyz=f['coordinates_angstrom'].copy();request=str(f['request_hash'])
s=load_derivative_point(cp,request_hash=request,coordinates_angstrom=xyz,require_gradient=False).reference_state
calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
with threadpool_limits(limits=2):
 ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)
 source=Path('/home/enzo/MOUSE-roks-independent-20261030/qualified_spectral_response.py');tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calibrate');module=Diagnostic().visit(ast.Module(body=[node],type_ignores=[]));ast.fix_missing_locations(module);exec(compile(module,str(source),'exec'))
 calibrate(case,s,ctx)
 print((ROOT/name/'canonical_ordering.json').read_text(),flush=True)
