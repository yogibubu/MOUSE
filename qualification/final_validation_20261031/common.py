"""Remote-only, checkpointed helpers for the final frozen validation campaign."""
from pathlib import Path
import os,socket,sys,json,hashlib,ast,time
assert socket.getfqdn()=='oracle.sns.it', 'Electronic work is allowed only on verified Oracle'
V=Path('/home/enzo/MOUSE-orbital-performance-20261031');W=V/'multicsf_integration'
P=Path('/home/enzo/MOUSE-publication-validation-20261030');OUT=W/'final_validation'
OLD=Path('/home/enzo/MOUSE-roks-independent-20261030')
os.environ.update(MOUSE_VALIDATION_ROOT=str(W),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
sys.path.insert(0,str(W/'source/qualification'));import remote_bootstrap
import numpy as np
from dataclasses import replace
from types import SimpleNamespace
from scipy.linalg import eigh,solve_sylvester
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits as raw_limits
from pyscf import scf,gto,dft,ao2mo,lib
from matrix_gfn2 import GFN2Calculator,MultiCSFModel,FrozenNonlocalResponseReference,write_multicsf_model,evaluate_multicsf_surface,save_multicsf_result,load_multicsf_result
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.restricted_ground import RestrictedGroundResult,evaluate_restricted_ground
from matrix_gfn2.multicsf_nuclear import metric_power
sys.path.insert(0,'/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029')
from extent_support import source_molecule
lib.num_threads(2);HOST=socket.getfqdn();EV=27.211386245988
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=sha
key=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 tmp.write_text(json.dumps(obj,indent=2,default=lambda a:a.tolist() if hasattr(a,'tolist') else a.item())+'\n');tmp.replace(p)
def loadpoint(p):
 with np.load(p) as a:x=a['coordinates_angstrom'].copy();request=str(a['request_hash'])
 return load_derivative_point(p,request_hash=request,coordinates_angstrom=x,require_gradient=False)
def extract(path,names,env):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert len(nodes)==len(names)
 env['__file__']=str(path);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
def threadpool_limits(*args,**kwargs):return raw_limits(limits=2)
def calculator(s):return GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
def canonical_state(raw):
 s=raw.reference_state;f=raw.fock[0];e,c=eigh(f,s.overlap)
 # Direct minimization fixes occupied/virtual stationarity, not intrasector
 # canonicalization. Diagonalize the full ground Fock before orbital transport.
 return replace(s,coefficients=c,fock=f,orbital_energies_hartree=e)
def calibrate(folder,reference,s,ctx):
 """Unchanged full AO-pair and qualified-spectral profiles, isolated outputs."""
 froot=folder/'full_response';qroot=folder/'qualified_spectral_response'
 froot.mkdir(parents=True,exist_ok=True);qroot.mkdir(parents=True,exist_ok=True)
 full=SimpleNamespace(ROOT=froot);env=dict(globals(),full=full)
 extract(OLD/'full_response.py',['symmetric_basis'],env);full.symmetric_basis=env['symmetric_basis']
 extract(OLD/'full_fast.py',['features','invsqrt','calibrate'],env)
 case=dict(name='anchor',reference=str(reference));data,request=env['calibrate'](case,s,ctx)
 # The spectral helper consumes the same full response with the original
 # qualified overlap/margin rules and occupied-centroid energy gauge.
 (folder/'results').mkdir(exist_ok=True)
 link=folder/'results/full_response'
 if not link.exists():link.symlink_to(froot.resolve(),target_is_directory=True)
 spectral=folder/'spectral_protocol.json'
 if not spectral.exists():write(spectral,dict(profile='unchanged qualified spectral',lambda_exchange=.9,weights=[.5,.5],offset=0))
 env=dict(globals(),ROOT=qroot,BASE=folder,p=spectral)
 tree=ast.parse((OLD/'qualified_spectral_response.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calibrate')
 class Alias(ast.NodeTransformer):
  def visit_Constant(self,n):return ast.copy_location(ast.Constant('orbital_energies'),n) if n.value=='orbital_energies_hartree' else n
 node=Alias().visit(node);ast.fix_missing_locations(node)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(OLD/'qualified_spectral_response.py'),'exec'),env)
 qdata,qrequest=env['calibrate'](case,s,ctx)
 return [data,qdata],froot/'anchor/projected_eri.npz'
def ground(folder,calc,xyz,s,response=None,seed=None):
 cp=folder/'stationary.npz';cert=folder/'certificate.json'
 if cert.exists():
  row=json.loads(cert.read_text());g=RestrictedGroundResult(loadpoint(cp),row['minimum_curvature'],row['index'])
  assert g.spin_reference_fingerprint==(None if response is None else response.fingerprint);return g
 g=evaluate_restricted_ground(calc,xyz,template=s,spin_reference=response,reference_result=seed,gradient=False)
 folder.mkdir(parents=True,exist_ok=True);save_derivative_point(cp,g.reference,request_hash='final_validation:ground',coordinates_angstrom=xyz)
 write(cert,dict(minimum_curvature=g.minimum_orbital_curvature_hartree,index=g.orbital_saddle_index));return g
