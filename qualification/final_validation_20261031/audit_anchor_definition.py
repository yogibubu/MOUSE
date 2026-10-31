"""Algebraic baseline audit using archived operators; no SCF/DFT job."""
from common import *

def main():
 rows=[];D=Path('/home/enzo/MOUSE-state-following-20261031/reduced_validation')
 for name,sid in [('Furan','B022'),('Cyclopentadiene','B016'),('Phthalimide','B024')]:
  model=MultiCSFModel(W/f'models/{name}/{sid}/model.json');s=model.anchor;g=model.grounds[0];density=(s.coefficients*s.alpha_occupations)@s.coefficients.T;error=float(np.max(abs(density-g.spin_densities_ao[0])))
  out=OUT/'baseline_definition';froot=out/'results/full_response';froot.mkdir(parents=True,exist_ok=True);link=froot/name
  if not link.exists():link.symlink_to(P/'results/full_response'/name,target_is_directory=True)
  root=out/'qualified';root.mkdir(exist_ok=True);protocol=out/'spectral_protocol.json'
  if not protocol.exists():write(protocol,dict(profile='algebraic reconstruction of unchanged Q definition'))
  env=dict(globals(),ROOT=root,BASE=out,p=protocol)
  tree=ast.parse((OLD/'qualified_spectral_response.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calibrate')
  class Alias(ast.NodeTransformer):
   def visit_Constant(self,n):return ast.copy_location(ast.Constant('orbital_energies'),n) if n.value=='orbital_energies_hartree' else n
  node=Alias().visit(node);ast.fix_missing_locations(node);exec(compile(ast.Module(body=[node],type_ignores=[]),str(OLD/'qualified_spectral_response.py'),'exec'),env)
  data,_=env['calibrate'](dict(name=name,reference=str(P/'blind/mouse_reference'/name)),s,None)
  with np.load(D/'models'/name/'model.npz') as a:differences={k:float(np.max(abs(data[k]-a[k]))) for k in ['operators','charge','spin','q0','linear']}
  assert error<1e-9 and max(differences.values())<1e-9
  rows.append(dict(molecule=name,native_reference_density_error=error,reconstructed_Q_operator_errors=differences))
 write(OUT/'baseline_definition_audit.json',dict(status='passed',rows=rows,scope='native calibration density equals direct restricted S0, and exact same Q prescription reconstructs original baseline operators; no new SCF, ERI or quadrature calculation',source_sha256=sha(__file__)))
 print('BASELINE DEFINITION PASS',rows,flush=True)
if __name__=='__main__':
 with threadpool_limits():main()
