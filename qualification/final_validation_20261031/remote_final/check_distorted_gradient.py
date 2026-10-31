"""Independent total-energy derivative check after symmetry-breaking descent."""
from common import *
from science_slots import science_locks
import argparse
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('sid');a=p.parse_args();locks=science_locks()
with threadpool_limits():
 model=MultiCSFModel(W/f'models/{a.name}/{a.sid}/model.json');calc=calculator(model.anchor);root=OUT/f'geometry_escape/{a.name}/stage_0';point=load_multicsf_result(root/json.loads((root/'accepted.json').read_text())['path']/'result.json',model,calc);point=replace(point,state_following='local_csf');xyz=point.reference_state.coordinates_bohr;g=point.gradient_hartree_per_bohr.ravel();d=g/np.linalg.norm(g);eps=1e-4;values=[]
 for sign in [-1,1]:
  trial=evaluate_multicsf_surface(calc,(xyz+sign*eps*d.reshape(-1,3))/ANGSTROM_TO_BOHR,model,reference_result=point,follow_previous_csf=True);values.append(trial.energy_hartree)
 numerical=(values[1]-values[0])/(2*eps);analytic=float(g@d);error=abs(numerical-analytic)
 write(root.parent/'distorted_gradient_check.json',dict(status='passed' if error<2e-6 else 'failed',analytic=analytic,numerical=numerical,error=error,step_bohr=eps,geometry_bohr=xyz,model_fingerprint=model.fingerprint,source_sha256=sha(__file__)))
 print(a.name,'DISTORTED GRADIENT',analytic,numerical,error,flush=True);assert error<2e-6
