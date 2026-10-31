"""Oracle integration checks for rank and followed-state restart contracts."""
from common import *
import pytest

def test_explicit_rank_is_preserved_in_portable_model(tmp_path):
 original=MultiCSFModel(W/'models/Furan/B022/model.json')
 opts=dict(template=original.template,responses=original.responses,eri=original.arrays['eri'],initial_coefficients=original.arrays['initial_coefficients'],reference_coefficients=original.arrays['reference_coefficients'],reference_transitions=original.arrays['reference_transitions'],transition_weights=np.array([.9,.1]),grounds=original.grounds)
 default=MultiCSFModel(write_multicsf_model(tmp_path/'default',**opts));expanded=MultiCSFModel(write_multicsf_model(tmp_path/'expanded',active_rank=2,**opts))
 assert default.rank==1 and expanded.rank==2 and expanded.manifest['rank_policy']=='explicit_fixed_rank'
 calc=calculator(original.anchor);old,oc=original.objectives(calc,original.xyz);new,nc=expanded.objectives(calc,expanded.xyz)
 for a,c,b,d in zip(old,oc,new,nc):
  ea,ga,_=a.evaluate(c,False);eb,gb,_=b.evaluate(d,False);assert abs(ea-eb)<1e-12 and np.max(abs(ga-gb))<1e-12
 for bad in [True,2.,0,5]:
  with pytest.raises(ValueError,match='explicit active rank'):write_multicsf_model(tmp_path/f'bad_{bad}',active_rank=bad,**opts)

def test_local_csf_continuation_and_restart_keep_energy_derivatives(tmp_path):
 model=MultiCSFModel(W/'models/Furan/B022/model.json');calc=calculator(model.anchor);root=OUT/'geometry/Furan';base=load_multicsf_result(root/'analytic_stationary/result.json',model,calc)
 with np.load(root/'mode_0_1.npz') as a:xyz=a['coordinates'].copy();expected_g=a['gradient'].copy();expected_e=float(a['energy'])
 point=evaluate_multicsf_surface(calc,xyz,model,reference_result=replace(base,state_following='local_csf'),follow_previous_csf=True)
 assert abs(point.energy_hartree-expected_e)<1e-10 and np.max(abs(point.gradient_hartree_per_bohr-expected_g))<2e-7
 file=save_multicsf_result(tmp_path/'accepted',point,model);restored=load_multicsf_result(file,model,calc)
 assert restored.state_following=='local_csf' and abs(restored.energy_hartree-point.energy_hartree)<1e-12
 assert np.max(abs(restored.ci_coefficients-point.ci_coefficients))<1e-9
 with pytest.raises(ValueError,match='state-following policy'):evaluate_multicsf_surface(calc,xyz,model,reference_result=restored)
