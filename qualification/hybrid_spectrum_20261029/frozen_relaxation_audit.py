"""Independent cached-energy identities and spin/charge checks; no electronic reruns."""
from remote_bootstrap import HOST
from pathlib import Path
import hashlib,json,numpy as np
from matrix_gfn2.deltascf_derivatives import load_derivative_point
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'frozen_relaxation';EV=27.211386245988
summary=json.loads((OUT/'summary.json').read_text());plan=json.loads((OUT/'plan.json').read_text())
assert not summary['case_failures'] and summary['qualified_states']==10
assert summary['grid_validation']['passed']
checks=[];spins=[]
metadata_paths={
 'uracil':ROOT.parent/'uracil_uridine_20261029/inputs/QUEST_Uracil.json',
 'thymine':Path('/home/enzo/MATRIX-runs/tda-frozen-validation-20261022-v1/docs/reports/tda_frozen_validation_20261022/thymine/inputs/Thymine.json'),
 **{name:ROOT/'generalization/inputs'/file for name,file in [('nitroxyl','Nitroxyl.json'),('nitrous_acid','Nitrous_acid.json'),('propynal','Propynal.json'),('pyridine','Pyridine.json'),('cytosine','Cytosine.json')]}}
for row in summary['rows']:
 if not row['qualified']:continue
 name=row['molecule'];state=row['id'];folder=OUT/name
 metadata=json.loads(metadata_paths[name].read_text());family='npi' if state=='n_pi' else 'ppi'
 reference=next(x for x in metadata if x.get('Spin')==1 and x.get('V/R')=='V' and x.get('Type')==family)
 assert abs(reference['TBE/AVTZ']-row['tbe_ev'])<1e-9
 se=json.loads((folder/f'{state}_se.json').read_text());df=json.loads((folder/f'{state}_dft.json').read_text());gs=json.loads((folder/'dft_ground.json').read_text())
 dft_frozen=2*df['mixed']['energy_hartree']-df['triplet']['energy_hartree']-gs['energy_hartree']
 se_frozen=2*se['mixed_hartree']-se['triplet_hartree']-se['ground_hartree']
 residuals=[dft_frozen-df['frozen_excitation_hartree'],se_frozen-se['frozen_excitation_hartree'],se['relaxed_excitation_hartree']-se_frozen-se['relaxation_hartree'],dft_frozen+se['relaxation_hartree']-row['hybrid_ev']/EV]
 assert max(abs(x) for x in residuals)<1e-11
 na=gs['alpha_electrons'];nb=gs['beta_electrons'];m=df['mixed'];t=df['triplet']
 assert max(abs(m['alpha_electrons']-na),abs(m['beta_electrons']-nb),abs(t['alpha_electrons']-na-1),abs(t['beta_electrons']-nb+1))<1e-7
 coulomb=m['coulomb_hartree']-t['coulomb_hartree'];assert abs(coulomb)<1e-8
 xc_identity=(m['energy_hartree']-t['energy_hartree'])-(m['xc_including_hf_hartree']-t['xc_including_hf_hartree']);assert abs(xc_identity)<1e-8
 case=next(c for c in plan['cases'] if c['name']==name);target=next(x for x in case['targets'] if x['id']==state)
 path=Path(target['checkpoint']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['checkpoint_sha256']
 with np.load(path) as f:xyz=f['coordinates_angstrom']
 result=load_derivative_point(path,request_hash=target['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
 assert result.converged and result.spin_squared==0 and result.orbital_residual<=1e-9
 assert abs(result.excitation_energy_hartree-se['relaxed_excitation_hartree'])<1e-12
 if 'spin_flip_error_hartree' in df:
  assert abs(df['spin_flip_error_hartree'])<1e-8;spins.append(dict(molecule=name,state=state,error_hartree=df['spin_flip_error_hartree']))
 checks.append(dict(molecule=name,state=state,energy_identity_max_error_hartree=max(abs(x) for x in residuals),mixed_triplet_coulomb_difference_hartree=coulomb,spin_xc_identity_error_hartree=xc_identity,orbital_residual=result.orbital_residual))
assert len(checks)==10 and len(spins)==2
excluded=[r for r in summary['rows'] if not r['qualified']];assert len(excluded)==1 and excluded[0]['molecule']=='cytosine' and excluded[0]['id']=='n_pi'
output=dict(host=HOST,states_audited=len(checks),quest_values_checked=len(checks),spin_flip_checks=spins,maximum_energy_identity_error_hartree=max(x['energy_identity_max_error_hartree'] for x in checks),passed=True,rows=checks)
(OUT/'audit.json').write_text(json.dumps(output,indent=2)+'\n');print(output)
