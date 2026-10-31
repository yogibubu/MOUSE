"""Small explicit calibration with molecule-excluded prediction on Oracle."""
from remote_bootstrap import HOST
from pathlib import Path
import json,numpy as np
from matrix_gfn2.deltascf_derivatives import load_derivative_point
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/extent/calibration';OUT.mkdir(exist_ok=True)
EV=27.211386245988
uracil={}
for name in ['b3lyp_avdz','pbe0_avdz','pbe0_avtz']:
 d=json.loads((ROOT/'results/scale_0.9'/f'{name}_comparison.json').read_text())
 with np.load(ROOT/f'results/extent/{name}_moments.npz') as f:rd=f['mapped_radii']
 with np.load(ROOT/'results/extent/gfn2_moments.npz') as f:rs=f['radii']
 rows=[]
 for row,target in zip(d['targets'],[4.935,5.381]):
  rows.append(dict(state=row['state'],gap=row['dft_gap_ev'],residual=row['se_roks_ev']-row['se_gap_ev'],radius_ratio=float(rs[row['hole']]/rd[row['hole']]),tbe=target))
 uracil[name]=rows
hold=json.loads((ROOT/'results/extent/thymine/comparison.json').read_text())
checkpoint=ROOT/'results/extent/thymine/ground.npz'
# The existing derivative checkpoint keeps both coordinates and the original spectrum.
with np.load(checkpoint) as f:xyz=f['coordinates_angstrom']
g=load_derivative_point(checkpoint,request_hash='extent_thymine_ground',coordinates_angstrom=xyz);s=g.reference_state
thymine={}
for name in ['pbe0_avdz','pbe0_avtz']:
 rows=[]
 for state in ['n_pi','pi_pi']:
  row=next(x for x in hold['records'] if x['reference']==name and x['policy']=='constant_0.9' and x['state']==state)
  saved=next(x for x in hold['states'] if x['state']==state)
  assert saved['character_qualified'] and saved['canonical_csf_overlap']>=.7
  source=saved['se_gap_ev'];sgap=float((s.orbital_energies_hartree[saved['particle']]-s.orbital_energies_hartree[saved['hole']])*EV)
  gap=(row['roks_ev']-source+sgap)/.9
  rows.append(dict(state=state,gap=gap,residual=source-sgap,radius_ratio=row['source_hole_radius_bohr']/row['reference_hole_radius_bohr'],tbe=row['tbe_ev']))
 thymine[name]=rows

def fit(rows,power):
 x=np.array([x['gap']*x['radius_ratio']**power for x in rows]);y=np.array([x['tbe']-x['residual'] for x in rows])
 scale=float(x@y/(x@x))
 if scale<=0:raise ValueError('unphysical fitted spectral scale')
 return dict(reference_energy_scale=scale,extent_power=power,fitted_parameter_count=1)

def fit_two(rows):
 # A two-state interpolation is explicitly labeled as such; never validation.
 e=np.array([(x['tbe']-x['residual'])/x['gap'] for x in rows]);r=np.array([x['radius_ratio'] for x in rows])
 if min(e)<=0 or abs(np.log(r[0]/r[1]))<1e-8:raise ValueError('singular calibration')
 p=float(np.log(e[0]/e[1])/np.log(r[0]/r[1]));scale=float(e[0]/r[0]**p)
 return dict(reference_energy_scale=scale,extent_power=p,fitted_parameter_count=2)

def predict(rows,model):
 values=np.array([model['reference_energy_scale']*x['radius_ratio']**model['extent_power']*x['gap']+x['residual'] for x in rows]);target=np.array([x['tbe'] for x in rows]);err=values-target
 return dict(energies_ev=values.tolist(),errors_ev=err.tolist(),mae_ev=float(np.mean(abs(err))),maximum_absolute_error_ev=float(max(abs(err))),correct_order=bool(values[0]<values[1]))
records=[]
for name,rows in uracil.items():
 training_name='pbe0_avdz' if name=='pbe0_avtz' else name;train=uracil[training_name]
 for policy,model in [('baseline',dict(reference_energy_scale=.9,extent_power=1.,fitted_parameter_count=0)),('fixed_power_1.5_one_scale',fit(train,1.5)),('two_parameter_interpolation',fit_two(train))]:
  records.append(dict(reference=name,policy=policy,model=model,training_molecule='uracil',training_reference=training_name,training= predict(train,model),uracil=predict(rows,model),**({'thymine':predict(thymine[name],model)} if name in thymine else {})))
loo=[]
for train_name,test_name,data_train,data_test in [('uracil','thymine',uracil['pbe0_avdz'],thymine['pbe0_avdz']),('thymine','uracil',thymine['pbe0_avdz'],uracil['pbe0_avdz'])]:
 for policy,model in [('baseline',dict(reference_energy_scale=.9,extent_power=1.,fitted_parameter_count=0)),('fixed_power_1.5_one_scale',fit(data_train,1.5)),('two_parameter_interpolation',fit_two(data_train))]:
  loo.append(dict(policy=policy,training_molecule=train_name,test_molecule=test_name,reference='pbe0_avdz',model=model,test=predict(data_test,model)))
summary=[]
for policy in ['baseline','fixed_power_1.5_one_scale','two_parameter_interpolation']:
 errors=[e for row in loo if row['policy']==policy for e in row['test']['errors_ev']]
 summary.append(dict(policy=policy,leave_one_molecule_out_mae_ev=float(np.mean(np.abs(errors))),leave_one_molecule_out_max_error_ev=float(np.max(np.abs(errors))),errors_ev=errors))
result=dict(host=HOST,scope='four states on two related molecules, not a general validation; candidate comparison is exploratory',fixed_candidate_power=1.5,inputs=dict(uracil=uracil,thymine=thymine),records=records,leave_one_molecule_out=loo,summary=summary)
(OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
profile=dict(profile_id='experimental_pbe0_uracil_extent_20261029',reference_method='PBE0',training_basis='aug-cc-pVDZ',training_molecule='uracil',training_states=['n_pi','pi_pi'],validation_molecule='thymine',global_scale_fitted=True,power_fixed=True,**fit(uracil['pbe0_avdz'],1.5))
(OUT/'profile_pbe0_uracil.json').write_text(json.dumps(profile,indent=2)+'\n')
for row in records:
 if row['reference']=='pbe0_avdz':print(row,flush=True)
for row in summary:print('LOO',row,flush=True)
