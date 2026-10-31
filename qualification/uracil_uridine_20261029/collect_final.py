from remote_bootstrap import HOST
from benchmark import BASE,write,EV,characterize
import json,numpy as np,hashlib,sys,platform
from pathlib import Path
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_apoc import singlet_csf_overlap
from matrix_link.qmqm_component import CappedQMComponent
records={}
for path in sorted((BASE/'results').glob('*/gmf_*/summary.json')):
 d=json.loads(path.read_text());name=d['name'];target=d['target'];item=dict(d)
 if 'request_hash' in d:
  meta=json.loads((path.parent.parent/'summary.json').read_text());x=np.array(meta['input']['coordinates_angstrom']);r=load_derivative_point(path.parent/'converged.npz',request_hash=d['request_hash'],coordinates_angstrom=x)
  ids=np.flatnonzero(r.occupations[0]!=r.occupations[1]);s=r.reference_state
  seed=r.common_coefficients[:,ids].T@s.overlap@s.coefficients[:,ids]
  item['initial_active_pair_overlap']=abs(seed[0,0]*seed[1,1]+seed[0,1]*seed[1,0])
  canonical=__import__('dataclasses').replace(r,common_coefficients=s.coefficients)
  item['initial_csf_overlap']=singlet_csf_overlap(canonical.to_apoc(),r.to_apoc(),s.overlap)[0]
  pp,nn,low,_=characterize(s,meta['input']['base_atoms'],r.common_coefficients);hi=int(np.flatnonzero(r.occupations[1]-r.occupations[0]==1)[0]);pa=int(np.flatnonzero(r.occupations[0]-r.occupations[1]==1)[0]);sugar=[i for i in range(len(x)) if i not in meta['input']['base_atoms']];pop=np.sum(low[np.isin(s.ao_atom,sugar)]**2,axis=0)
  item.update(sugar_hole=float(pop[hi]),sugar_particle=float(pop[pa]))
 records[str(path.parent.relative_to(BASE/'results'))]=item
write(BASE/'gmf_summary.json',records)
quest=json.loads((BASE/'results/quest_s0/summary.json').read_text());parent=json.loads((BASE/'parent_response_diagnostic.json').read_text());qref=json.loads((BASE/'inputs/QUEST_Uracil.json').read_text())
rows=[]
for target,qi,ri in [('n_pi',0,0),('pi_pi',1,2)]:
 r=records[f'quest_s0/gmf_{target}_{1 if target=="n_pi" else 2}_ground'];tbe=qref[qi]['TBE/AVTZ'];a=dict(state=target,tbe_ev=tbe,roks_gmf_ev=r['gap_ev'],empirical_compact_tda_ev=quest['assignments'][target]['energy_ev'],parent_functional_tda_ev=parent['roots'][ri]['energy_ev'])
 for key in ['roks_gmf','empirical_compact_tda','parent_functional_tda']:a[key+'_error_ev']=a[key+'_ev']-tbe
 rows.append(a)
write(BASE/'accuracy_gfn2_only.json',dict(host=HOST,ground_reference='identical cached zero-temperature canonical GFN2 state',quest_commit='460482bcc6231dab41fe32188dc43959a7c2bdaa',rows=rows,mae_ev={key:sum(abs(r[key+'_error_ev']) for r in rows)/2 for key in ['roks_gmf','empirical_compact_tda','parent_functional_tda']}))
shifts=[]
for geom in ['s0','n_pi']:
 r=records.get(f'uridine_{geom}/gmf_n_pi_1_ground');u=records.get(f'uracil_{geom}/gmf_n_pi_1_ground');cap=records.get(f'capped_{geom}/gmf_n_pi_1_ground')
 if all(v is not None and 'gap_ev' in v for v in [r,u,cap]):shifts.append(dict(geometry=geom,state='n_pi',uridine_minus_uracil_ev=r['gap_ev']-u['gap_ev'],full_minus_capped_ev=r['gap_ev']-cap['gap_ev']))
write(BASE/'shifts_gmf.json',shifts)
source=BASE.parent/'source';files=[]
for package,names in [('matrix-gfn2',['deltascf.py','roks.py','roks_orbital_response.py','orbital_saddle.py','deltascf_analytic_hessian.py','deltascf_derivatives.py','spin_polarization.py']),('matrix-link',['scan.py','optimizer.py','qmqm_component.py','deltascf_component.py']),('matrix-apoc',['deltascf.py'])]:
 mod=package.replace('-','_');files.extend(source/'packages'/package/'src'/mod/n for n in names)
write(BASE/'final_provenance.json',dict(host=HOST,python=sys.version,platform=platform.platform(),thread_limits=2,source_hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.py')},input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE/'inputs').iterdir() if p.is_file()}))
print(json.dumps(dict(accuracy=rows,shifts=shifts),indent=2))
