"""Parameter-free extent hypotheses; same configuration correction in both methods."""
from remote_bootstrap import HOST
from pathlib import Path
from dataclasses import replace
import numpy as np,json
from scipy.optimize import linear_sum_assignment
from matrix_gfn2.hybrid_spectrum import MatchedOrbitalSpectrum,replace_tda_reference_gaps,with_occupied_extent_scaling
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/extent';EV=27.211386245988
with np.load(ROOT/'results/source_matrices.npz') as f:A=f['parent'];ii=f['occupied'];aa=f['virtual']
basevalues,basevectors=np.linalg.eigh(A)
anchors=[int(np.argmax(basevectors[np.flatnonzero((ii==h)&(aa==21))[0]]**2)) for h in [20,19]]
with np.load(OUT/'gfn2_moments.npz') as f:rs=f['radii']
records=[]
for name in ['b3lyp_avdz','pbe0_avdz','pbe0_avtz']:
 with np.load(ROOT/'results/scale_0.9'/f'{name}_mapped_spectrum.npz') as f:
  m=MatchedOrbitalSpectrum(f['source_energies'],np.array([2.]*21+[0.]*(len(f['source_energies'])-21)),f['reference_energies'],f['reference_indices'],f['overlaps'],f['margins'],'cached-source',name,.7,.05,.9)
 with np.load(OUT/f'{name}_moments.npz') as f:rd=f['mapped_radii']
 old=json.loads((ROOT/'results/scale_0.9'/f'{name}_comparison.json').read_text());source=np.array([t['se_roks_ev'] for t in old['targets']]);raw=m.reference_energies_hartree
 # Predeclared parameter-free candidate: repair occupied-basis compactness,
 # referenced to the lowest mapped reference virtual energy (gauge covariant).
 anchor=float(min(raw[m.source_occupations==0]));weights=rs/rd;occ=m.source_occupations==2
 corrected=with_occupied_extent_scaling(m,rs,rd)
 corrections={'constant_0.9':m.configuration_corrections(ii,aa,require_qualified=False),
  'basis_compactness':corrected.configuration_corrections(ii,aa,require_qualified=False)}
 for origin,r in [('se',rs),('dft',rd)]:
  for p in [1,2,4]:
   factor=.9*np.minimum(r[ii]/r[aa],1.)**p
   corrections[f'{origin}_hole_particle_power_{p}']=factor*(raw[aa]-raw[ii])-(m.source_energies_hartree[aa]-m.source_energies_hartree[ii])
 for policy,delta in corrections.items():
  roks=[]
  for k,h in enumerate([20,19]):
   pair=np.flatnonzero((ii==h)&(aa==21))[0];roks.append(float(source[k]+delta[pair]*EV))
  row=dict(reference=name,policy=policy,roks_ev=roks,roks_errors_ev=(np.array(roks)-[4.935,5.381]).tolist(),roks_correct_order=roks[0]<roks[1],tda=[])
  if policy=='basis_compactness':row.update(occupied_weights=weights[occ].tolist(),frontier_lambda=[float(.9*weights[h]) for h in [20,19]],anchor_hartree=anchor)
  for threshold in [.7,.6,.5]:
   allowed=(m.overlaps>=threshold)&(m.margins>=.05);ids=np.flatnonzero(allowed[ii]&allowed[aa]);v,x=np.linalg.eigh(A[np.ix_(ids,ids)]+np.diag(delta[ids]));full=np.zeros((len(ii),len(v)));full[ids]=x
   score=abs(basevectors[:,anchors].T@full);rr,cc=linear_sum_assignment(-score**2)
   for k,col in zip(rr,cc):row['tda'].append(dict(state=['n_pi','pi_pi'][k],minimum_overlap=threshold,energy_ev=float(v[col]*EV),state_overlap=float(score[k,col]),margin=float(score[k,col]-np.max(np.delete(score[k],col),initial=0.))))
  records.append(row)
(OUT/'comparison.json').write_text(json.dumps(dict(host=HOST,calibration_targets_used=False,primary_candidate='basis_compactness',records=records),indent=2)+'\n')
for row in records:
 if row['policy'] in ['basis_compactness','constant_0.9']:print(row,flush=True)
