"""TDA on actual GFN2 orbitals, using native or matched ground-state response.

No DFT orbitals enter the TDA transition basis. Optional mapped DFT gaps are
reported separately, in a common qualified subspace, at scales 1 and 0.9.
The 0.9 exact-exchange response factor is already in U and is not applied twice.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json,time,argparse,numpy as np
from types import SimpleNamespace
from pyscf import scf,gto,lib
OLD=Path('/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029');sys.path.insert(0,str(OLD))
from frozen_relaxation import plan,state_from,write,key,digest,EV
from extent_support import source_molecule
from matrix_gfn2 import GFN2Calculator,FrozenResponseReference
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.hybrid_spectrum import match_orbital_spectrum,replace_tda_reference_gaps
lib.num_threads(8);ROOT=BASE/'results/tda';ROOT.mkdir(exist_ok=True)

def run(case,cohort):
 prefix=BASE/'results' if cohort=='development' else BASE/'results/holdout'
 model=prefix/'response_kernel'/case['name'];refpath=prefix/'bhandhlyp'/case['name']/'reference.npz'
 if not (model/'kernel.npz').exists() or not (model/'result.json').exists():return
 source=json.loads((model/'result.json').read_text());valid=[t for t in case['targets'] if 'checkpoint' in t]
 out=ROOT/cohort/case['name'];out.mkdir(parents=True,exist_ok=True)
 request=key(dict(kernel=digest(model/'kernel.npz'),source=digest(valid[0]['checkpoint']),script=digest(__file__)))
 final=out/'result.json'
 if final.exists():
  old=json.loads(final.read_text());assert old['request_hash']==request;print('REUSE TDA',cohort,case['name'],flush=True);return
 r,xyz=state_from(valid[0]);s=r.reference_state
 with np.load(model/'kernel.npz') as f:
  ref=FrozenResponseReference(f['atomic_numbers'],f['orbital_shell'],f['spin_coupling'],
   'PBE0/frozen-charge-spin-response/'+str(f['request_hash']),.9,f['charge_correction'],f['reference_shell_populations'])
 assert all(x.get('spin_reference_fingerprint')==ref.fingerprint for x in source['records'] if x.get('qualified'))
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r.ground_energy_hartree)
 ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=ref)
 occupied=np.flatnonzero(s.alpha_occupations+s.beta_occupations==2);virtual=np.flatnonzero(s.alpha_occupations+s.beta_occupations==0)
 ii=np.repeat(occupied,len(virtual));aa=np.tile(virtual,len(occupied));ci=s.coefficients[:,ii];ca=s.coefficients[:,aa]
 t0=time.perf_counter();base=calc.density_response_kernel(s);columns=[];corrected=[]
 for i,a in zip(ii,aa):
  tr=np.outer(s.coefficients[:,i],s.coefficients[:,a]);tr=.5*(tr+tr.T)
  f=base(tr);delta=ctx.charge_correction.response(tr)
  columns.append(np.einsum('mi,mn,ni->i',ci,f,ca))
  corrected.append(np.einsum('mi,mn,ni->i',ci,f+delta,ca))
 diagonal=np.diag(s.orbital_energies_hartree[aa]-s.orbital_energies_hartree[ii])
 a0=diagonal+2*np.array(columns).T;a1=diagonal+2*np.array(corrected).T
 build=time.perf_counter()-t0
 assert max(np.max(abs(a0-a0.T)),np.max(abs(a1-a1.T)))<1e-9
 t0=time.perf_counter();v0,u0=np.linalg.eigh(a0);d0=time.perf_counter()-t0
 t0=time.perf_counter();v1,u1=np.linalg.eigh(a1);d1=time.perf_counter()-t0
 np.savez_compressed(out/'tda.npz',native_matrix=a0,matched_matrix=a1,native_energies=v0,matched_energies=v1,
  native_vectors=u0,matched_vectors=u1,occupied=ii,virtual=aa)
 spectral=[]
 if refpath.exists():
  with np.load(refpath) as f:d={k:f[k] for k in f.files}
  from dataclasses import replace
  mol=scf.chkfile.load_mol(str(refpath.parent/'backend.chk'));small,t=source_molecule(s)
  cross=(gto.intor_cross('int1e_ovlp',mol,small)@t).T
  mapping=match_orbital_spectrum(s.coefficients,s.overlap,s.alpha_occupations+s.beta_occupations,
   s.orbital_energies_hartree,d['coefficients'],d['overlap'],d['occupations'],d['orbital_energies_hartree'],cross,
   reference_model='BHANDHLYP/aug-cc-pVDZ')
  ids=np.flatnonzero(mapping.qualified[ii]&mapping.qualified[aa])
  if len(ids):
   for scale in [1.,.9]:
    mapped=replace(mapping,reference_energy_scale=scale)
    matrix=replace_tda_reference_gaps(a1[np.ix_(ids,ids)],ii[ids],aa[ids],mapped)
    values,vectors=np.linalg.eigh(matrix)
    spectral.append((scale,ids,values,vectors))
    np.savez_compressed(out/f'dft_gaps_{scale}.npz',energies=values,vectors=vectors,configuration_indices=ids)
 rows=[]
 for target in valid:
  anchor=np.zeros(len(ii));anchor[np.flatnonzero((ii==target['hole'])&(aa==target['particle']))[0]]=1
  if cohort=='holdout':
   with np.load(BASE/'holdout_data'/case['name']/'reference_transition.npz') as f:anchor=f['amplitude'].ravel()
  row=dict(state=target['id'],tbe_ev=target['tbe_ev'],configuration_count=len(ii),spectral_tda=[])
  for label,values,vectors in [('native',v0,u0),('matched',v1,u1)]:
   scores=abs(anchor@vectors);index=int(np.argmax(scores));margin=float(scores[index]-np.max(np.delete(scores,index),initial=0.))
   energy=float(values[index]*EV)
   row[label]=dict(energy_ev=energy,error_ev=energy-target['tbe_ev'],state_overlap=float(scores[index]),
    margin=margin,qualified=bool(scores[index]>=.7 and margin>=.05))
  for scale,ids,values,vectors in spectral:
   scores=abs(anchor[ids]@vectors);index=int(np.argmax(scores));weight=float(np.sum(anchor[ids]**2));margin=float(scores[index]-np.max(np.delete(scores,index),initial=0.))
   energy=float(values[index]*EV)
   row['spectral_tda'].append(dict(reference_gap_scale=scale,energy_ev=energy,error_ev=energy-target['tbe_ev'],
    state_overlap=float(scores[index]),margin=margin,trusted_configuration_weight=weight,
    qualified=bool(scores[index]>=.7 and margin>=.05 and weight>=.95),configuration_count=len(ids)))
  match=next(x for x in source['records'] if x['id']==target['id']);row['roks']=match
  rows.append(row)
 report=dict(host=HOST,request_hash=request,molecule=case['name'],cohort=cohort,
  orbitals='GFN2 canonical ground orbitals only',lambda_exchange_already_in_response=.9,
  openmp_threads=8,shared_matrix_build_seconds=build,native_diagonalization_seconds=d0,
  matched_diagonalization_seconds=d1,matched_tda_solver_seconds=build+d1,
  dft_preparation_excluded_from_solver_timing=True,
  roks_initialization='canonical GFN2 ground orbitals' if cohort=='holdout' else 'retained relaxed SE checkpoint',rows=rows)
 write(final,report);print('TDA',case['name'],report,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['development','holdout']);args=p.parse_args()
 if args.cohort!='holdout':
  for case in plan():run(case,'development')
 if args.cohort!='development':
  for p in (BASE/'holdout_data').glob('*/case.json'):run(json.loads(p.read_text()),'holdout')
