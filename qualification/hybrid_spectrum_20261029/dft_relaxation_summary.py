"""Independently check saved DFT relaxation points and compare energy terms."""
from remote_bootstrap import HOST
from pathlib import Path
import json,numpy as np
from frozen_relaxation import write,digest,EV
from matrix_apoc.deltascf import (DeterminantOrbitals,DeltaSCFState,
                                SpinAdaptedSinglet,singlet_csf_overlap)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'dft_relaxation'

def main():
 protocol=json.loads((OUT/'protocol.json').read_text());rows=[];missing=[];checks=[]
 plan=json.loads((ROOT/'frozen_relaxation/plan.json').read_text())
 for molecule in protocol['cases']:
  reference=Path(next(c for c in plan['cases'] if c['name']==molecule)['reference'])/'reference.npz'
  with np.load(reference) as f:ref={k:f[k] for k in f.files}
  previous=json.loads((ROOT/'frozen_relaxation'/molecule/'result.json').read_text())
  def csf(c,occ):
   w=DeterminantOrbitals(ref['atomic_numbers'],ref['coordinates_bohr'],ref['overlap'],
    (c,c),tuple(occ),np.zeros(len(ref['atomic_numbers']),dtype=int),model='PBE0/aug-cc-pVDZ diagnostic')
   return SpinAdaptedSinglet(1,DeltaSCFState(1,w))
  for state in protocol['states']:
   folder=OUT/molecule/state;path=folder/'result.json'
   if not path.exists():missing.append(f'{molecule}/{state}');continue
   r=json.loads(path.read_text());a=json.loads((folder/'derivative_audit.json').read_text())
   with np.load(folder/'latest.npz') as f:
    assert str(f['request_hash'])==r['request_hash']==a['request_hash']
    assert np.max(abs(f['gradient']))<=protocol['orbital_tolerance']
    assert abs(float(f['energy_hartree'])-r['energy_hartree'])<1e-11
    c=f['coefficients'].copy();occ=f['occupations'].copy()
   metric_error=float(np.max(abs(c.T@ref['overlap']@c-np.eye(len(c)))))
   assert metric_error<2e-7
   candidate=csf(c,occ);scores={}
   for other in previous['records']:
    o=np.array([ref['occupations']/2]*2);o[0,other['dft_hole']]=0;o[0,other['dft_particle']]=1
    scores[other['id']]=singlet_csf_overlap(csf(ref['coefficients'],o),candidate,ref['overlap'])[0]
   margin=scores[state]-max(v for k,v in scores.items() if k!=state)
   assert scores[state]>=protocol['minimum_seed_csf_overlap'] and margin>=.05
   assert r['host']==HOST and r['converged'] and a['passed']
   assert r['seed_csf_overlap']>=protocol['minimum_seed_csf_overlap'] and r['spin_squared_csf']==0
   assert abs(2*r['mixed_energy_hartree']-r['triplet_energy_hartree']-r['energy_hartree'])<1e-11
   assert abs((r['energy_hartree']-r['ground_hartree'])*EV-r['dft_relaxed_ev'])<1e-9
   assert abs(r['dft_relaxed_ev']-r['dft_frozen_ev']-r['dft_relaxation_ev'])<1e-10
   assert abs(r['hybrid_error_ev']-r['dft_error_ev']-r['relaxation_mismatch_ev'])<1e-9
   checks.append(dict(molecule=molecule,state=state,passed=True,
    orthonormality_error=metric_error,reference_csf_overlaps=scores,identity_margin=margin,
    result_sha256=digest(path),checkpoint_sha256=digest(folder/'latest.npz'),
    derivative_audit_sha256=digest(folder/'derivative_audit.json')));rows.append(r)
   if (folder/'minimum_seed/result.json').exists():
    checks[-1]['initializer_sha256']=digest(folder/'minimum_seed/result.json')
    checks[-1]['initializer_method']='Cs-preserving variational initializer followed by full orbital-space Newton certification'
 splitting=[]
 for mol in protocol['cases']:
  states={r['state']:r for r in rows if r['molecule']==mol}
  if len(states)!=2:continue
  n,p=states['n_pi'],states['pi_pi']
  splitting.append(dict(molecule=mol,tbe_pi_minus_n_ev=p['tbe_ev']-n['tbe_ev'],
    dft_pi_minus_n_ev=p['dft_relaxed_ev']-n['dft_relaxed_ev'],
    hybrid_pi_minus_n_ev=p['hybrid_ev']-n['hybrid_ev'],
    dft_differential_relaxation_ev=p['dft_relaxation_ev']-n['dft_relaxation_ev'],
    se_differential_relaxation_ev=p['se_relaxation_ev']-n['se_relaxation_ev']))
 summary=dict(host=HOST,rows=rows,missing=missing,checks=checks,splittings=splitting,
  complete=not missing,production_ready=False,nuclear_hybrid_derivatives_available=False,
  hessian_index_certified=False,script_sha256=digest(ROOT/'dft_relaxation.py'))
 if rows:
  summary['mae_ev']={label:float(np.mean([abs(r[field]) for r in rows]))
   for label,field in [('dft_roks','dft_error_ev'),('hybrid','hybrid_error_ev'),('relaxation_transfer','relaxation_mismatch_ev')]}
 write(OUT/'summary.json',summary)
 lines=['# DFT versus GFN2 orbital relaxation: uracil and thymine','',
  'Oracle-only PBE0/aug-cc-pVDZ, identical cached geometries and ground references. '
  'Variational common-orbital ROKS objective 2 E(mixed) − E(triplet); all AO orbitals, '
  'including cores, can relax. The two-determinant CSF is a pure singlet. '
  'The functional energy is the approximate ROKS expression, not exact spin projection of DFT.',
  '',f'Completed and checked: {len(rows)}/4. No fitted parameters. '
  'Stationarity uses an analytic orbital gradient and UKS response Hessian-vector products; '
  'directional finite differences check those derivatives. Newton-GMRES permits saddles. '
  'The Hessian index is not certified. This is not a nuclear Hessian of the hybrid model.',
  '', 'Thymine n→π* required a Cs-preserving variational initializer resumed from '
  'the retained Newton checkpoint. The geometry was not changed. The final Newton '
  'residual includes every active rotation, including those between different irreps. '
  'The initial failed derivative checks and subsequent finite-difference step '
  'refinements are retained; the original acceptance tolerances were unchanged.',
  '', '| Molecule | State | DFT frozen | DFT relaxation | SE relaxation | DFT relaxed | Hybrid | QUEST |',
  '|---|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  lines.append(f"| {r['molecule']} | {r['state']} | {r['dft_frozen_ev']:.6f} | {r['dft_relaxation_ev']:.6f} | {r['se_relaxation_ev']:.6f} | {r['dft_relaxed_ev']:.6f} | {r['hybrid_ev']:.6f} | {r['tbe_ev']:.6f} |")
 lines+=['','All energies in eV. Hybrid = DFT frozen + SE relaxation.',
  'Hybrid error − full DFT ROKS error = SE relaxation − DFT relaxation; '
  'this identity is independently checked against the saved energies.',
  '', '| Molecule | State | Full DFT ROKS error | Relaxation transfer error | Hybrid error |',
  '|---|---|---:|---:|---:|']
 for r in rows:
  lines.append(f"| {r['molecule']} | {r['state']} | {r['dft_error_ev']:.6f} | {r['relaxation_mismatch_ev']:.6f} | {r['hybrid_error_ev']:.6f} |")
 lines+=[
  '', '| Molecule | QUEST π−n | DFT π−n | Hybrid π−n | DFT Δrelax(π−n) | SE Δrelax(π−n) |',
  '|---|---:|---:|---:|---:|---:|']
 for s in splitting:
  lines.append('| '+s['molecule']+' | '+' | '.join(f"{s[k]:.6f}" for k in ['tbe_pi_minus_n_ev','dft_pi_minus_n_ev','hybrid_pi_minus_n_ev','dft_differential_relaxation_ev','se_differential_relaxation_ev'])+' |')
 if not missing:
  lines+=['','## Interpretation','',
   'Full PBE0 ROKS restores the n→π* / π→π* ordering for both molecules, '
   'although the splittings remain smaller than QUEST. For n→π*, GFN2 supplies '
   'too little relaxation: replacing it with the DFT relaxation lowers the hybrid '
   'energy by about 0.44 eV in uracil and 0.47 eV in thymine. '
   'For π→π*, SE and DFT relaxations agree within 0.008 eV: '
   'the remaining 0.36–0.42 eV underestimation is already present in PBE0 ROKS.',
   '', 'A change to the relaxation transfer alone cannot remove the π→π* error. '
   'The next model must distinguish the relaxation-transfer defect for n→π* '
   'from the reference-functional defect for π→π*, and be evaluated on the '
   'broader molecular set before any new orbital-dependent scaling is adopted. '
   'These four diagnostic cases do not establish a universal correction.']
 lines+=['','These four states are diagnostic cases selected after earlier errors, '
  'not an independent validation set. No timing claim against TDA is made. '
  'The hybrid nuclear gradient and analytic nuclear Hessian remain unavailable.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(complete=summary['complete'],missing=missing,splittings=splitting),indent=2))

if __name__=='__main__':main()
