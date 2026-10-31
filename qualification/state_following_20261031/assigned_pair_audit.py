"""Compare recovered single-pair directions with all available same-symmetry roots."""
from pathlib import Path
import socket,os,sys,json
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MOUSE_VALIDATION_ROOT='/home/enzo/MOUSE-orbital-performance-20261031')
R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030')
V=Path('/home/enzo/MOUSE-orbital-performance-20261031')
sys.path.insert(0,str(V/'source/qualification'));import remote_bootstrap
import numpy as np
from pyscf import scf,gto,lib
from threadpoolctl import threadpool_limits
from matrix_gfn2.relative_pair_roks import load_relative_pair_roks
from matrix_gfn2.deltascf_derivatives import load_derivative_point
sys.path.append('/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029');from extent_support import source_molecule
lib.num_threads(2);rows=[]
with threadpool_limits(limits=2):
 for outcome in sorted((V/'results').glob('*/*/result.json')):
  result=json.loads(outcome.read_text())
  if result.get('adopted_completed_exploratory_result'):
   a=result['assignment_audit'];f=a['target_parent_fidelities'];m=a['projected_overlap_margin_against_all_converged_same_symmetry_roots']
   assert all(q*q>.5 for q in f) and all(q>=.05 for q in m)
   result['identity_policy_reaudit']=dict(attainable_single_pair_fidelities=f,projected_margins=m,passed=True,reference_coverage_complete=False)
   outcome.write_text(json.dumps(result,indent=2)+'\n');continue
  if result['status'] not in ['stationary_pending_assignment_audit','matched','assignment_ambiguous']:continue
  name=result['molecule'];sid=result['state'];d=outcome.parent
  joint=load_relative_pair_roks(d/'stationary.npz',request_hash='automatic_v2_relative',require_gradient=False)
  # Joint geometry-only metadata carries a core-Hamiltonian seed chart.
  # Assignment must use the original fixed native-ground occupied/virtual spaces.
  ground=R/'reduced_validation/prepared'/name/'ground.npz'
  with np.load(ground) as f:gxyz=f['coordinates_angstrom'].copy();grequest=str(f['request_hash'])
  s=load_derivative_point(ground,request_hash=grequest,coordinates_angstrom=gxyz,require_gradient=False).reference_state
  reference=P/'blind/mouse_reference'/name
  mol=scf.chkfile.load_mol(str(reference/'backend.chk'));small,t=source_molecule(s);cross=gto.intor_cross('int1e_ovlp',mol,small)@t
  with np.load(reference/'reference.npz') as f:cd=f['coefficients'];od=f['occupations']
  oi=np.flatnonzero(s.alpha_occupations==1);vi=np.flatnonzero(s.alpha_occupations==0);di=np.flatnonzero(od==2);da=np.flatnonzero(od==0)
  overlap=s.coefficients.T@cross.T@cd
  with np.load(reference/'backend_tda.npz') as f:amplitudes=f['amplitudes'].reshape(-1,len(di),len(da));labels=f['state_symmetries'];converged=f['converged']
  case=json.loads((R/'reduced_validation/prepared'/name/'case.json').read_text());target=next(t for t in case['targets'] if t['id']==sid);root=target['reference_root'];wanted=labels[root-1]
  candidates=[]
  for i,x in enumerate(amplitudes):
   if labels[i]!=wanted:continue
   x=x/np.linalg.norm(x);q=overlap[np.ix_(oi,di)]@x@overlap[np.ix_(vi,da)].T;norm=float(np.linalg.norm(q));sigma=np.linalg.svd(q,compute_uv=False);cap=float(sigma[0]/norm) if norm else 0
   if norm<1e-12 or cap<1e-12:continue
   q/=norm;scores=[]
   for parent in joint.components:
    pair=np.flatnonzero(parent.occupations[0]!=parent.occupations[1]);directions=[]
    for h,p in (pair,pair[::-1]):
     hole=s.coefficients[:,oi].T@s.overlap@parent.common_coefficients[:,h];particle=s.coefficients[:,vi].T@s.overlap@parent.common_coefficients[:,p];rank=np.outer(hole,particle);n=np.linalg.norm(rank)
     if min(np.linalg.norm(hole),np.linalg.norm(particle))>=.7:directions.append(float(abs(np.sum(q*rank))/n)/cap)
    scores.append(max(directions,default=0.))
   candidates.append(dict(root=i+1,converged=bool(converged[i]),projection_norm=norm,rank_one_limit=cap,parent_fidelities=scores,parent_projected_overlaps=[float(q*cap*norm) for q in scores],representable_single_pair=bool(norm>=.7 and cap>=.7)))
  own=next(r for r in candidates if r['root']==root);margin=[own['parent_fidelities'][i]-max((r['parent_fidelities'][i] for r in candidates if r['root']!=root and r['converged'] and r['representable_single_pair']),default=0) for i in range(2)]
  physical_margin=[own['parent_projected_overlaps'][i]-max((r['parent_projected_overlaps'][i] for r in candidates if r['root']!=root and r['converged']),default=0) for i in range(2)]
  row=dict(molecule=name,state=sid,target_root=root,symmetry=str(wanted),target_parent_fidelities=own['parent_fidelities'],margins_against_other_converged_representable_same_symmetry_roots=margin,projected_overlap_margin_against_all_converged_same_symmetry_roots=physical_margin,target_raw_overlaps=[q*own['rank_one_limit'] for q in own['parent_fidelities']],attainable_single_pair_fidelity_squared_gt_half=all(q*q>.5 for q in own['parent_fidelities']),uniquely_assigned=all(m>=.05 for m in physical_margin) and all(q*q>.5 for q in own['parent_fidelities']),reference_coverage_complete=False,available_same_symmetry_candidates=candidates,numerical_reference_energies_used=False)
  rows.append(row);print(row,flush=True)
  result['assignment_audit']=row;result['status']='matched' if row['uniquely_assigned'] else 'assignment_ambiguous'
  temp=outcome.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(outcome)
(V/'assignment_audit.json').write_text(json.dumps(dict(host=socket.getfqdn(),rows=rows,scope='available frozen DFT roots only; absence of additional roots is not proof of global uniqueness',baseline_untouched=True),indent=2)+'\n')
