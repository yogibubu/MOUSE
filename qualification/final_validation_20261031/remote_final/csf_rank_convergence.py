"""Systematic CSF-truncation diagnostic, same response/weights/lambda and no offset."""
from common import *
from independent import optimize_parent
from matrix_gfn2.multicsf import MultiCSFObjective
import argparse,fcntl,traceback

def run(name,rank):
 root=OUT/'independent_challenge';rows=[r for r in json.loads((root/'manifest.json').read_text())['states'] if r['molecule']==name];src=root/'predictions'/name;refdir=root/'references'/name
 raw=loadpoint(src/'native_ground.npz');s=canonical_state(raw);raw=replace(raw,reference_state=s);xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR;calc=calculator(s);ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s);no=s.electron_count//2
 data=[]
 for profile in ['full_response','qualified_spectral_response']:
  f=src/profile/'anchor/model.npz'
  with np.load(f) as a:d={k:a[k].copy() for k in ['operators','charge','spin','linear','q0']}
  data.append(FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,d,sha(f)))
 grounds=[ground(src/f'grounds/{i}',calc,xyz,s,r) for i,r in enumerate(data)]
 cert=json.loads((src/'native_certificate.json').read_text());allgrounds=[RestrictedGroundResult(raw,cert['minimum_curvature'],cert['index'])]+grounds
 with np.load(src/'full_response/anchor/projected_eri.npz') as a:eri=a['eri'].copy()
 with np.load(refdir/'reference.npz') as a:dc=a['coefficients'].copy();dn=int(np.count_nonzero(a['occupations']==2))
 mol=scf.chkfile.load_mol(str(refdir/'backend.chk'));small,t=source_molecule(s);cross=(gto.intor_cross('int1e_ovlp',small,mol).T@t).T;overlap=s.coefficients.T@cross@dc
 for row in rows:
  out=OUT/f'csf_convergence/{name}/{row["id"]}/rank_{rank}';out.mkdir(parents=True,exist_ok=True)
  if (out/'result.json').exists():continue
  try:
   with np.load(refdir/f'{row["id"]}.npz') as a:dx=a['amplitudes'][row['ordinal_same_symmetry']-1].copy()
   amp=overlap[:no,:dn]@dx@overlap[no:,dn:].T;amp/=np.linalg.norm(amp);weights=np.linalg.svd(amp,compute_uv=False)**2;weights/=weights.sum();parents=[];refs=[];amps=[];started=time.monotonic()
   for i,(response,g) in enumerate(zip(data,grounds)):
    c0=g.coefficients[0];q=c0[:,:no].T@s.overlap@s.coefficients[:,:no]@amp@s.coefficients[:,no:].T@s.overlap@c0[:,no:];q/=np.linalg.norm(q);refs.append(c0);amps.append(q);u,sv,vh=np.linalg.svd(q,full_matrices=True);c=np.column_stack([c0[:,:no]@u,c0[:,no:]@vh.T]);context=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=response);obj=MultiCSFObjective(context,eri,no,rank,c0,q)
    previous=OUT/f'csf_convergence/{name}/{row["id"]}/rank_{rank-1}/parent_{i}/converged.npz'
    if previous.exists():
     with np.load(previous) as a:c=a['coefficients'].copy()
    if i==1:
     from parent_homotopy import follow_parent
     c=follow_parent(calc,xyz,s,data,eri,no,rank,c0,q,parents[0],out/'homotopy_Q');(out/f'parent_{i}').mkdir(exist_ok=True);np.savez_compressed(out/f'parent_{i}/converged.npz',coefficients=c);parents.append(c)
    else:parents.append(optimize_parent(obj,c,out/f'parent_{i}'))
   bundle=out/'portable_model'
   if not (bundle/'model.json').exists():write_multicsf_model(bundle,template=raw,responses=data,eri=eri,initial_coefficients=parents,reference_coefficients=refs,reference_transitions=amps,transition_weights=weights,grounds=allgrounds,active_rank=rank,provenance=dict(source_sha256=sha(__file__),purpose='CSF expansion convergence, no reference excitation energies used',rank=rank,weights=[.5,.5],lambda_exchange=.9,offset=0))
   model=MultiCSFModel(bundle/'model.json');assert model.rank==rank and model.manifest['rank_policy']=='explicit_fixed_rank';point=evaluate_multicsf_surface(calc,xyz,model,compute_hessian=False);save_multicsf_result(out/'accepted',point,model)
   write(out/'result.json',dict(status='matched',id=row['id'],molecule=name,rank=rank,model_rank=model.rank,rank_policy=model.manifest['rank_policy'],excitation_ev=.5*sum(point.parent_gaps_hartree)*EV,parent_gaps_ev=np.array(point.parent_gaps_hartree)*EV,residual=point.orbital_residual,fidelity=point.fidelity,nto_captured_weight=float(sum(weights[:rank])),seconds=time.monotonic()-started,energy_reference_labels_used=False,scope='post-holdout model-truncation development; not an independent accuracy validation'))
  except Exception:write(out/'attempt_closed.json',dict(status='unresolved',traceback=traceback.format_exc(),checkpoints_retained=True))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('rank',type=int);args=a.parse_args();assert 2<=args.rank<=4;from science_slots import science_locks;locks=science_locks()
 with threadpool_limits():run(args.name,args.rank)
