"""Frozen converged-DFT-ground Fock candidate; no reference excitation labels."""
from common import *
from parent_homotopy import follow_parent
from matrix_gfn2.relative_pair_roks import load_relative_pair_roks
from matrix_gfn2.multicsf import MultiCSFObjective
from science_slots import science_locks
import argparse,traceback

def run(cohort,name):
 root=OUT/cohort;manifest=json.loads((root/'manifest.json').read_text());rows=[r for r in manifest['states'] if r['molecule']==name];src=root/'predictions'/name;refdir=root/'references'/name;out=OUT/f'ground_fock_candidate/{cohort}/{name}';out.mkdir(parents=True,exist_ok=True)
 if (out/'pass_complete.json').exists():return
 valid=[]
 for row in rows:
  previous=json.loads((src/row['id']/'result.json').read_text())
  if previous['status']=='not_representable':write(out/row['id']/'result.json',dict(id=row['id'],molecule=name,status='not_representable',same_projection_gate=True));continue
  projection=json.loads((src/row['id']/'projection.json').read_text());assert projection['rank']==1,'This frozen new cohort has only rank-one representable targets'
  valid.append((row,projection))
 if not valid:write(out/'pass_complete.json',dict(status='closed',labels_used=False));return
 raw=loadpoint(src/'native_ground.npz');s=canonical_state(raw);raw=replace(raw,reference_state=s);calc=calculator(s);xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR;context=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s);no=s.electron_count//2;references=[];data=[]
 for profile in ['full_response','qualified_spectral_response']:
  p=src/profile/'anchor/model.npz'
  with np.load(p) as a:d={k:a[k].copy() for k in ['operators','charge','spin','linear','q0']}
  data.append(d);references.append(FrozenNonlocalResponseReference.from_vertical_model(s,context.kernel.partition,d,sha(p)))
 with np.load(refdir/'reference.npz') as a:dc=a['coefficients'].copy();sd=a['overlap'].copy();ed=a['orbital_energies' if 'orbital_energies' in a else 'orbital_energies_hartree'].copy();dn=int(np.count_nonzero(a['occupations']==2))
 with np.load(src/'full_response/anchor/embedding.npz') as a:lift=a['embedding'].copy()
 target=lift.T@((sd@dc*ed)@dc.T@sd)@lift;target=.5*(target+target.T);dual=np.array([2*solve_sylvester(s.overlap,s.overlap,a) for a in data[0]['operators']]);newdata=dict(data[0],linear=np.einsum('aij,ij->a',dual,target-s.fock));np.savez_compressed(out/'ground_fock_model.npz',**newdata)
 gref=FrozenNonlocalResponseReference.from_vertical_model(s,context.kernel.partition,newdata,sha(out/'ground_fock_model.npz'));contexts=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=gref);p0=(s.coefficients*(s.alpha_occupations+s.beta_occupations))@s.coefficients.T;reproduced,_=contexts.fock_and_energy(np.array([.5*p0,.5*p0]));error=float(np.max(abs(reproduced[0]-target)));assert error<1e-8
 write(out/'calibration_certificate.json',dict(projected_converged_dft_ground_fock_error_hartree=error,reference_sha256=sha(refdir/'reference.npz'),embedding_sha256=sha(src/'full_response/anchor/embedding.npz'),protocol_sha256=sha(OUT/'ground_fock_candidate_protocol.json'),energy_labels_used=False,weights=[.5,.5],lambda_exchange=.9,offset=0))
 newground=ground(out/'ground_G',calc,xyz,s,gref);oldgrounds=[ground(src/f'grounds/{i}',calc,xyz,s,r) for i,r in enumerate(references)];cert=json.loads((src/'native_certificate.json').read_text());grounds=[RestrictedGroundResult(raw,cert['minimum_curvature'],cert['index']),newground,oldgrounds[1]]
 with np.load(src/'full_response/anchor/projected_eri.npz') as a:eri=a['eri'].copy()
 mol=scf.chkfile.load_mol(str(refdir/'backend.chk'));small,t=source_molecule(s);cross=(gto.intor_cross('int1e_ovlp',small,mol).T@t).T;overlap=s.coefficients.T@cross@dc
 for row,projection in valid:
  dest=out/row['id'];dest.mkdir(exist_ok=True)
  if (dest/'result.json').exists():continue
  try:
   with np.load(refdir/f'{row["id"]}.npz') as a:dx=a['amplitudes'][row['ordinal_same_symmetry']-1].copy()
   amp=overlap[:no,:dn]@dx@overlap[no:,dn:].T;amp/=np.linalg.norm(amp);joint=load_relative_pair_roks(src/row['id']/'stationary.npz',request_hash='independent:relative',require_gradient=False);parents=[];refs=[];amps=[]
   for i,(response,g) in enumerate(zip([gref,references[1]],grounds[1:])):
    c0=g.coefficients[0];q=c0[:,:no].T@s.overlap@s.coefficients[:,:no]@amp@s.coefficients[:,no:].T@s.overlap@c0[:,no:];q/=np.linalg.norm(q);refs.append(c0);amps.append(q)
    if i==0:c=follow_parent(calc,xyz,s,[references[0],gref],eri,no,1,c0,q,joint.components[0].common_coefficients,dest/'homotopy_G')
    else:c=joint.components[1].common_coefficients.copy()
    obj=MultiCSFObjective(prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=response),eri,no,1,c0,q);e,g,h=obj.evaluate(c,True);assert np.max(abs(g))<1e-9 and obj.fidelity(c)[1]>=.7;parents.append(c)
   bundle=dest/'portable_model'
   if not (bundle/'model.json').exists():write_multicsf_model(bundle,template=raw,responses=[gref,references[1]],eri=eri,initial_coefficients=parents,reference_coefficients=refs,reference_transitions=amps,transition_weights=np.array(projection['weights']),grounds=grounds,provenance=dict(candidate_protocol_sha256=sha(OUT/'ground_fock_candidate_protocol.json'),source_sha256=sha(__file__),cohort=cohort,state=row['id'],energy_labels_used=False))
   model=MultiCSFModel(bundle/'model.json');assert model.rank==1;point=evaluate_multicsf_surface(calc,xyz,model);save_multicsf_result(dest/'accepted',point,model)
   write(dest/'result.json',dict(status='matched',molecule=name,id=row['id'],rank=1,excitation_ev=.5*sum(point.parent_gaps_hartree)*EV,parent_gaps_ev=np.array(point.parent_gaps_hartree)*EV,residual=point.orbital_residual,fidelity=point.fidelity,energy_labels_used=False,model_fingerprint=model.fingerprint))
  except Exception:write(dest/'attempt_closed.json',dict(status='unresolved',traceback=traceback.format_exc(),checkpoints_retained=True))
 write(out/'pass_complete.json',dict(status='closed',labels_used=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('cohort');a.add_argument('name');args=a.parse_args();locks=science_locks()
 with threadpool_limits():run(args.cohort,args.name)
