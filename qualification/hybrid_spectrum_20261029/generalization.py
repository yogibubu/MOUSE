"""Frozen-profile, checkpointed molecular validation on Oracle; no refitting."""
from remote_bootstrap import HOST,BASE
from extent_support import source_molecule,radii
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import argparse,hashlib,json,sys,time,traceback,numpy as np
from scipy.optimize import linear_sum_assignment
from pyscf import gto,scf,lib
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_deltascf,evaluate_gfn2_roks_singlet
from matrix_gfn2.response_reference import stationary_response_reference
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.roks import ROKSSingletResult
from matrix_gfn2.hybrid_spectrum import match_orbital_spectrum,with_occupied_extent_scaling,correct_roks_excitation,replace_tda_reference_gaps
from matrix_apoc.deltascf import singlet_csf_overlap
lib.num_threads(2)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'generalization';EV=27.211386245988
PROTOCOL=json.loads((OUT/'protocol.json').read_text())
INDEPENDENT=Path('/home/enzo/MATRIX-runs/tda-dft-independent-20261022-v1/docs/reports/independent_dft_tda_20261022')
FROZEN=Path('/home/enzo/MATRIX-runs/tda-frozen-validation-20261022-v1/docs/reports/tda_frozen_validation_20261022')
PYRIDINE=Path('/home/enzo/MATRIX-runs/tda-pyridine-pyrrole-20261022-v1/docs/reports/tda_pyridine_pyrrole_20261022/pyridine')

def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 tmp.write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n');tmp.replace(path)

def character(s,c):
 # All atoms define the plane, including H in three-atom HNO.
 _,sv,v=np.linalg.svd(s.coordinates_bohr-s.coordinates_bohr.mean(axis=0));normal=v[-1]
 if sv[-1]>1e-4 or sv[1]<1e-4:raise ValueError('nonplanar or linearly ambiguous orbital character')
 w,u=np.linalg.eigh(s.overlap);low=(u*np.sqrt(w))@u.T@c;cart=s.basis.spherical_to_cartesian@low
 pi=np.zeros(s.nmo);hetero=np.zeros(s.nmo)
 for atom,z in enumerate(s.atomic_numbers):
  ids=[np.flatnonzero((s.basis.cartesian_atom==atom)&(s.basis.cartesian_type==t)) for t in (2,3,4)]
  if not all(len(i)==1 for i in ids):continue
  block=cart[[i[0] for i in ids]];perp=(normal@block)**2;pi+=perp
  if z in [7,8]:hetero+=np.sum(block**2,axis=0)-perp
 return pi,hetero,normal

def inputs(case):
 if case['dataset']=='independent':
  folder=INDEPENDENT/'results'/case['folder'];meta_path=INDEPENDENT/'inputs'/case['metadata'];metadata=json.loads(meta_path.read_text())
  target=[]
  for item in case['targets']:
   m=metadata[item['metadata_index']]
   assert m['Spin']==1 and m['V/R']=='V' and m['Type']==item['family']
   target.append(dict(**item,tbe_ev=m['TBE/AVTZ'],state_label=m['State'],assignment_basis='lowest valence n-pi singlet; backend root checked by orbital character'))
  vectors_path=folder/'backend_tda.npz'
 else:
  parent=PYRIDINE if case['dataset']=='pyridine' else FROZEN/case['name'];folder=parent/'results/pbe0_avdz';meta_path=folder/'result.json';metadata=json.loads(meta_path.read_text());target=[]
  for item in case['targets']:
   m=next(x for x in metadata['assignments'] if x['metadata_index']==item['metadata_index'])
   assert m['resolved'] and m['family']==item['family'] and m['symmetry']['reliable']
   target.append(dict(**item,reference_root=m['reference_root'],tbe_ev=m['tbe_ev'],state_label=m['state'],assignment_basis='retained symmetry/character assignment',dft_leading_configurations=m['descriptor']['leading_configurations'][:4]))
  vectors_path=folder/'selected_full.npz'
 with np.load(folder/'reference.npz') as f:ref={k:f[k] for k in f.files}
 with np.load(vectors_path) as f:vectors=f['amplitudes']
 no=int(np.count_nonzero(ref['occupations']>1));nv=len(ref['occupations'])-no
 vectors=vectors.reshape(len(vectors),no,nv)
 assert np.allclose(np.sum(vectors**2,axis=(1,2)),1.,atol=1e-6)
 for t in target:t['reference_amplitude']=vectors[t['reference_root']-1]
 return folder,meta_path,ref,target

def run(case,*,index_zero=False):
 name=case['name'];out=OUT/name;out.mkdir(exist_ok=True);folder,meta_path,ref,targets=inputs(case)
 xyz=ref['coordinates_bohr']*.529177210903;zs=ref['atomic_numbers'];request='generalization_'+name
 suffix='_index0' if index_zero else ''
 status_path=out/f'status{suffix}.json'
 status=dict(name=name,host=HOST,solver_search_scope=('additional restricted-CSF minimum diagnostic' if index_zero else 'original positive-index search'),profile=dict(reference_energy_scale=PROTOCOL['reference_energy_scale'],extent_power=PROTOCOL['extent_power']),input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'reference.npz',folder/'backend.chk',meta_path]},targets=[])
 write(status_path,dict(**status,stage='ground reference'));t0=time.perf_counter()
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in zs],temperature_kelvin=0.)
 file=out/'ground.npz'
 if file.exists():
  gs=load_derivative_point(file,request_hash=request,coordinates_angstrom=xyz);s=gs.reference_state
 else:
  ground,cert=stationary_response_reference(calc,xyz);s=ground.orbital_state
  gs=evaluate_gfn2_deltascf(calc,xyz,s.alpha_occupations,s.beta_occupations,ground_state=ground,selection='aufbau',gradient=True,density_tolerance=1e-10,orbital_tolerance=1e-10,energy_tolerance=1e-12)
  save_derivative_point(file,gs,request_hash=request,coordinates_angstrom=xyz);s=gs.reference_state;write(out/'ground_certificate.json',cert)
 status['ground_load_or_solve_seconds']=time.perf_counter()-t0
 ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=gs.energy_hartree)
 small,transform=source_molecule(s);mol=scf.chkfile.load_mol(str(folder/'backend.chk'))
 assert np.array_equal(zs,s.atomic_numbers) and np.max(abs(mol.atom_coords()-s.coordinates_bohr))<1e-8
 assert np.max(abs(mol.intor('int1e_ovlp')-ref['overlap']))<1e-9
 cross=(gto.intor_cross('int1e_ovlp',mol,small)@transform).T
 mapping=match_orbital_spectrum(s.coefficients,s.overlap,s.alpha_occupations+s.beta_occupations,s.orbital_energies_hartree,ref['coefficients'],ref['overlap'],ref['occupations'],ref['orbital_energies_hartree'],cross,reference_model='PBE0/aug-cc-pVDZ/'+name,reference_energy_scale=PROTOCOL['reference_energy_scale'])
 rs=radii(small,transform@s.coefficients);rd_all=radii(mol,ref['coefficients']);rd=rd_all[mapping.reference_indices]
 # Keep a mapping failure visible even when one descriptor or anchor is unavailable.
 try:extent=with_occupied_extent_scaling(mapping,rs,rd,power=PROTOCOL['extent_power'])
 except Exception as exc:extent=None;status['extent_error']=str(exc)
 previous=with_occupied_extent_scaling(replace(mapping,reference_energy_scale=.9),rs,rd) if extent is not None else None
 occupied=np.flatnonzero(s.alpha_occupations>.5);virtual=np.flatnonzero(s.alpha_occupations<.5);di=np.flatnonzero(ref['occupations']>1);da=np.flatnonzero(ref['occupations']<1)
 ii=np.repeat(occupied,len(virtual));aa=np.tile(virtual,len(occupied));good=mapping.qualified[ii]&mapping.qualified[aa];ids=np.flatnonzero(good)
 overlap=s.coefficients.T@cross@ref['coefficients'];oi=overlap[np.ix_(occupied,di)];ov=overlap[np.ix_(virtual,da)]
 pi,hetero,normal=character(s,s.coefficients)
 np.savez_compressed(out/'mapping.npz',source_radii=rs,reference_radii=rd,reference_indices=mapping.reference_indices,overlaps=mapping.overlaps,margins=mapping.margins,qualified=mapping.qualified,source_energies=s.orbital_energies_hartree,reference_energies=mapping.reference_energies_hartree,cross_overlap=cross)
 # Parent-functional TDA on the same GFN2 orbitals, with explicitly retained space.
 t0=time.perf_counter();cache=out/'parent_tda.npz'
 if cache.exists():
  with np.load(cache) as f:A=f['matrix'];assert np.array_equal(f['occupied'],ii) and np.array_equal(f['virtual'],aa)
 else:
  kernel=calc.density_response_kernel(s);ci=s.coefficients[:,ii];ca=s.coefficients[:,aa];columns=[]
  for i,a in zip(ii,aa):
   tr=np.outer(s.coefficients[:,i],s.coefficients[:,a]);fock=kernel(.5*(tr+tr.T));columns.append(np.einsum('mi,mn,ni->i',ci,fock,ca))
  A=np.diag(s.orbital_energies_hartree[aa]-s.orbital_energies_hartree[ii])+2*np.array(columns).T
  np.savez_compressed(cache,matrix=A,occupied=ii,virtual=aa)
 assert np.max(abs(A-A.T))<1e-10
 original_values,original_vectors=np.linalg.eigh(A)
 restricted_values,restricted_vectors=np.linalg.eigh(A[np.ix_(ids,ids)]) if len(ids) else (None,None)
 if extent is not None and len(ids):hybrid_values,hybrid_vectors=np.linalg.eigh(replace_tda_reference_gaps(A[np.ix_(ids,ids)],ii[ids],aa[ids],extent))
 else:hybrid_values=hybrid_vectors=None
 status['tda_build_and_diagonalization_seconds']=time.perf_counter()-t0;status.update(full_configurations=len(ii),trusted_configurations=len(ids),plane_normal=normal)
 ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
 for target in targets:
  projected=oi@target.pop('reference_amplitude')@ov.T;norm=float(np.linalg.norm(projected));anchor=projected.reshape(-1)/norm if norm>0 else projected.reshape(-1)
  pair=int(np.argmax(abs(anchor)));hole=int(ii[pair]);particle=int(aa[pair])
  data=dict(**target,hole=hole,particle=particle,reference_transition_projection_norm=norm,leading_projected_configuration_weight=float(anchor[pair]**2),source_hole_pi=float(pi[hole]),source_particle_pi=float(pi[particle]),source_hole_hetero_inplane=float(hetero[hole]),hole_dft=int(mapping.reference_indices[hole]),particle_dft=int(mapping.reference_indices[particle]),hole_mapping_overlap=float(mapping.overlaps[hole]),particle_mapping_overlap=float(mapping.overlaps[particle]),attempts=[])
  is_n=target['family']=='npi';seed_character=bool(pi[particle]>.65 and (pi[hole]<.35 and hetero[hole]>.25 if is_n else pi[hole]>.65))
  data['seed_character_qualified']=seed_character
  base_root=int(np.argmax(abs(anchor@original_vectors)));data.update(se_tda_ev=float(original_values[base_root]*EV),se_tda_state_overlap=float(abs(anchor@original_vectors[:,base_root])))
  if hybrid_values is not None:
   score=abs(anchor[ids]@hybrid_vectors);root=int(np.argmax(score));prior_score=abs(anchor[ids]@restricted_vectors);prior_root=int(np.argmax(prior_score))
   data.update(hybrid_tda_ev=float(hybrid_values[root]*EV),hybrid_tda_error_ev=float(hybrid_values[root]*EV-target['tbe_ev']),hybrid_tda_state_overlap=float(score[root]),hybrid_tda_margin=float(score[root]-np.max(np.delete(score,root),initial=0.)),same_space_se_tda_ev=float(restricted_values[prior_root]*EV),trusted_reference_transition_weight=float(np.sum(anchor[ids]**2)))
   data['tda_qualified']=bool(norm>=.7 and seed_character and score[root]>=.7 and data['hybrid_tda_margin']>=.05 and data['trusted_reference_transition_weight']>=.95)
  else:data['tda_qualified']=False
  if not seed_character or norm<PROTOCOL['minimum_projected_transition_norm']:
   data.update(roks_qualified=False,exclusion_reason='unqualified reference projection or chemical seed character');status['targets'].append(data);write(status_path,dict(**status,stage='state selection'));continue
  occ=np.array([s.alpha_occupations,s.beta_occupations]);occ[0,hole]=0;occ[0,particle]=1;triplet=np.array([np.maximum(*occ),np.minimum(*occ)])
  accepted=None
  for index in ([0] if index_zero else PROTOCOL['saddle_indices_to_try']):
   state_dir=out/(target['id']+f'_index_{index}');state_dir.mkdir(exist_ok=True);state_key=request+f'_{target["id"]}_{hole}_{particle}_{index}';checkpoint=state_dir/'converged.npz';t0=time.perf_counter()
   if checkpoint.exists():r=load_derivative_point(checkpoint,request_hash=state_key,coordinates_angstrom=xyz,require_gradient=False)
   else:
    c=s.coefficients.copy();latest=state_dir/'latest.npz'
    if latest.exists():
     with np.load(latest) as f:c=f['coefficients'];assert np.array_equal(f['occupations'],occ)
    pm=np.array([(c*o)@c.T for o in occ]);pt=np.array([(c*o)@c.T for o in triplet]);fm,em=ctx.fock_and_energy(pm);ft,et=ctx.fock_and_energy(pt)
    seed=ROKSSingletResult(c,occ,triplet,s.overlap,fm,ft,2*em-et,1.,0,False,reference_state=s,ground_energy_hartree=gs.energy_hartree,core_electrons=np.asarray(s.atomic_numbers)-np.asarray(calc._valence,dtype=int))
    def checkpoint_step(point):
     np.savez_compressed(latest,coefficients=point.point,occupations=occ,coordinates_angstrom=xyz);write(state_dir/'history.json',list(point.history))
    try:
     r=evaluate_gfn2_roks_singlet(calc,xyz,*occ,ground_state=ground,reference_result=seed,minimum_state_overlap=0.,gradient=False,orbital_solver='gmf',saddle_index=index,maximum_iterations=PROTOCOL['maximum_iterations_per_attempt'],maximum_step=.2,orbital_tolerance=1e-9,freeze_and_release=True,freeze_iterations=8,checkpoint=checkpoint_step)
     save_derivative_point(checkpoint,r,request_hash=state_key,coordinates_angstrom=xyz)
    except Exception as exc:
     attempt=dict(index=index,error=str(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-t0);data['attempts'].append(attempt);write(state_dir/'summary.json',attempt);print(name,target['id'],attempt['error'],flush=True);continue
   pp,nn,_=character(s,r.common_coefficients);csf,_=singlet_csf_overlap(replace(r,common_coefficients=s.coefficients).to_apoc(),r.to_apoc(),s.overlap)
   h=s.coefficients[:,occupied].T@s.overlap@r.common_coefficients[:,hole];p=s.coefficients[:,virtual].T@s.overlap@r.common_coefficients[:,particle];rankone=np.outer(h,p).reshape(-1);ranknorm=np.linalg.norm(rankone);transition=float(abs(anchor@rankone)/ranknorm) if ranknorm else 0.
   charok=bool(pp[particle]>.65 and (pp[hole]<.35 and nn[hole]>.25 if is_n else pp[hole]>.65))
   qualified=bool(charok and csf>=.7 and transition>=.7 and r.orbital_saddle_index==index and r.orbital_residual<=1e-9)
   attempt=dict(index=index,observed_index=r.orbital_saddle_index,converged=r.converged,residual=r.orbital_residual,canonical_csf_overlap=csf,reference_transition_overlap=transition,hole_pi=float(pp[hole]),particle_pi=float(pp[particle]),hole_hetero_inplane=float(nn[hole]),state_qualified=qualified,se_roks_ev=r.excitation_energy_hartree*EV,seconds=time.perf_counter()-t0)
   data['attempts'].append(attempt);write(state_dir/'summary.json',attempt);print(name,target['id'],attempt,flush=True)
   if qualified:accepted=r;break
  if accepted is None:data.update(roks_qualified=False,exclusion_reason='no stationary state passed identity checks')
  else:
   data.update(roks_state_qualified=True,se_roks_ev=accepted.excitation_energy_hartree*EV,se_roks_error_ev=accepted.excitation_energy_hartree*EV-target['tbe_ev'])
   try:
    if extent is None:raise ValueError(status['extent_error'])
    corrected=correct_roks_excitation(accepted,extent,hole,particle);prior=correct_roks_excitation(accepted,previous,hole,particle)
    data.update(roks_qualified=True,hybrid_roks_ev=corrected.excitation_energy_hartree*EV,hybrid_roks_error_ev=corrected.excitation_energy_hartree*EV-target['tbe_ev'],previous_ratio_roks_ev=prior.excitation_energy_hartree*EV,previous_ratio_roks_error_ev=prior.excitation_energy_hartree*EV-target['tbe_ev'],source_hole_radius_bohr=float(rs[hole]),reference_hole_radius_bohr=float(rd[hole]),extent_weight=float(extent.extent_scaling.weights[hole]),se_orbital_gap_ev=float((s.orbital_energies_hartree[particle]-s.orbital_energies_hartree[hole])*EV))
   except Exception as exc:data.update(roks_qualified=False,exclusion_reason='spectral mapping: '+str(exc))
  status['targets'].append(data);write(status_path,dict(**status,stage='state completed'));print('RESULT',name,target['id'],{k:data.get(k) for k in ['roks_qualified','hybrid_roks_ev','hybrid_roks_error_ev','exclusion_reason']},flush=True)
 write(out/f'result{suffix}.json',dict(**status,stage='complete'))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--molecule',choices=[x['name'] for x in PROTOCOL['cases']]);parser.add_argument('--index-zero',action='store_true');args=parser.parse_args()
 cases=[x for x in PROTOCOL['cases'] if args.molecule is None or x['name']==args.molecule]
 for case in cases:
  try:run(case,index_zero=args.index_zero)
  except Exception as exc:
   write(OUT/case['name']/('failure_index0.json' if args.index_zero else 'failure.json'),dict(host=HOST,error=str(exc),traceback=traceback.format_exc()));print('CASE ERROR',case['name'],str(exc),flush=True)
