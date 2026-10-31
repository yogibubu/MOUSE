"""Six frozen, label-segregated states; all attempts retained on verified Oracle."""
from common import *
import argparse,fcntl,traceback
from matrix_gfn2 import evaluate_gfn2_roks_singlet
from matrix_gfn2.roks import ROKSSingletResult
from matrix_gfn2.multicsf import MultiCSFObjective
from matrix_gfn2.orbital_saddle import generalized_mode_following
from scipy.linalg import expm

def reference(name):
 root=OUT/os.environ.get('MOUSE_HOLDOUT_DIRECTORY','independent');manifest=json.loads((root/'manifest.json').read_text());rows=[r for r in manifest['states'] if r['molecule']==name];out=root/'references'/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return out
 xyz=root/'geometries'/f"{rows[0]['id']}.xyz";assert sha(xyz)==rows[0]['geometry_sha256'];lines=xyz.read_text().splitlines();n=int(lines[0])
 mol=gto.M(atom='\n'.join(lines[2:2+n]),unit='Angstrom',basis='aug-cc-pVDZ',symmetry=True,verbose=4,output=str(out/'pyscf.log'),max_memory=24000)
 mf=dft.RKS(mol);mf.xc='PBE0';mf.grids.level=3;mf.small_rho_cutoff=0;mf.conv_tol=1e-10;mf.max_cycle=150;mf.chkfile=str(out/'backend.chk');cp=out/'reference.npz';start=time.monotonic()
 if cp.exists():
  with np.load(cp) as a:mf.mo_coeff=a['coefficients'].copy();mf.mo_occ=a['occupations'].copy();mf.mo_energy=a['orbital_energies'].copy();mf.e_tot=float(a['energy_hartree']);mf.converged=True
 else:
  dm=mf.init_guess_by_chkfile(str(out/'backend.chk')) if (out/'backend.chk').exists() else None
  mf.kernel(dm0=dm)
  if not mf.converged:old=mf;mf=mf.newton();mf.kernel(old.mo_coeff,old.mo_occ)
  assert mf.converged,'DFT SCF not converged'
  np.savez_compressed(cp,coefficients=mf.mo_coeff,occupations=mf.mo_occ,orbital_energies=mf.mo_energy,overlap=mf.get_ovlp(),coordinates_bohr=mol.atom_coords(),atomic_numbers=mol.atom_charges(),energy_hartree=mf.e_tot)
  write(out/'ground_cost.json',dict(seconds=time.monotonic()-start,nao=mol.nao_nr()))
 from pyscf.tdscf import _lr_eig
 from pyscf import symm
 _lr_eig.MAX_SPACE_INC=None
 spectra=[]
 for row in rows:
  dest=out/f"{row['id']}.npz"
  if dest.exists():continue
  symmetry=row['state_label'].removeprefix('^1').replace('_','').replace('{','').replace('}','').replace("''",'"')
  # PySCF reduces C3v to its Abelian Cs subgroup. A1 correlates
  # uniquely with A-prime; this changes the representation, not the state.
  if mol.topgroup=='C3v' and mol.groupname=='Cs' and symmetry=='A1':
   write(out/f"{row['id']}_symmetry_correlation.json",dict(source_group='C3v',source_irrep='A1',working_group='Cs',working_irrep="A'",reason='exact group-subgroup correlation before response evaluation'))
   symmetry="A'"
  td=mf.TDA();td.wfnsym=symmetry;td.nstates=max(2,row['ordinal_same_symmetry']+1);td.conv_tol=1e-7;td.max_cycle=150;td.lindep=1e-20;started=time.monotonic();td.kernel()
  assert np.all(td.converged),'Reference response not converged'
  x=np.array([a for a,b in td.xy]);x/=np.linalg.norm(x.reshape(len(x),-1),axis=1)[:,None,None]
  vind,_=td.gen_vind(mf);residuals=np.linalg.norm(vind(x.reshape(len(x),-1))-td.e[:,None]*x.reshape(len(x),-1),axis=1);assert max(residuals)<1.1e-7
  np.savez_compressed(dest,amplitudes=x,energies_hartree=td.e,converged=td.converged,residuals=residuals,symmetry=symmetry,ordinal=row['ordinal_same_symmetry'])
  write(out/f"{row['id']}_cost.json",dict(seconds=time.monotonic()-started,roots=len(x),symmetry=symmetry))
 write(out/'result.json',dict(status='reference_ready',molecule=name,states=[r['id'] for r in rows],reference_sha256=sha(cp),numerical_labels_read=False));return out

def optimize_parent(obj,c,out):
 latest=out/'latest.npz';out.mkdir(parents=True,exist_ok=True)
 if (out/'converged.npz').exists():
  with np.load(out/'converged.npz') as a:return a['coefficients'].copy()
 if latest.exists():
  with np.load(latest) as a:c=a['coefficients'].copy()
 def retract(point,step):
  k=np.zeros_like(point);k[obj.left,obj.right]=step;k[obj.right,obj.left]=-step;return point@expm(k)
 def guard(old,new):
  obj.evaluate(old,False);ci=obj.cache['ci'].copy();obj.evaluate(new,False)
  return obj.fidelity(new)[0]**2>.5 and obj.cache['coefficient_margin']>=.05 and abs(obj.signed_overlap(old,ci,new,obj.cache['ci']))>=.9
 def checkpoint(r):np.savez_compressed(latest,coefficients=r.point);write(out/'history.json',list(r.history))
 active=np.r_[np.arange(obj.rank),obj.no+np.arange(obj.rank)];free=np.flatnonzero(~(np.isin(obj.left,active)|np.isin(obj.right,active)))
 if len(free) and not latest.exists():
  def frozen(point,curvature=True):
   e,g,h=obj.evaluate(point,curvature);return e,g[free],None if not curvature else h[np.ix_(free,free)]
  def fr(point,step):v=np.zeros(obj.dimension);v[free]=step;return retract(point,v)
  solved=generalized_mode_following(frozen,fr,c,saddle_index=0,gradient_tolerance=1e-9,maximum_iterations=80,maximum_step=.08,trial_guard=guard);c=solved.point
 e,g,h=obj.evaluate(c,True);index=int(np.count_nonzero(np.linalg.eigvalsh(h)<-1e-7))
 # The initial curvature count need not identify the final local branch.
 # Reassess only at bounded phase boundaries; state guards stay unchanged.
 phases=[]
 for phase in range(3):
  solved=generalized_mode_following(obj.evaluate,retract,c,saddle_index=index,gradient_tolerance=1e-9,maximum_iterations=50,maximum_step=.08,trial_guard=guard,checkpoint=checkpoint)
  phases.append(dict(phase=phase,target_index=index,history=list(solved.history)))
  write(out/'solver_phases.json',phases)
  if solved.converged:break
  c=solved.point;e,g,h=obj.evaluate(c,True);index=int(np.count_nonzero(np.linalg.eigvalsh(h)<-1e-7))
 assert solved.converged,'Multi-CSF parent unresolved; checkpoint retained'
 np.savez_compressed(out/'converged.npz',coefficients=solved.point);return solved.point

def predict(name):
 root=OUT/os.environ.get('MOUSE_HOLDOUT_DIRECTORY','independent');rows=[r for r in json.loads((root/'manifest.json').read_text())['states'] if r['molecule']==name];refdir=reference(name);out=root/'predictions'/name;out.mkdir(parents=True,exist_ok=True)
 with np.load(refdir/'reference.npz') as a:xyz=a['coordinates_bohr'].copy()/ANGSTROM_TO_BOHR;z=a['atomic_numbers'].copy();dc=a['coefficients'].copy();dn=int(np.count_nonzero(a['occupations']==2))
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(k)] for k in z],temperature_kelvin=0.)
 gp=out/'native_ground.npz'
 if gp.exists():raw=loadpoint(gp)
 else:
  from matrix_gfn2.response_reference import stationary_response_reference
  initial,_=stationary_response_reference(calc,xyz);g=evaluate_restricted_ground(calc,xyz,template=initial.orbital_state,gradient=False);raw=g.reference;save_derivative_point(gp,raw,request_hash='independent:native',coordinates_angstrom=xyz);write(out/'native_certificate.json',dict(minimum_curvature=g.minimum_orbital_curvature_hartree,index=g.orbital_saddle_index))
 s=canonical_state(raw);raw=replace(raw,reference_state=s);ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s);small,t=source_molecule(s);mol=scf.chkfile.load_mol(str(refdir/'backend.chk'));cross=(gto.intor_cross('int1e_ovlp',small,mol).T@t).T;overlap=s.coefficients.T@cross@dc;no=s.electron_count//2
 eligible=[]
 for row in rows:
  dest=out/row['id'];dest.mkdir(exist_ok=True)
  if (dest/'result.json').exists():continue
  with np.load(refdir/f"{row['id']}.npz") as a:dx=a['amplitudes'][row['ordinal_same_symmetry']-1].copy()
  amp=overlap[:no,:dn]@dx@overlap[no:,dn:].T;pn=float(np.linalg.norm(amp));amp/=max(pn,1e-300);w=np.linalg.svd(amp,compute_uv=False)**2;w/=w.sum();rank=int(np.floor(1/(w@w)+.5))
  write(dest/'projection.json',dict(norm=pn,weights=w,rank=rank,effective_pairs=float(1/(w@w))))
  if pn<.7 or rank>4:
   write(dest/'result.json',dict(id=row['id'],molecule=name,family=row['family'],status='not_representable',reason='compact projection below .7 or effective rank exceeds four',projection_norm=pn,rank=rank));continue
  eligible.append((row,amp,w,rank))
 if not eligible:
  write(out/'pass_complete.json',dict(status='attempt_pass_complete',molecule=name,labels_read=False));return
 data,erifile=calibrate(out,refdir,s,ctx);responses=[FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,a,sha(out/profile/'anchor/model.npz')) for a,profile in zip(data,['full_response','qualified_spectral_response'])]
 cert=json.loads((out/'native_certificate.json').read_text());grounds=[RestrictedGroundResult(raw,cert['minimum_curvature'],cert['index'])]+[ground(out/f'grounds/{i}',calc,xyz,s,r) for i,r in enumerate(responses)]
 with np.load(erifile) as a:eri=a['eri'].copy()
 for row,amp,weights,rank in eligible:
  dest=out/row['id'];parents=[];refs=[];amps=[];started=time.monotonic()
  try:
   for i,(response,g) in enumerate(zip(responses,grounds[1:])):
    c0=g.coefficients[0];q=c0[:,:no].T@s.overlap@s.coefficients[:,:no]@amp@s.coefficients[:,no:].T@s.overlap@c0[:,no:];q/=np.linalg.norm(q);refs.append(c0);amps.append(q);u,sv,vh=np.linalg.svd(q,full_matrices=True);c=np.column_stack([c0[:,:no]@u,c0[:,no:]@vh.T])
    context=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=response)
    if rank>=2:
     obj=MultiCSFObjective(context,eri,no,rank,c0,q)
     if i==1:
      from parent_homotopy import follow_parent
      c=follow_parent(calc,xyz,s,responses,eri,no,rank,c0,q,parents[0],dest/'homotopy_Q')
      (dest/f'parent_{i}').mkdir(exist_ok=True);np.savez_compressed(dest/f'parent_{i}/converged.npz',coefficients=c)
      parents.append(c)
     else:parents.append(optimize_parent(obj,c,dest/f'parent_{i}'))
    else:
     # Keep the original single-CSF energy and relative-pair manifold.
     occ=np.array([s.alpha_occupations,s.beta_occupations]);occ[0,0]=0;occ[0,no]=1;trip=np.array([np.maximum(*occ),np.minimum(*occ)])
     fm,em=context.fock_and_energy(np.array([(c*o)@c.T for o in occ]));ft,et=context.fock_and_energy(np.array([(c*o)@c.T for o in trip]))
     seed=ROKSSingletResult(c,occ,trip,s.overlap,fm,ft,2*em-et,1.,0,False,ground_energy_hartree=g.energy_hartree,reference_state=s,core_electrons=s.atomic_numbers-np.asarray(calc._valence),spin_reference_fingerprint=response.fingerprint)
     solved=None
     for index in range(6):
      cp=dest/f'parent_{i}_index_{index}.npz'
      try:
       if cp.exists():candidate=loadpoint(cp)
       else:candidate=evaluate_gfn2_roks_singlet(calc,xyz,*occ,reference_result=seed,spin_reference=response,reference_policy='geometry_only',minimum_state_overlap=0.,orbital_solver='gmf',saddle_index=index,orbital_tolerance=1e-9,maximum_iterations=120,maximum_step=.2,freeze_and_release=True,freeze_iterations=8);save_derivative_point(cp,candidate,request_hash='independent:parent',coordinates_angstrom=xyz)
       hh=c0[:,:no].T@s.overlap@candidate.common_coefficients[:,0];pp=c0[:,no:].T@s.overlap@candidate.common_coefficients[:,no];score=abs(hh@q@pp)/max(np.linalg.norm(hh)*np.linalg.norm(pp),1e-300)
       if candidate.orbital_residual<=1e-9 and score>=.7:solved=candidate;break
      except Exception:write(dest/f'parent_{i}_index_{index}_failure.json',dict(traceback=traceback.format_exc()))
     if solved is None:raise RuntimeError('Single-CSF parent not qualified')
     parents.append(solved)
   if rank>=2:
    bundle=dest/'portable_model'
    if not (bundle/'model.json').exists():write_multicsf_model(bundle,template=raw,responses=responses,eri=eri,initial_coefficients=parents,reference_coefficients=refs,reference_transitions=amps,transition_weights=weights,grounds=grounds,provenance=dict(independent_manifest_sha256=sha(root/'manifest.json'),state=row['id'],labels_read=False))
    model=MultiCSFModel(bundle/'model.json');point=evaluate_multicsf_surface(calc,xyz,model,compute_hessian=False);save_multicsf_result(dest/'accepted',point,model);gap=.5*sum(point.parent_gaps_hartree)*EV;residual=point.orbital_residual
   else:
    from matrix_gfn2.relative_pair_roks import evaluate_relative_pair_roks,save_relative_pair_roks
    joint=evaluate_relative_pair_roks(calc,xyz,responses,initial_components=parents,anchor_components=parents,minimum_state_overlap=.7,maximum_iterations=150)
    save_relative_pair_roks(dest/'stationary.npz',joint,request_hash='independent:relative')
    gap=.5*sum(p.energy_hartree-g.energy_hartree for p,g in zip(joint.components,grounds[1:]))*EV;residual=joint.orbital_residual
   assert gap>0 and residual<=1e-8,'Nonpositive gap or unresolved stationary state'
   write(dest/'result.json',dict(id=row['id'],molecule=name,family=row['family'],status='matched',rank=rank,excitation_ev=gap,residual=residual,projection_norm=float(json.loads((dest/'projection.json').read_text())['norm']),seconds=time.monotonic()-started,labels_read=False))
  except Exception:
   write(dest/'failure.json',dict(traceback=traceback.format_exc(),checkpoints_preserved=True))
   write(dest/'result.json',dict(id=row['id'],molecule=name,family=row['family'],status='nonconverged',rank=rank,reason='See retained stationary-search failure record',labels_read=False))
 write(out/'pass_complete.json',dict(status='attempt_pass_complete',molecule=name,labels_read=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('name');args=a.parse_args();from science_slots import science_locks;locks=science_locks()
 write(OUT/os.environ.get('MOUSE_HOLDOUT_DIRECTORY','independent')/f'executions/{args.name}.json',dict(script_sha256=sha(__file__),common_sha256=sha(Path(__file__).parent/'common.py'),protocol_sha256=sha(OUT/'protocol.json'),manifest_sha256=sha(OUT/os.environ.get('MOUSE_HOLDOUT_DIRECTORY','independent')/'manifest.json'),labels_read=False))
 with threadpool_limits():predict(args.name)
