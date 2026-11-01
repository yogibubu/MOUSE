"""Parameter-free bath relaxation of the DFT reference embedding.

Keep the native GFN2 AO chart and orbitals. Enrich its DFT image with the
orthogonal full-DFT complement at already qualified orbital energies, remove
the archived frozen core, and symmetrically orthonormalize. Rebuild F/Q,
charge/spin response, integrals and screening together in that image. No
excitation label, energy offset, fitted weight or state-family switch enters.
"""
from prepare_fresh import *
from scipy.linalg import null_space
from matrix_gfn2.nosi_surface import PreparedNOSIModel
import shutil

def radial(cohort,name,mode='bath'):
 original=PreparedNOSIModel(REPAIR/'prepared_models'/cohort/name/'model.json');old=prepare(cohort,name);s=old['s'];c=s.coefficients;no=old['no'];newcohort=('radial_' if mode=='bath' else 'subspace_')+cohort;out=HERE/('radial_embedding' if mode=='bath' else 'subspace_embedding')/cohort/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 data=CAM/'data'/newcohort;src=data/'predictions'/name;ref=data/'references'/name;src.mkdir(parents=True,exist_ok=True);ref.mkdir(parents=True,exist_ok=True)
 for filename in ['reference.npz','backend.chk']:link(old['ref']/filename,ref/filename)
 link(old['src']/'native_ground.npz',src/'native_ground.npz');link(old['src']/'native_ground.npz',src/'grounds/0/stationary.npz')
 folder=src/'full_response/anchor';folder.mkdir(parents=True,exist_ok=True)
 if not (folder/'embedding.npz').exists():
  with np.load(CAM/cohort/name/'CAM_full_reference/reference.npz') as z:cd=z['coefficients'].copy();ed=z['orbital_energies'].copy();od=z['occupations'].copy()
  nc=np.count_nonzero(od==2)-no;projection=cd.T@old['sd']@old['lift']@c;core=cd.T@old['sd']@old['cd'][:,:nc]
  assert np.max(abs(projection.T@projection-np.eye(len(c))))<1e-8
  assert np.max(abs(core.T@projection),initial=0)<1e-8
  complement=null_space(np.column_stack([projection,core]).T);hqq=complement.T@(ed[:,None]*complement);pole,u=eigh(hqq);coupling=u.T@complement.T@(ed[:,None]*projection)
  certificate=json.loads((NATIVE/cohort/name/'coherent_common/reference_certificate.json').read_text());good=np.array(certificate['qualified_mapping'],bool);mapping=np.array(certificate['mapping'],int);dressed=projection.copy();tail_norm=np.zeros(len(c));pole_distance=np.zeros(len(c))
  for i in np.flatnonzero(good):
   gap=ed[mapping[i]]-pole;pole_distance[i]=np.min(abs(gap),initial=np.inf);use=abs(gap)>1e-10;assert np.max(abs(coupling[~use,i]),initial=0)<1e-8,'Unresolved radial complement pole';tail=np.zeros(len(pole));tail[use]=coupling[use,i]/gap[use];dressed[:,i]+=complement@u@tail;tail_norm[i]=float(tail@tail)
  if mode=='subspace':
   selected=projection.copy()
   for i in np.flatnonzero(good):selected[:,i]=np.eye(len(ed))[:,mapping[i]]
   selected-=core@(core.T@selected)
   projected=projection.T@selected;left,singular,right=np.linalg.svd(projected);assert singular.min()>1e-7,'Unresolved selected valence subspace'
   dressed=selected@np.linalg.inv(projected)
   assert np.max(abs(projection.T@dressed-np.eye(len(c))))<1e-8
   tail_norm=np.sum((dressed-projection)**2,axis=0)
  gram=dressed.T@dressed;ev,uv=eigh(gram);assert ev.min()>1e-8,'Unresolved dressed valence chart';dressed=dressed@((uv*ev**-.5)@uv.T);lift=cd@dressed@c.T@s.overlap;assert np.max(abs(lift.T@old['sd']@lift-s.overlap))<1e-8;assert np.max(abs(old['cd'][:,:nc].T@old['sd']@lift),initial=0)<1e-8
  np.savez_compressed(folder/'embedding.npz',embedding=lift)
  write(out/'embedding_certificate.json',dict(parent_model=original.fingerprint,native_orbitals_preserved=True,qualified=good,mapping=mapping,bath_norm_squared=tail_norm,minimum_complement_pole_distance=pole_distance,symmetric_orthonormalization_eigenvalues=ev,gram_error=float(np.max(abs(lift.T@old['sd']@lift-s.overlap))),frozen_core_error=float(np.max(abs(old['cd'][:,:nc].T@old['sd']@lift),initial=0)),parameters_fitted=0,weights=[.5,.5],lambda_exchange=.9,energy_labels_read=False,method=mode+' complement dressing followed by symmetric orthonormalization; all response and screening rebuilt consistently'))
 with np.load(folder/'embedding.npz') as z:lift=z['embedding'].copy()
 if not (folder/'projected_eri.npz').exists():np.savez_compressed(folder/'projected_eri.npz',eri=ao2mo.kernel(old['mol'],(lift,)*4,compact=False).reshape((len(c),)*4))
 # This is also the exact old Hamiltonian for a solver continuation if needed.
 link(NATIVE/cohort/name/'coherent_common/model.npz',folder/'model.npz');link(folder/'model.npz',src/'qualified_spectral_response/anchor/model.npz')
 # Reuse the completed full DFT reference. Only the compact embedded SCF,
 # response quadrature and retained interactions change with the image.
 target=CAM/newcohort/name;target.mkdir(parents=True,exist_ok=True)
 for filename in ['reference.npz','backend.chk','certificate.json']:
  source=CAM/cohort/name/'CAM_full_reference'/filename
  if source.exists():link(source,target/'CAM_full_reference'/filename)
 xyzfile=out/'anchor.xyz';symbols=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers];xyzfile.write_text(str(len(symbols))+'\nUnchanged native anchor, Angstrom\n'+''.join(f'{symbol} {x[0]:.12f} {x[1]:.12f} {x[2]:.12f}\n' for symbol,x in zip(symbols,old['xyz'])))
 for identity,coefficients in zip(original.state_ids,original.arrays['states']):
  np.savez_compressed(src/('native_seed_'+identity+'.npz'),coefficients=coefficients)
  initial(name,identity,xyzfile,newcohort,original.path)
 run(name,original.state_ids[0],xyzfile,newcohort,original.path)
 write(out/'result.json',dict(cohort=newcohort,molecule=name,parent_model=original.fingerprint,prepared_model=str(REPAIR/'prepared_models'/newcohort/name/'model.json'),predictions=json.loads((REPAIR/'even_sector_5'/newcohort/name/'result.json').read_text())['primary']['predictions'],same_native_orbitals=True,all_response_and_interaction_rebuilt=True,no_fitted_excitation_shift=True))
 print(name,'consistent radial embedding complete',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');p.add_argument('--mode',choices=['bath','subspace'],default='bath');a=p.parse_args()
 with threadpool_limits(2):radial(a.cohort,a.name,a.mode)
