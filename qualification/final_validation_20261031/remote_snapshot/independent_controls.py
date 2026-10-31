"""Native/corrected GFN2-orbital TDA controls with coherent state assignment."""
from common import *
import fcntl,argparse

def run(name):
 root=OUT/'independent';manifest=json.loads((root/'manifest.json').read_text());rows=[r for r in manifest['states'] if r['molecule']==name];out=root/'predictions'/name;dest=root/'tda'/name;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return
 if not (out/'full_response/anchor/model.npz').exists():
  write(dest/'result.json',dict(molecule=name,rows=[dict(id=r['id'],status='not_representable',reason='same fixed compact projection gate; no qualified compact model') for r in rows]));return
 raw=loadpoint(out/'native_ground.npz');s=canonical_state(raw);xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR;calc=calculator(s);ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s);spectra=[]
 for index,profile in enumerate(['full_response','qualified_spectral_response']):
  cp=dest/f'parent_{index}.npz'
  if cp.exists():
   with np.load(cp) as a:spectra.append({k:a[k].copy() for k in a.files});continue
  file=out/profile/'anchor/model.npz'
  with np.load(file) as a:data={k:a[k].copy() for k in ['operators','charge','spin','linear','q0']}
  ref=FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,data,sha(file));g=ground(out/f'grounds/{index}',calc,xyz,s,ref);context=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=ref);e,c=eigh(g.fock[0],s.overlap);state=replace(s,coefficients=c,fock=g.fock[0],orbital_energies_hartree=e,total_energy_hartree=g.energy_hartree);kernel=context.density_response_kernel(state);no=s.electron_count//2;ii=np.repeat(np.arange(no),len(c)-no);aa=np.tile(np.arange(no,len(c)),no);ci=c[:,ii];ca=c[:,aa];columns=[];started=time.monotonic()
  for i,a in zip(ii,aa):
   p=np.outer(c[:,i],c[:,a]);f=kernel(.5*(p+p.T));columns.append(np.einsum('mi,mn,ni->i',ci,f,ca))
  matrix=np.diag(e[aa]-e[ii])+2*np.array(columns).T;assert np.max(abs(matrix-matrix.T))<1e-8;ee,vec=np.linalg.eigh(matrix);data=dict(coefficients=c,energies_hartree=ee,amplitudes=vec.T.reshape(len(ee),no,-1),matrix=matrix);np.savez_compressed(cp,**data);spectra.append(data);write(dest/f'parent_{index}_cost.json',dict(seconds=time.monotonic()-started,dimension=len(ee),orbitals='own corrected GFN2 ground orbitals'))
 with np.load(root/'references'/name/'reference.npz') as a:dc=a['coefficients'].copy();dn=int(np.count_nonzero(a['occupations']==2))
 mol=scf.chkfile.load_mol(str(root/'references'/name/'backend.chk'));small,t=source_molecule(s);cross=(gto.intor_cross('int1e_ovlp',small,mol).T@t).T;overlap=s.coefficients.T@cross@dc;no=s.electron_count//2;results=[]
 for row in rows:
  projection=json.loads((out/row['id']/'projection.json').read_text())
  if projection['norm']<.7 or projection['rank']>4:results.append(dict(id=row['id'],status='not_representable',projection_norm=projection['norm']));continue
  with np.load(root/'references'/name/f"{row['id']}.npz") as a:dx=a['amplitudes'][row['ordinal_same_symmetry']-1].copy()
  amp=overlap[:no,:dn]@dx@overlap[no:,dn:].T;amp/=np.linalg.norm(amp);matches=[]
  for index,data in enumerate(spectra):
   c=data['coefficients'];q=c[:,:no].T@s.overlap@s.coefficients[:,:no]@amp@s.coefficients[:,no:].T@s.overlap@c[:,no:];q/=np.linalg.norm(q);scores=abs(np.einsum('kij,ij->k',data['amplitudes'],q));scores[data['energies_hartree']<=0]=0;chosen=int(np.argmax(scores));margin=float(scores[chosen]-np.max(np.delete(scores,chosen),initial=0.));valid=scores[chosen]>=.7 and margin>=.05
   matches.append(dict(profile=['F','Q'][index],status='matched' if valid else 'completed_unassigned',root=chosen+1,coherent_transition_overlap=float(scores[chosen]),margin=margin,excitation_ev=float(data['energies_hartree'][chosen]*EV)))
  results.append(dict(id=row['id'],molecule=name,family=row['family'],status='matched' if all(r['status']=='matched' for r in matches) else 'completed_unassigned',parents=matches,weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0))
 write(dest/'result.json',dict(molecule=name,rows=results,labels_read=False,assignment='maximum coherent projected DFT transition overlap, score>=.7 and margin>=.05; no reference excitation energy used'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');args=p.parse_args();lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 with threadpool_limits():run(args.name)
