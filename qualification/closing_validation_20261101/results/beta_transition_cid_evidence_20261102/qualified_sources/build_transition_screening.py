"""Constrain native screening by the actual explicit singlet transition space."""
from run_pigment_cid import *
import fcntl
from scipy.linalg import cho_factor,cho_solve
from pigment_block_matmul import product
from matrix_gfn2._nosi_slater import transition_density
from matrix_gfn2._transition_projector import constrained_vertices,canonical_frontier_indices

def run(name,frontier):
 out=screening_path(name,frontier,'coherent_transitions');out.mkdir(parents=True,exist_ok=True)
 if (out/'certificate.json').exists():return
 start=time.monotonic();model,s=load_state_only(name);n=len(s.overlap);no=s.electron_count//2;original=screening_path(name,5);prior=json.loads((original/'certificate.json').read_text());eta=prior['bath_efficiency_eta']
 with np.load(HERE/'pigment_full_response'/name/'ri_xc_chart/field.npz') as z:levels=z['common_levels'].copy();fg=z['ground_fock'].copy()
 gaps=levels[no:][None,:]-levels[:no,None];assert gaps.min()>0
 source=native_parent(name,frontier,frontier,'coherent_transitions')
 if source is None:
  # Screening only needs the retained orbitals and the explicit singles span,
  # not a completed CID calculation at the screening frontier.
  source=HERE/'pigment_cid'/name
 r=json.loads((source/'result.json').read_text());ys=[]
 for ordinal in r['retained_ROKS_ordinals']:
  with np.load(HERE/f'pigment_roks/{name}/state_{ordinal}/checkpoint.npz') as z:ys.append(s.coefficients.T@s.overlap@z['coefficients'])
 basis=scalar_basis(np.eye(n),ys,[],no);transitions=np.column_stack([(transition_density(basis[0],b)[no:,:no].T/np.sqrt(2)).ravel() for b in basis[1:]])
 canonical=canonical_frontier_indices(transitions,no,n-no,frontier)
 common=HERE/'pigment_screening_transition'/name/'common';common.mkdir(parents=True,exist_ok=True);full=common/'native_unconstrained.npy';marker=common/'native_complete.json'
 if not marker.exists():
  with np.load(HERE/'pigment_dft_reference'/name/'CAM/reference.npz') as z:nc=int(np.count_nonzero(z['occupations']==2)-no)
  native=np.load(HERE/'pigment_integrals'/name/'coulomb_factors.npy',mmap_mode='r')[:,nc:,nc:];na=len(native);b=np.lib.format.open_memmap(full,mode='w+',shape=(na,no*(n-no)),dtype=float)
  for lo in range(0,na,32):
   hi=min(na,lo+32);a=np.asarray(native[lo:hi]);b[lo:hi]=2*(s.coefficients[:,:no].T@a@s.coefficients[:,no:]).reshape(hi-lo,-1)/np.sqrt(gaps.ravel())[None,:]
  b.flush();write(marker,dict(model_sha256=hashlib.sha256((model/'manifest.json').read_bytes()).hexdigest(),complete=True))
 assert json.loads(marker.read_text())['model_sha256']==hashlib.sha256((model/'manifest.json').read_bytes()).hexdigest()
 b=np.load(full,mmap_mode='r');projected,details=constrained_vertices(b,gaps,canonical,transitions);np.save(out/'native_polarization.npy',projected);np.save(out/'native_features.npy',projected.T);del projected
 for filename in ['lp.npy','bath_polarization.npy']:
  if not (out/filename).exists():(out/filename).symlink_to(original/filename)
 # Reuse the already verified original native Gram, not a new bath calculation.
 legacy=HERE/'pigment_screening_expansion'/name/'frontier_16/previous_native_gram.npy'
 if legacy.exists():old=np.load(legacy)
 else:
  old_path=common/'original_native_features.npy'
  if not old_path.exists():np.save(old_path,np.load(original/'native_polarization.npy',mmap_mode='r').T)
  old=np.array(product(old_path,old_path,common/'original_native_gram.npy'))
 new=product(out/'native_features.npy',out/'native_features.npy',out/'native_gram.npy');d=np.array(np.load(original/'dielectric_polarization.npy',mmap_mode='r'))-old+new;d=.5*(d+d.T);np.save(out/'dielectric_polarization.npy',d)
 bp=np.load(out/'native_polarization.npy',mmap_mode='r');bb=np.load(out/'bath_polarization.npy',mmap_mode='r');rng=np.random.default_rng(821+frontier);errors=[]
 for unused in range(3):
  v=rng.normal(size=len(d));v/=np.linalg.norm(v);expected=float(np.sum((v@bp)**2)+eta*np.sum((v@bb)**2));errors.append(abs(float(v@d@v)-expected)/max(1.,abs(expected)))
 assert max(errors)<1e-9,errors
 dielectric=d.copy();dielectric.flat[::len(d)+1]+=1;factor=cho_factor(dielectric,lower=True);lp=np.load(out/'lp.npy',mmap_mode='r');np.save(out/'weighted_lp.npy',cho_solve(factor,lp));screen=product(out/'lp.npy',out/'weighted_lp.npy',out/'screened_pair.npy')
 for lo in range(0,n*n,512):
  hi=min(n*n,lo+512);tile=.5*(screen[lo:hi]+screen[:,lo:hi].T);screen[lo:hi]=tile;screen[:,lo:hi]=tile.T
 screen.flush();v,u=eigh(s.overlap);sr=(u*np.sqrt(v))@u.T;si=(u/np.sqrt(v))@u.T;pg=2*sr@s.coefficients[:,:no]@s.coefficients[:,:no].T@sr;w=screen.reshape((n,)*4);h=si@fg@si-np.einsum('pqrs,rs->pq',w,pg,optimize=True)+.5*np.einsum('prqs,rs->pq',w,pg,optimize=True);np.save(out/'h.npy',h)
 report=dict(prior);report.update(host=socket.getfqdn(),active_frontier=frontier,screening_mode='coherent_transitions',active_exclusion=details,quadratic_form_errors=errors,wall_seconds=time.monotonic()-start,definition='Exact positive constrained native resolvent in actual canonical-single plus optimized-CSF one-body transition span; no Cartesian product of hole and particle spans',bath_vertices_reused=True,native_functional_changed=False,parameters_retuned=False,DFT_imports=0,new_SCC_calls=0);write(out/'certificate.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--frontier',type=int,required=True);p.add_argument('--threads',type=int,default=8);a=p.parse_args()
 lockdir=screening_path(a.name,a.frontier,'coherent_transitions');lockdir.mkdir(parents=True,exist_ok=True)
 with (lockdir/'.build.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with threadpool_limits(a.threads):run(a.name,a.frontier)
