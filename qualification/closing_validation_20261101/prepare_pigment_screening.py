"""Explicit orthogonal-bath eta=0.60 screening for prepared pigment states."""
from pigment_compact_reference import HERE,load_state
from pigment_reference_embedding import write,sha,power
from pigment_block_matmul import product
from pathlib import Path
from scipy.linalg import null_space,eigh,cho_factor,cho_solve
from pyscf import scf,df,lib
from threadpoolctl import threadpool_limits
import numpy as np,json,time,os,argparse,socket

def sector(columns,n):
 u,s,vh=np.linalg.svd(np.column_stack(columns),full_matrices=False);return u[:,s>1e-10]

def run(name):
 out=HERE/'pigment_screening'/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'certificate.json').exists():return
 start=time.monotonic();s=load_state(name);c=s.coefficients;n=len(c);no=s.electron_count//2;sr=power(s.overlap,.5);si=power(s.overlap,-.5)
 with np.load(HERE/'pigment_embedding'/name/'embedding.npz') as z:lift=z['embedding'].copy()
 with np.load(HERE/'pigment_dft_reference'/name/'CAM/reference.npz') as z:cd=z['coefficients'].copy();ed=z['orbital_energies'].copy();od=z['occupations'].copy();sd=z['overlap'].copy()
 with np.load(HERE/'pigment_dft_reference'/name/'PBE0/reference.npz') as z:cp=z['coefficients'].copy()
 with np.load(HERE/'pigment_full_response'/name/'ri_xc_chart/field.npz') as z:en=z['common_levels'].copy();fg=z['ground_fock'].copy()
 nc=int(np.count_nonzero(od==2)-no);projection=cd.T@sd@lift@si;core=cd.T@sd@cp[:,:nc];assert np.max(abs(projection.T@projection-np.eye(n)))<1e-8 and np.max(abs(core.T@projection),initial=0.)<1e-8
 bath=null_space(np.column_stack([projection,core]).T);eb,ub=eigh(bath.T@(ed[:,None]*bath));bath=cd@bath@ub;gauge=float(np.mean(en[:no])-np.mean(ed[nc:nc+no]));eb+=gauge
 native_gap=en[no:][None,:]-en[:no,None];bath_gap=eb[None,:]-en[:no,None];write(out/'gap_diagnostic.json',dict(minimum_native_gap=float(native_gap.min()),minimum_bath_gap=float(bath_gap.min()),bath_virtuals=len(eb),eta=.6))
 assert min(native_gap.min(),bath_gap.min())>1e-7,'Unresolved occupied component in the bath; do not use negative denominators'
 hs=[np.eye(no)[:,i] for i in range(max(0,no-5),no)];ps=[np.eye(n-no)[:,i] for i in range(min(5,n-no))]
 report=json.loads((HERE/'pigment_roks'/name/'result.json').read_text())
 for row in report['states']:
  with np.load(HERE/'pigment_roks'/name/f"state_{row['ordinal']}"/'checkpoint.npz') as z:cc=z['coefficients'].copy()
  hs.append(c[:,:no].T@s.overlap@cc[:,0]);ps.append(c[:,no:].T@s.overlap@cc[:,no])
 ah=sector(hs,no);ap=sector(ps,n-no);fraction=np.clip(1-np.sum(ah**2,axis=1)[:,None]*np.sum(ap**2,axis=1)[None,:],0,1)
 root=HERE/'pigment_integrals'/name;native=np.load(root/'coulomb_factors.npy',mmap_mode='r')[:,nc:,nc:];na=len(native);lp_path=out/'lp.npy';bn_path=out/'native_polarization.npy';bb_path=out/'bath_polarization.npy';complete=out/'vertices_complete.json';co=lift@c[:,:no]
 if not complete.exists():
  lp=np.load(lp_path,mmap_mode='r+') if lp_path.exists() else np.lib.format.open_memmap(lp_path,mode='w+',shape=(na,n*n),dtype=float);bn=np.load(bn_path,mmap_mode='r+') if bn_path.exists() else np.lib.format.open_memmap(bn_path,mode='w+',shape=(na,no*(n-no)),dtype=float);bb=np.load(bb_path,mmap_mode='r+') if bb_path.exists() else np.lib.format.open_memmap(bb_path,mode='w+',shape=(na,no*len(eb)),dtype=float);progress=out/'vertices_progress.json';completed=json.loads(progress.read_text())['completed'] if progress.exists() else 0
  mol=scf.chkfile.load_mol(str(HERE/'pigment_dft_reference'/name/'PBE0/backend.chk'));full=df.DF(mol);full.auxbasis='def2-universal-jkfit';full._cderi=str(HERE/'pigment_dft_reference'/name/'PBE0/df_integrals.h5');offset=0
  for block in full.loop(blksize=16):
   hi=offset+len(block)
   if hi>completed:
    a=np.asarray(native[offset:hi]);lp[offset:hi]=(si@a@si).reshape(len(a),-1);ln=(c[:,:no].T@a@c[:,no:]).reshape(len(a),-1);bn[offset:hi]=2*ln*np.sqrt(fraction.ravel()/native_gap.ravel())[None,:];ao=lib.unpack_tril(block);mixed=np.einsum('mi,Lmn,nj->Lij',co,ao,bath,optimize=True);bb[offset:hi]=2*mixed.reshape(len(a),-1)/np.sqrt(bath_gap.ravel())[None,:];lp.flush();bn.flush();bb.flush();write(progress,dict(completed=hi,total=na));print(name,'screened vertices',hi,'/',na,flush=True)
   offset=hi
  assert offset==na;write(complete,dict(complete=True))
 lp=np.load(lp_path,mmap_mode='r');bn=np.load(bn_path,mmap_mode='r');bb=np.load(bb_path,mmap_mode='r');polarization=out/'polarization_features.npy'
 if not polarization.exists():
  features=np.lib.format.open_memmap(polarization,mode='w+',shape=(bn.shape[1]+bb.shape[1],na),dtype=float);features[:bn.shape[1]]=bn.T;features[bn.shape[1]:]=np.sqrt(.6)*bb.T;features.flush()
 d=product(polarization,polarization,out/'dielectric_polarization.npy');dielectric=np.array(d);dielectric.flat[::na+1]+=1;factor=cho_factor(dielectric,lower=True);weighted=out/'weighted_lp.npy'
 if not weighted.exists():np.save(weighted,cho_solve(factor,lp))
 screened=product(lp_path,weighted,out/'screened_pair.npy')
 for lo in range(0,n*n,512):
  hi=min(n*n,lo+512);tile=.5*(screened[lo:hi]+screened[:,lo:hi].T);screened[lo:hi]=tile;screened[:,lo:hi]=tile.T
 screened.flush();w=screened.reshape((n,)*4);pg=2*sr@c[:,:no]@c[:,:no].T@sr;h=si@fg@si-np.einsum('pqrs,rs->pq',w,pg,optimize=True)+.5*np.einsum('prqs,rs->pq',w,pg,optimize=True);np.save(out/'h.npy',h)
 write(out/'certificate.json',dict(host=socket.getfqdn(),molecule=name,wall_seconds=time.monotonic()-start,bath_efficiency_eta=.6,eta_retuned=False,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,auxbasis='def2-universal-jkfit',auxiliary_functions=na,native_orbitals=n,bath_virtuals=len(eb),minimum_native_gap=float(native_gap.min()),minimum_bath_gap=float(bath_gap.min()),positive_dielectric_by_construction=True,retained_active_hole_rank=ah.shape[1],retained_active_particle_rank=ap.shape[1],CI_selection='Frozen original two-occupied/two-virtual frontier; even-sector augmentation only if nuclear reflection is certified',reference_energies_read=False))
 print(name,'screening complete',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--wait-for-roks',action='store_true');a=p.parse_args()
 while not (HERE/'pigment_roks'/a.name/'result.json').exists():
  if not a.wait_for_roks:raise RuntimeError('ROKS targets incomplete')
  time.sleep(20)
 with threadpool_limits(8):run(a.name)
