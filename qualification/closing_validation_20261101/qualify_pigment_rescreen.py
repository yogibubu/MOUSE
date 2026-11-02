"""Recover an archived physical CI matrix using cached native functional values."""
from run_pigment_cid import *
from matrix_gfn2._nosi_rescreen import reassemble
p=argparse.ArgumentParser();p.add_argument('name',nargs='?',default='Beta_carotene');p.add_argument('--output',default='pigment_rescreen_qualification.json');a=p.parse_args()
with threadpool_limits(2):
 folder=HERE/f'pigment_cid/{a.name}';r=json.loads((folder/'result.json').read_text());model,state=load_state_only(a.name);no=state.electron_count//2;n=len(state.overlap)
 v,u=eigh(state.overlap);sr=(u*v**.5)@u.T;frame=TransportedOperators(state,state.coordinates_bohr,np.zeros((0,n,n))).u;y=frame.T@sr@state.coefficients;ys=[]
 for ordinal in r['retained_ROKS_ordinals']:
  with np.load(HERE/f'pigment_roks/{a.name}/state_{ordinal}/checkpoint.npz') as z:ys.append(frame.T@sr@z['coefficients'])
 basis=scalar_basis(y,ys,r['auxiliary_records'],no)
 with np.load(folder/'matrices.npz') as z:old_h=z['h'];old_s=z['s']
 elements={}
 screen=screening_path(a.name,5);h=np.load(screen/'h.npy');w=np.load(screen/'screened_pair.npy',mmap_mode='r').reshape((n,)*4)
 hh,ss,certificate=reassemble(y,basis,r['auxiliary_records'],1+2*len(ys),h,w,old_h,old_s,elements=elements,checkpoint=lambda *a,**k:None,coulomb_factors=(np.load(screen/'lp.npy',mmap_mode='r'),np.load(screen/'weighted_lp.npy',mmap_mode='r')))
 eh=float(np.max(abs(hh-old_h)));es=float(np.max(abs(ss-old_s)));assert eh<1e-8 and es<1e-10,(eh,es)
 certificate.update(host=socket.getfqdn(),Hamiltonian_error_hartree=eh,metric_error=es,reference='Archived full native-functional CI matrix',DFT_imports=0,new_SCC_calls=0)
 write(HERE/a.output,certificate);print(json.dumps(certificate,indent=2),flush=True)
