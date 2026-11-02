"""Check exact batched retained-CSF contractions against a saved scalar journal."""
from run_pigment_cid import *
from matrix_gfn2._nosi_slater import Hamiltonian,schur
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--frontier',type=int,default=2);p.add_argument('--screening-frontier',type=int,default=5);p.add_argument('--output',required=True);a=p.parse_args()
with threadpool_limits(8):
 start=time.monotonic();model,s=load_state_only(a.name);n=len(s.overlap);no=s.electron_count//2;v,u=eigh(s.overlap);sr=(u*np.sqrt(v))@u.T;frame=TransportedOperators(s,s.coordinates_bohr,np.zeros((0,n,n))).u;y=frame.T@sr@s.coefficients
 r=json.loads((HERE/f'pigment_roks/{a.name}/result.json').read_text());ys=[]
 for row in r['states']:
  with np.load(HERE/f"pigment_roks/{a.name}/state_{row['ordinal']}/checkpoint.npz") as z:ys.append(frame.T@sr@z['coefficients'])
 retained=scalar_basis(y,ys,[],no);screen=screening_path(a.name,a.screening_frontier);w=np.load(screen/'screened_pair.npy',mmap_mode='r').reshape((n,)*4);ham=Hamiltonian(np.load(screen/'h.npy'),w)
 folder=(HERE/f'pigment_cid/{a.name}' if a.frontier==2 else HERE/f'pigment_cid_expansion/{a.name}/frontier_{a.frontier}') if a.screening_frontier==5 else HERE/f'pigment_cid_expansion/{a.name}/screening_{a.screening_frontier}/frontier_{a.frontier}'
 expected=json.loads((folder/'scalar_elements.json').read_text())['elements'];densities=[]
 for i,bra in enumerate(retained):
  for j,ket in enumerate(retained[:i+1]):
   for unused,b in bra:
    for unused,k in ket:
     for x,z in zip(b,k):densities.extend(schur(x,z)[2:4])
 contraction_start=time.monotonic();ham.prefetch_densities(densities);errors=[]
 for i,bra in enumerate(retained):
  for j,ket in enumerate(retained[:i+1]):errors.append(np.array(ham.csf(bra,ket))-expected[f'{i},{j}'])
 errors=np.array(errors);maximum=float(np.max(abs(errors[:,1])));overlap_error=float(np.max(abs(errors[:,0])));assert maximum<1e-8 and overlap_error<1e-10,(maximum,overlap_error)
 result=dict(molecule=a.name,host=socket.getfqdn(),retained_CSFs=len(retained),scalar_pairs=len(errors),maximum_Hamiltonian_error_hartree=maximum,maximum_overlap_error=overlap_error,batched_contraction_wall_seconds=time.monotonic()-contraction_start,total_wall_seconds=time.monotonic()-start,SCF_calls=0,DFT_imports=0,approximation_changed=False,source_sha256={k:hashlib.sha256(Path(sys.modules[k].__file__).read_bytes()).hexdigest() for k in ['matrix_gfn2._nosi_slater','matrix_gfn2._nosi_surface_response']})
 write(HERE/a.output,result);print(json.dumps(result,indent=2),flush=True)
