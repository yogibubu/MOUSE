"""Independent checks of saved pigment spectra; remote, no SCF/DFT."""
from run_pigment_cid import *
from scipy.optimize import linear_sum_assignment
parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);args=parser.parse_args()
protocol=json.loads((HERE/args.protocol).read_text());NAME=protocol['molecule'];prefix=protocol['artifact_prefix'];fixed_w=protocol['fixed_screening_frontier'];baseline_w=protocol['baseline_screening_frontier'];root_ids=np.array(protocol['roots'])-1;root_count=len(root_ids)
def folder(n,w):
 return HERE/('pigment_cid/'+NAME if n==2 and w==5 else f'pigment_cid_expansion/{NAME}/'+('' if w==5 else f'screening_{w}/')+f'frontier_{n}')
def read(n,w):return json.loads((folder(n,w)/'result.json').read_text())
def screen_check(w):
 f=screening_path(NAME,w);d=np.load(f/'dielectric_polarization.npy');bn=np.load(f/'native_polarization.npy',mmap_mode='r');bb=np.load(f/'bath_polarization.npy',mmap_mode='r');lp=np.load(f/'lp.npy',mmap_mode='r');weighted=np.load(f/'weighted_lp.npy',mmap_mode='r');screen=np.load(f/'screened_pair.npy',mmap_mode='r');rng=np.random.default_rng(184+w);qe=re=we=0.
 for _ in range(3):
  v=rng.normal(size=len(d));v/=np.linalg.norm(v);expected=float(np.sum((v@bn)**2)+.6*np.sum((v@bb)**2));qe=max(qe,abs(float(v@d@v)-expected)/max(1.,abs(expected)))
  j=int(rng.integers(lp.shape[1]));re=max(re,float(np.linalg.norm(d@weighted[:,j]+weighted[:,j]-lp[:,j])/np.linalg.norm(lp[:,j])));we=max(we,float(np.max(abs(lp[:,j]@weighted-screen[j]))))
 assert max(qe,re,we)<1e-9
 return dict(W=w,quadratic_form_relative_error=qe,dielectric_solve_relative_residual=re,factorized_screened_row_max_error=we)
def eigen_check(n,w):
 f=folder(n,w);r=read(n,w)
 model_hash=hashlib.sha256((HERE/f'pigment_models/{NAME}/manifest.json').read_bytes()).hexdigest();assert r['model_sha256']==model_hash
 coefficients=[]
 for ordinal in r['retained_ROKS_ordinals']:
  with np.load(HERE/f'pigment_roks/{NAME}/state_{ordinal}/checkpoint.npz') as saved:coefficients.append(saved['coefficients'].copy())
 expected=hashlib.sha256((HERE/f'pigment_models/{NAME}/manifest.json').read_bytes()+(screening_path(NAME,w)/'certificate.json').read_bytes()+np.asarray(coefficients).tobytes()+json.dumps(r['auxiliary_records'],sort_keys=True).encode()).hexdigest()
 assert json.loads((f/'scalar_elements.json').read_text())['identity']==expected,'CI checkpoint identity changed'
 assert r['weights']==protocol['weights'] and r['lambda_exchange']==protocol['lambda_exchange'] and r['bath_efficiency_eta']==protocol['eta'] and r['energy_offset_ev']==protocol['energy_offset_ev']
 with np.load(f/'matrices.npz') as a:h=a['h'];s=a['s'];eg=float(a['ground_energy'])
 with np.load(f/'eigenstates.npz') as a:e=a['energies'];v=a['ci']
 residual=h@v-(s@v)*e;p=s[1:,0]/s[0,0];projected=residual[1:]-p[:,None]*residual[0];er=float(np.max(np.linalg.norm(projected,axis=0)));ortho=float(np.max(abs(v.T@s@v-np.eye(len(e)))));ground=float(np.max(abs(s[0]@v)));energy=float(np.max(abs((e-eg)*27.211386245988-np.array([x['excitation_ev'] for x in r['states']]))))
 assert max(er,ortho,ground,energy)<1e-8,(n,w,er,ortho,ground,energy)
 assert min(x['excitation_ev'] for x in r['states'])>0 and min(x['oscillator_strength'] for x in r['states'])>=0
 assert r['retained_ROKS_ordinals']==read(protocol['baseline_frontiers'][0],baseline_w)['retained_ROKS_ordinals'] and r['target_coverage']['minimum_target_coverage']>=.49-1e-10
 with np.load(f/'observables_phase_corrected.npz') as a:mu=a['transition_dipoles'];a=a['native_target_amplitudes']
 ferr=float(np.max(abs((2/3)*(e-eg)*np.sum(mu**2,axis=1)-np.array([x['oscillator_strength'] for x in r['states']]))));assert ferr<1e-12
 independent=None
 if n==protocol['baseline_frontiers'][0] and w==baseline_w:
  sp=s[1:,1:]-np.outer(s[1:,0],s[0,1:])/s[0,0];hp=h[1:,1:]-np.outer(h[1:,0],p)-np.outer(p,h[0,1:])+h[0,0]*np.outer(p,p);independent=float(np.max(abs(eigh(hp,sp,eigvals_only=True)[:len(e)]-e)));assert independent<1e-9
 return dict(N=n,W=w,basis_size=r['basis_size'],roots=len(e),projected_residual_hartree=er,metric_orthonormality_error=ortho,ground_overlap=ground,energy_reconstruction_error_ev=energy,oscillator_reconstruction_error=ferr,independent_dense_energy_error_hartree=independent,metric_certificate=r['metric_positive_definiteness_certificate'],target_coverage=r['target_coverage'])
def tracking(n1,w1,n2,w2):
 r1=read(n1,w1);r2=read(n2,w2);base=1+2*len(r1['retained_ROKS_ordinals']);lookup={json.dumps(x,sort_keys=True):j+base for j,x in enumerate(r2['auxiliary_records'])};mapping=list(range(base))+[lookup[json.dumps(x,sort_keys=True)] for x in r1['auxiliary_records']]
 with np.load(folder(n1,w1)/'matrices.npz') as z:h1=z['h'];s1=z['s'];eg1=float(z['ground_energy'])
 with np.load(folder(n2,w2)/'matrices.npz') as z:h2=z['h'];s2=z['s'];eg2=float(z['ground_energy'])
 eh=float(np.max(abs(h2[np.ix_(mapping,mapping)]-h1)));es=float(np.max(abs(s2[np.ix_(mapping,mapping)]-s1)));assert es<1e-10 and abs(eg1-eg2)<1e-8
 if w1==w2:assert eh<1e-8
 with np.load(folder(n1,w1)/'eigenstates.npz') as z:v1=z['ci'][:,:max(12,max(protocol['roots']))]
 with np.load(folder(n2,w2)/'eigenstates.npz') as z:v2=z['ci'][:,:max(12,max(protocol['roots']))]
 cross=v2.T@s2[:,mapping]@v1;pop=abs(cross)**2;child,parent=linear_sum_assignment(-pop);angles=np.linalg.svd(cross[np.ix_(root_ids,root_ids)],compute_uv=False)**2;assert max(angles)<1+1e-8
 f1=np.array([r1['states'][i]['oscillator_strength'] for i in root_ids]);f2=np.array([r2['states'][i]['oscillator_strength'] for i in root_ids])
 return dict(parent=[n1,w1],child=[n2,w2],same_interaction=w1==w2,Hamiltonian_principal_block_error_hartree=eh,metric_principal_block_error=es,monitored_subspace_principal_cosines_squared=angles.tolist(),assignments=[dict(parent_root=int(j+1),child_root=int(i+1),overlap_squared=float(pop[i,j])) for i,j in zip(child,parent)],monitored_subspace_oscillator_sum_relative_change=float(abs(f2.sum()-f1.sum())/max(f1.sum(),1e-15)),monitored_subspace_oscillator_changes=(f2-f1).tolist())
with threadpool_limits(4):
 convergence=json.loads((HERE/f'{prefix}_convergence.json').read_text());ns=convergence['windows'];pairs=[(protocol['baseline_frontiers'][0],baseline_w)]+[(n,n) for n in ns]+[(n,fixed_w) for n in ns];checks=[eigen_check(n,w) for n,w in dict.fromkeys(pairs)];screens=[screen_check(w) for w in sorted(set([baseline_w,fixed_w]+ns))];tracks=[tracking(a,a,b,b) for a,b in zip(ns,ns[1:])]+[tracking(a,fixed_w,b,fixed_w) for a,b in zip(ns,ns[1:])]
 result=dict(convergence=convergence,eigenstate_checks=checks,screening_checks=screens,root_tracking=tracks,coherent_final_states=read(ns[-1],ns[-1])['states'],coherent_selected_states=read(ns[-1],ns[-1])['selected_states'],all_algebra_checks_passed=True,energy_converged=convergence['coherent_converged'] and convergence['fixed_Hamiltonian_converged'],scope='Numerical fixed-model vertical validation; independent molecular accuracy and pigment nuclear derivatives not established',host=socket.getfqdn())
 write(HERE/f'{prefix}_assessment.json',result);print(json.dumps(result,indent=2),flush=True)
