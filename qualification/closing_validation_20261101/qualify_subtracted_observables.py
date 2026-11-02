"""Replay exact fixed-ground transition operators for a subtracted CI spectrum."""
from run_pigment_cid import *
p=argparse.ArgumentParser();p.add_argument('--diagnostic',required=True);a=p.parse_args();path=HERE/a.diagnostic;diagnostic=json.loads(path.read_text());name=diagnostic['molecule'];nfront=diagnostic['frontier'];w=diagnostic['screening_frontier'];out=path.with_name(path.stem+'_observables.json')
if out.exists():print('REUSE',out,flush=True);raise SystemExit(0)
with threadpool_limits(2):
 folder=HERE/f'pigment_cid_transition/{name}/screening_{w}/frontier_{nfront}';report=json.loads((folder/'result.json').read_text());model,s=load_state_only(name);no=s.electron_count//2;n=len(s.overlap);v,u=eigh(s.overlap);sr=(u*np.sqrt(v))@u.T;si=(u*v**-.5)@u.T;frame=TransportedOperators(s,s.coordinates_bohr,np.zeros((0,n,n))).u;y=frame.T@sr@s.coefficients;ys=[]
 assert report['model_sha256']==hashlib.sha256((model/'manifest.json').read_bytes()).hexdigest()
 for ordinal in report['retained_ROKS_ordinals']:
  with np.load(HERE/f'pigment_roks/{name}/state_{ordinal}/checkpoint.npz') as z:ys.append(frame.T@sr@z['coefficients'])
 retained=scalar_basis(y,ys,[],no);nbase=len(retained);records=report['auxiliary_records'];nb=nbase+len(records)
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O',12:'Mg'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.);calc.evaluate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('new SCC forbidden'))
 dipole_ao=calc._model(s.coordinates_bohr)[2]
 if dipole_ao is None:
  pairs=calc._post_scf_pair_integrals;dipole_ao=np.zeros((3,n,n));dipole_ao[:,pairs.rows,pairs.columns]=pairs.dipole.T;dipole_ao[:,pairs.columns,pairs.rows]=pairs.dipole.T
 dipoles=np.array([-frame.T@si@a@si@frame for a in dipole_ao]);transition=np.zeros((nb,3));transition[:nbase]=[transition_dipole(retained[0],b,dipoles) for b in retained];mu_mo=np.einsum('pi,xpq,qj->xij',y,dipoles,y,optimize=True)
 roks=json.loads((HERE/f'pigment_roks/{name}/result.json').read_text());native=np.zeros((len(roks['states']),nb))
 for k,row in enumerate(roks['states']):
  hole,particle=row['hole'],row['particle'];order=[hole]+[j for j in range(no) if j!=hole]+[particle]+[j for j in range(no,n) if j!=particle];target=opened(y[:,order],no);native[k,:nbase]=[overlap(target,b) for b in retained]
 for j,record in enumerate(records,nbase):
  if record['kind']!='open':continue
  holes=[i for i in range(no) if i not in record['closed']]
  if len(holes)!=1 or record['open'][0]!=holes[0] or record['open'][1]<no:continue
  hole,particle=record['open'];transition[j]=np.sqrt(2)*mu_mo[:,hole,particle]
  for k,row in enumerate(roks['states']):
   if [row['hole'],row['particle']]==record['open']:native[k,j]=float((-1)**(no-1))
 with np.load(folder/'eigenstates.npz') as z:old_ci=z['ci'].copy()
 with np.load(folder/'observables_phase_corrected.npz') as z:old_moment=z['transition_dipoles'];old_native=z['native_target_amplitudes']
 moment_error=float(np.max(abs(old_ci.T@transition-old_moment)));native_error=float(np.max(abs(native@old_ci-old_native)))
 assert max(moment_error,native_error)<1e-8,(moment_error,native_error)
 projection=np.zeros(nb-1);projection[:nbase-1]=[overlap(retained[0],b)/overlap(retained[0],retained[0]) for b in retained[1:]]
 assert np.max(abs(old_ci[0]+projection@old_ci[1:]))<1e-9
 with np.load(path.with_suffix('.npz')) as z:e=z['energies'];zci=z['ci']
 ci=np.vstack((-projection@zci,zci));moment=ci.T@transition;osc=(2/3)*e*np.sum(moment**2,axis=1);amplitudes=native@ci;selected,coverage=select_target_subspace(amplitudes,minimum_coverage=.49)
 states=[dict(root=i+1,excitation_ev=float(value*27.211386245988),oscillator_strength=float(osc[i]),transition_dipole_au=moment[i].tolist(),native_frontier_subspace_weight=float(np.sum(amplitudes[:,i]**2)),selected=bool(i in selected)) for i,value in enumerate(e)]
 result=dict(molecule=name,frontier=nfront,screening_frontier=w,states=states,target_coverage=coverage,original_spectrum_transition_replay_error_au=moment_error,original_native_projector_replay_error=native_error,spin_squared=0.,spin_scope='Every basis CSF is a coherent spin singlet',DFT_imports=0,new_SCC_calls=0,TDA_solves=0,scope='Exact fixed-native-ground CSF transition operators for experimental static-matched Hamiltonian; no correlated-ground or pigment nuclear-derivative claim')
 write(out,result);np.savez_compressed(out.with_suffix('.npz'),energies=e,ci=ci,transition_operator=transition,native_target_operator=native,transition_dipoles=moment);print(json.dumps(result,indent=2),flush=True)
