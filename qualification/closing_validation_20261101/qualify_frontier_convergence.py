"""Remote equivalence checks for polynomial enumeration and selected-root solves."""
from run_pigment_cid import *
from scipy.linalg import cho_factor,cho_solve
from matrix_gfn2._nosi_selected_roots import solve,metric_certificate,projector_population
from matrix_gfn2._nosi_configurations import frontier_rank_two

def brute(no,nf):
 core=list(range(no-nf));sites=list(range(no-nf,no+nf));out=[]
 for paired in combinations(sites,nf):
  if 2*sum(j>=no for j in paired)<=2:out.append(dict(kind='closed',closed=core+list(paired)))
 for paired in combinations(sites,nf-1):
  for opened in combinations([j for j in sites if j not in paired],2):
   if 2*sum(j>=no for j in paired)+sum(j>=no for j in opened)<=2:out.append(dict(kind='open',closed=core+list(paired),open=list(opened)))
 for paired in combinations(sites,nf-2):
  for opened in combinations([j for j in sites if j not in paired],4):
   if 2*sum(j>=no for j in paired)+sum(j>=no for j in opened)<=2:
    for component in range(2):out.append(dict(kind='four',closed=core+list(paired),open=list(opened),component=component))
 return out

with threadpool_limits(8):
 for nf in range(2,6):assert frontier_rank_two(8,nf)==brute(8,nf)
 # Independent Slater transition operator verifies canonical signs and exact zeros.
 no=4;n=8;y=np.eye(n);rng=np.random.default_rng(793);mu=rng.normal(size=(3,n,n));mu=(mu+mu.transpose(0,2,1))/2
 error=0.
 for rec,csf in zip(frontier_rank_two(no,3),configuration_basis(y,frontier_rank_two(no,3))):
  expected=np.zeros(3)
  if rec['kind']=='closed' and rec['closed']==list(range(no)):expected=2*np.einsum('xii->x',mu[:,:no,:no])
  if rec['kind']=='open':
   holes=[i for i in range(no) if i not in rec['closed']]
   if len(holes)==1 and rec['open']==[holes[0],rec['open'][1]] and rec['open'][1]>=no:expected=2**.5*mu[:,holes[0],rec['open'][1]]
  error=max(error,float(np.max(abs(transition_dipole(closed(y,list(range(no))),csf,mu)-expected))))
 assert error<1e-12,error
 folder=HERE/'pigment_cid_expansion/Beta_carotene/screening_10/frontier_8'
 with np.load(folder/'matrices.npz') as z:h=z['h'];s=z['s']
 with np.load(folder/'eigenstates.npz') as z:old=z['energies'][:20];oldci=z['ci'][:,:20]
 p=s[0,1:]/s[0,0];sp=s[1:,1:]-np.outer(s[1:,0],p);hp=h[1:,1:]-np.outer(h[1:,0],p)-np.outer(p,h[0,1:])+h[0,0]*np.outer(p,p)
 e,z=eigh(hp,sp,subset_by_index=[0,19]);ci=np.vstack((-p@z,z));ee=float(np.max(abs(e-old))*27.211386245988);ov=float(np.min(abs(np.diag(oldci.T@s@ci))**2))
 assert ee<1e-8 and ov>1-1e-8,(ee,ov)
 start=time.monotonic();ar_e,ar_ci,ar_cert=solve(h,s,float(old[0])-json.loads((folder/'result.json').read_text())['states'][0]['excitation_ev']/27.211386245988,20)
 ar_error=float(np.max(abs(ar_e-old))*27.211386245988);assert ar_error<1e-8,ar_error
 metric_cert=metric_certificate(s,3);assert metric_cert['certified_shift']==1e-8
 toy=np.eye(27);toy[1,3:]=.15;toy[3:,1]=.15;toy[0,2]=toy[2,0]=.2
 border_cert=metric_certificate(toy,3);assert 'Schur' in border_cert['method']
 toy[1,1]=.2
 try:metric_certificate(toy,3)
 except ValueError:pass
 else:raise AssertionError('indefinite metric accepted')
 ar_wall=time.monotonic()-start
 model,state=load_state_only('Beta_carotene');no=state.electron_count//2;n=len(state.overlap)
 calc=GFN2Calculator([{1:'H',6:'C'}[int(a)] for a in state.atomic_numbers],temperature_kelvin=0.)
 vals,u=eigh(state.overlap);sr=(u*vals**.5)@u.T;si=(u*vals**-.5)@u.T;frame=TransportedOperators(state,state.coordinates_bohr,np.zeros((0,n,n))).u;y=frame.T@sr@state.coefficients
 roks=json.loads((HERE/'pigment_roks/Beta_carotene/result.json').read_text());ys=[]
 for ordinal in json.loads((folder/'result.json').read_text())['retained_ROKS_ordinals']:
  with np.load(HERE/f'pigment_roks/Beta_carotene/state_{ordinal}/checkpoint.npz') as a:ys.append(frame.T@sr@a['coefficients'])
 retained=scalar_basis(y,ys,[],no);records=json.loads((folder/'result.json').read_text())['auxiliary_records']
 dip=calc._model(state.coordinates_bohr)[2]
 if dip is None:
  pairs=calc._post_scf_pair_integrals;dip=np.zeros((3,n,n));dip[:,pairs.rows,pairs.columns]=pairs.dipole.T;dip[:,pairs.columns,pairs.rows]=pairs.dipole.T
 dip=np.array([-frame.T@si@a@si@frame for a in dip]);mu_mo=np.einsum('pi,xpq,qj->xij',y,dip,y,optimize=True)
 transition=np.zeros((len(s),3));transition[:len(retained)]=[transition_dipole(retained[0],b,dip) for b in retained]
 for j,rec in enumerate(records):
  if rec['kind']=='open':
   holes=[i for i in range(no) if i not in rec['closed']]
   if len(holes)==1 and rec['open'][0]==holes[0] and rec['open'][1]>=no:transition[len(retained)+j]=2**.5*mu_mo[:,holes[0],rec['open'][1]]
 with np.load(folder/'eigenstates.npz') as a:legacy_moment_difference=float(np.max(abs(oldci.T@transition-a['transition_dipoles'][:20])))
 with np.load(folder/'observables_phase_corrected.npz') as a:moment_error=float(np.max(abs(oldci.T@transition-a['transition_dipoles'][:20])))
 assert moment_error<1e-8,moment_error
 nbase=len(retained);indices=list(range(1+len(ys),nbase))
 for j,rec in enumerate(records):
  if 2*sum(k>=no for k in rec['closed'])+sum(k>=no for k in rec.get('open',[]))==2:indices.append(nbase+j)
 gram=s[np.ix_(indices,indices)]-np.outer(s[indices,0],s[0,indices])/s[0,0];cross=(s[indices]-np.outer(s[indices,0]/s[0,0],s[0]))@oldci
 population=np.sum(cross*cho_solve(cho_factor(gram,lower=True),cross),axis=0)
 previous=json.loads((folder/'analysis.json').read_text());population_error=float(np.max(abs(population-np.array([r['double_subspace_weight'] for r in previous['states'][:20]]))))
 assert population_error<1e-9,population_error
 sparse_projector_error=float(np.max(abs(projector_population(s,indices,oldci)-population)));assert sparse_projector_error<1e-9
 report=dict(host=socket.getfqdn(),enumeration_identical_frontiers=[2,3,4,5],iterative_selected_root_error_ev=ar_error,iterative_wall_seconds=ar_wall,iterative_certificate=ar_cert,metric_certificate=metric_cert,Schur_metric_certificate=border_cert,sparse_projector_error=sparse_projector_error,projector_population_error=population_error,canonical_transition_max_error=error,corrected_pigment_transition_max_error_au=moment_error,legacy_transition_phase_difference_au=legacy_moment_difference,projected_selected_root_energy_error_ev=ee,minimum_squared_old_new_overlap=ov,parameters_changed=False)
 write(HERE/'frontier_convergence_qualification.json',report);print(json.dumps(report),flush=True)
