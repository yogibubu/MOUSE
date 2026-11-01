"""Prepared DFT-free spin-singlet screened CID for the two pigment examples."""
from run_pigment_roks import *
from itertools import combinations
from scipy.linalg import eigh
from matrix_gfn2._nosi_configurations import basis as configuration_basis,validate,frontier_rank_two
from matrix_gfn2._nosi_surface_response import MatrixResponse
from matrix_gfn2._nosi_slater import schur
from matrix_gfn2._nosi_target_subspace import select_target_subspace
from matrix_gfn2._nosi_selected_roots import solve as selected_roots,metric_certificate
from matrix_gfn2.nonlocal_response import TransportedOperators

def closed(y,indices):return [(1.,(y[:,indices],y[:,indices]))]
def opened(y,no):
 a=y[:,list(range(1,no))+[0]];b=y[:,list(range(1,no))+[no]]
 return [(2**-.5,(a,b)),(2**-.5,(b,a))]
def overlap(a,b):return sum(x*y*np.linalg.det(u.T@v)*np.linalg.det(w.T@z) for x,(u,w) in a for y,(v,z) in b)
def scalar_basis(y,ys,records,no):return [closed(y,list(range(no)))]+[opened(c,no) for c in ys]+[closed(c,list(range(1,no))+[no]) for c in ys]+configuration_basis(y,records)
def transition_dipole(a,b,mu):
 value=np.zeros(3)
 for x,bra in a:
  for y,ket in b:
   aa,bb=[schur(u,v) for u,v in zip(bra,ket)];pa,da,ra,qa,*_=aa;pb,db,rb,qb,*_=bb
   value+=x*y*pa*pb*(db*(da*np.einsum('xij,ji->x',mu,ra)+np.einsum('xij,ji->x',mu,qa))+da*(db*np.einsum('xij,ji->x',mu,rb)+np.einsum('xij,ji->x',mu,qb)))
 return value

def screening_path(name,frontier):return HERE/'pigment_screening'/name if frontier==5 else HERE/'pigment_screening_expansion'/name/f'frontier_{frontier}'

def run(name,frontier=2,screening_frontier=5,roots=None):
 if not 2<=frontier<=16 and frontier<=screening_frontier:raise ValueError('CI must fit inside the excluded screening frontier')
 out=(HERE/'pigment_cid'/name if frontier==2 else HERE/'pigment_cid_expansion'/name/f'frontier_{frontier}') if screening_frontier==5 else HERE/'pigment_cid_expansion'/name/f'screening_{screening_frontier}'/f'frontier_{frontier}';out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 start=time.monotonic();matrices=out/'matrices.npz'
 if matrices.exists():
  model,s=load_state_only(name);calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O',12:'Mg'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.);frame=TransportedOperators(s,s.coordinates_bohr,np.zeros((0,len(s.overlap),len(s.overlap)))).u
  calc.evaluate=lambda *x,**k:(_ for _ in ()).throw(AssertionError('New SCC forbidden'))
  print(name,'resume saved CI matrices without loading response tensors',flush=True)
 else:
  model,s,reference,calc,base,ctx=load(name);frame=ctx.charge_correction.transport.u;print(name,'loaded full-XC model',flush=True)
 n=len(s.overlap);no=s.electron_count//2;assert screening_frontier<=min(no,n-no);v,u=eigh(s.overlap);sr=(u*v**.5)@u.T;si=(u*v**-.5)@u.T;c=s.coefficients;y=frame.T@sr@c
 roks=json.loads((HERE/'pigment_roks'/name/'result.json').read_text());cs=[];mus=[];ordinals=[]
 for row in roks['states']:
  with np.load(HERE/'pigment_roks'/name/f"state_{row['ordinal']}"/'checkpoint.npz') as z:cc=z['coefficients'].copy();mu=float(z['multiplier'])
  candidate=opened(sr@cc,no)
  if any(abs(overlap(candidate,opened(sr@previous,no)))>1-1e-9 for previous in cs):continue
  cs.append(cc);mus.append(mu);ordinals.append(row['ordinal'])
 ys=[frame.T@sr@cc for cc in cs];retained=scalar_basis(y,ys,[],no);core=list(range(no-frontier));sites=list(range(no-frontier,no+frontier));records=[]
 # Complete spin-singlet rank-two frontier rule. The default reproduces
 # the original two-hole/two-particle space; expansion changes membership only.
 for rec in frontier_rank_two(no,frontier):
  if rec['kind']=='four':records.append(rec);continue
  candidate=configuration_basis(y,[rec])[0]
  if max(abs(overlap(candidate,b)) for b in retained)<.7:records.append(rec)
 validate(records,no,n);basis=scalar_basis(y,ys,records,no);labels=['ground']+[f'ROKS_{i+1}' for i in range(len(cs))]+[f'paired_{i+1}' for i in range(len(cs))]+[f'canonical_{i+1}' for i in range(len(records))]
 matrices=out/'matrices.npz';screen=screening_path(name,screening_frontier);reused=matrices.exists();inherited=None
 if not matrices.exists():
  ctx.register_density_span(np.array([np.outer(c[:,j],c[:,j]) for j in sites]));h=np.load(screen/'h.npy');w=np.load(screen/'screened_pair.npy',mmap_mode='r').reshape((n,)*4)
  journal=out/'scalar_elements.json';identity=hashlib.sha256((model/'manifest.json').read_bytes()+(screen/'certificate.json').read_bytes()+np.asarray(cs).tobytes()+json.dumps(records,sort_keys=True).encode()).hexdigest();elements={}
  if journal.exists():
   saved=json.loads(journal.read_text());assert saved['identity']==identity;elements={tuple(map(int,key.split(','))):tuple(value) for key,value in saved['elements'].items()}
  # Identical physical bra/ket pairs survive a frontier expansion. Reuse
  # completed scalar elements only after verifying the full parent identity.
  if not elements and frontier>2:
   for parent_n in range(frontier-1,1,-1):
    parent=out.parent/f'frontier_{parent_n}'
    if not (parent/'result.json').exists() or not (parent/'scalar_elements.json').exists():continue
    previous=json.loads((parent/'result.json').read_text());previous_elements=json.loads((parent/'scalar_elements.json').read_text())
    expected=hashlib.sha256((model/'manifest.json').read_bytes()+(screen/'certificate.json').read_bytes()+np.asarray(cs).tobytes()+json.dumps(previous['auxiliary_records'],sort_keys=True).encode()).hexdigest()
    if previous_elements['identity']!=expected:continue
    assert previous['retained_ROKS_ordinals']==ordinals
    lookup={json.dumps(rec,sort_keys=True):j+len(retained) for j,rec in enumerate(records)}
    mapping=list(range(len(retained)))+[lookup[json.dumps(rec,sort_keys=True)] for rec in previous['auxiliary_records']]
    for key,value in previous_elements['elements'].items():
     i,j=map(int,key.split(','));elements[(mapping[i],mapping[j])]=tuple(value)
    inherited=dict(parent_frontier=parent_n,scalar_elements=len(elements),identity_verified=True)
    print(name,'reused exact scalar couplings',inherited,flush=True);break
  last_checkpoint=[len(elements)]
  def checkpoint(values,force=False):
   if not force and len(values)-last_checkpoint[0]<32:return
   last_checkpoint[0]=len(values)
   write(journal,dict(identity=identity,elements={f'{i},{j}':value for (i,j),value in values.items()}));print(name,'CI orbital couplings complete',len(values),flush=True)
  response=MatrixResponse(ctx,reference,c,cs,mus,y,labels,h,w,derivatives=False,auxiliary_records=records,scalar_elements=elements,scalar_checkpoint=checkpoint,fixed_bra=True);hh,ss,eg=response.matrices(np.zeros(s.coordinates_bohr.size));checkpoint(response.scalar_elements,force=True);np.savez(matrices,h=hh.v,s=ss.v,ground_energy=float(eg.v));del response,w
 with np.load(matrices) as z:h=z['h'];metric=z['s'];eg=float(z['ground_energy'])
 certificate=metric_certificate(metric,len(retained));smallest=certificate.get('lower_bound') if certificate['method']=='Smallest projected eigenvalue' else None;bound=certificate.get('lower_bound')
 if roots is not None:
  energies,ci,eigen_certificate=selected_roots(h,metric,eg,roots)
 else:
  p=metric[0,1:]/metric[0,0];sp=metric[1:,1:]-np.outer(metric[1:,0],p)
  hp=h[1:,1:]-np.outer(h[1:,0],p)-np.outer(p,h[0,1:])+h[0,0]*np.outer(p,p)
  energies,z=eigh(hp,sp);ci=np.vstack((-p@z,z));del hp,sp
  eigen_certificate=dict(method='Complete dense generalized spectrum',roots=len(energies))
 # A multi-promotion pigment is tracked as one retained target subspace.
 targets=np.eye(len(basis))[:,1:1+len(cs)].copy();targets[0]-=(metric[0]@targets)/metric[0,0];tv,tu=eigh(targets.T@metric@targets);assert tv.min()>1e-8;targets=targets@((tu*tv**-.5)@tu.T);weights=np.sum((targets.T@metric@ci)**2,axis=0)
 native_targets=[]
 for row in roks['states']:
  hole,particle=row['hole'],row['particle'];order=[hole]+[j for j in range(no) if j!=hole]+[particle]+[j for j in range(no,n) if j!=particle];native_targets.append(opened(y[:,order],no))
 native_cross=np.zeros((len(native_targets),len(basis)))
 for k,target in enumerate(native_targets):
  native_cross[k,:len(retained)]=[overlap(target,b) for b in retained]
  hole,particle=roks['states'][k]['hole'],roks['states'][k]['particle']
  for j,rec in enumerate(records):
   if rec['kind']=='open' and rec['open']==[hole,particle] and rec['closed']==[i for i in range(no) if i!=hole]:native_cross[k,len(retained)+j]=1.
 native_amplitudes=native_cross@ci;native_weights=np.sum(native_amplitudes**2,axis=0);selected,target_coverage=select_target_subspace(native_amplitudes,minimum_coverage=.49)
 dipole_ao=calc._model(s.coordinates_bohr)[2]
 if dipole_ao is None:
  pairs=calc._post_scf_pair_integrals;dipole_ao=np.zeros((3,n,n));dipole_ao[:,pairs.rows,pairs.columns]=pairs.dipole.T;dipole_ao[:,pairs.columns,pairs.rows]=pairs.dipole.T
 if dipole_ao.shape!=(3,n,n):raise ValueError(('native dipole shape',dipole_ao.shape))
 dipoles=np.array([-frame.T@si@a@si@frame for a in dipole_ao]);transition=np.zeros((len(basis),3));transition[:len(retained)]=[transition_dipole(retained[0],b,dipoles) for b in retained];mu_mo=np.einsum('pi,xpq,qj->xij',y,dipoles,y,optimize=True)
 for j,rec in enumerate(records):
  if rec['kind']=='open':
   holes=[i for i in range(no) if i not in rec['closed']]
   if len(holes)==1 and rec['open'][0]==holes[0] and rec['open'][1]>=no:transition[len(retained)+j]=2**.5*mu_mo[:,holes[0],rec['open'][1]]
 moment=ci.T@transition;gap=energies-eg;osc=(2/3)*gap*np.sum(moment**2,axis=1)
 rows=[dict(root=i+1,excitation_ev=float(gap[i]*27.211386245988),oscillator_strength=float(osc[i]),transition_dipole_au=moment[i].tolist(),retained_subspace_weight=float(weights[i]),native_frontier_subspace_weight=float(native_weights[i]),selected=bool(i in selected)) for i in range(len(gap))]
 np.savez(out/'eigenstates.npz',energies=energies,ci=ci,transition_dipoles=moment,metric=metric)
 report=dict(molecule=name,host=socket.getfqdn(),states=rows,selected_states=[rows[i] for i in selected],basis_size=len(basis),auxiliary_records=records,retained_ROKS_ordinals=ordinals,minimum_projected_metric_eigenvalue=smallest,projected_metric_Gershgorin_lower_bound=bound if certificate['method']=='Gershgorin' else None,metric_positive_definiteness_certificate=certificate,eigensolver_certificate=eigen_certificate,eigenstates_requested=roots,spin_squared=0.,weights=[.5,.5],lambda_exchange=.9,bath_efficiency_eta=.6,energy_offset_ev=0,DFT_imports=0,TDA_solves=0,current_process_wall_seconds=time.monotonic()-start,matrix_checkpoint_reused=reused,timing_scope='Current process only; excludes previous matrix construction when resumed and all DFT preparation',model_sha256=hashlib.sha256((model/'manifest.json').read_bytes()).hexdigest(),scope='Vertical examples on source geometries; no excited geometry or pigment nuclear derivative qualification',target_coverage=target_coverage,target_rule='Retain eigenstates until every fixed native-frontier target direction has at least 0.49 projector coverage; fragmented targets remain a subspace; ROKS weights reported separately',numerical_quadrature='Full frozen XC quadrature; RI used only in orbital Newton step')
 report['inherited_scalar_elements']=inherited;report['scalar_checkpoint_batch_size']=32;report['frontier_occupied']=frontier;report['frontier_virtual']=frontier;report['screened_interaction_reused']=True;report['screening_active_frontier']=screening_frontier;report['role']='Baseline' if frontier==2 else 'Post-exposure active-space convergence diagnostic; no parameter retuning'
 write(out/'result.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--wait-for-screening',action='store_true');p.add_argument('--frontier',type=int,choices=range(2,17),default=2);p.add_argument('--roots',type=int);p.add_argument('--screening-frontier',type=int,default=5);a=p.parse_args()
 while not (screening_path(a.name,a.screening_frontier)/'certificate.json').exists():
  if not a.wait_for_screening:raise RuntimeError('Screening incomplete')
  time.sleep(20)
 with threadpool_limits(8):run(a.name,a.frontier,a.screening_frontier,a.roots)
