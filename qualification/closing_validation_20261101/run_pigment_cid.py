"""Prepared DFT-free spin-singlet screened CID for the two pigment examples."""
from run_pigment_roks import *
from itertools import combinations
from scipy.linalg import eigh
from matrix_gfn2._nosi_configurations import basis as configuration_basis,validate
from matrix_gfn2._nosi_surface_response import MatrixResponse
from matrix_gfn2._nosi_slater import schur

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

def run(name,frontier=2,screening_frontier=5):
 if not 2<=frontier<=screening_frontier<=12:raise ValueError('CI must fit inside the excluded screening frontier')
 out=(HERE/'pigment_cid'/name if frontier==2 else HERE/'pigment_cid_expansion'/name/f'frontier_{frontier}') if screening_frontier==5 else HERE/'pigment_cid_expansion'/name/f'screening_{screening_frontier}'/f'frontier_{frontier}';out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 start=time.monotonic();model,s,reference,calc,base,ctx=load(name);print(name,'loaded full-XC model',flush=True);n=len(s.overlap);no=s.electron_count//2;v,u=eigh(s.overlap);sr=(u*v**.5)@u.T;si=(u*v**-.5)@u.T;c=s.coefficients;y=ctx.charge_correction.transport.u.T@sr@c
 roks=json.loads((HERE/'pigment_roks'/name/'result.json').read_text());cs=[];mus=[];ordinals=[]
 for row in roks['states']:
  with np.load(HERE/'pigment_roks'/name/f"state_{row['ordinal']}"/'checkpoint.npz') as z:cc=z['coefficients'].copy();mu=float(z['multiplier'])
  candidate=opened(sr@cc,no)
  if any(abs(overlap(candidate,opened(sr@previous,no)))>1-1e-9 for previous in cs):continue
  cs.append(cc);mus.append(mu);ordinals.append(row['ordinal'])
 ys=[ctx.charge_correction.transport.u.T@sr@cc for cc in cs];retained=scalar_basis(y,ys,[],no);core=list(range(no-frontier));sites=list(range(no-frontier,no+frontier));records=[]
 # Complete spin-singlet rank-two frontier rule. The default reproduces
 # the original two-hole/two-particle space; expansion changes membership only.
 for pair in combinations(sites,frontier):
  if 2*sum(j>=no for j in pair)>2:continue
  rec=dict(kind='closed',closed=core+list(pair));candidate=configuration_basis(y,[rec])[0]
  if max(abs(overlap(candidate,b)) for b in retained)<.7:records.append(rec)
 for paired in combinations(sites,frontier-1):
  for opens in combinations([j for j in sites if j not in paired],2):
   if 2*sum(j>=no for j in paired)+sum(j>=no for j in opens)>2:continue
   rec=dict(kind='open',closed=core+list(paired),open=list(opens));candidate=configuration_basis(y,[rec])[0]
   if max(abs(overlap(candidate,b)) for b in retained)<.7:records.append(rec)
 for paired in combinations(sites,frontier-2):
  for opens in combinations([j for j in sites if j not in paired],4):
   if 2*sum(j>=no for j in paired)+sum(j>=no for j in opens)>2:continue
   for component in range(2):records.append(dict(kind='four',closed=core+list(paired),open=list(opens),component=component))
 validate(records,no,n);basis=scalar_basis(y,ys,records,no);labels=['ground']+[f'ROKS_{i+1}' for i in range(len(cs))]+[f'paired_{i+1}' for i in range(len(cs))]+[f'canonical_{i+1}' for i in range(len(records))]
 matrices=out/'matrices.npz';screen=screening_path(name,screening_frontier);reused=matrices.exists()
 if not matrices.exists():
  ctx.register_density_span(np.array([np.outer(c[:,j],c[:,j]) for j in sites]));h=np.load(screen/'h.npy');w=np.load(screen/'screened_pair.npy',mmap_mode='r').reshape((n,)*4)
  journal=out/'scalar_elements.json';identity=hashlib.sha256((model/'manifest.json').read_bytes()+(screen/'certificate.json').read_bytes()+np.asarray(cs).tobytes()+json.dumps(records,sort_keys=True).encode()).hexdigest();elements={}
  if journal.exists():
   saved=json.loads(journal.read_text());assert saved['identity']==identity;elements={tuple(map(int,key.split(','))):tuple(value) for key,value in saved['elements'].items()}
  def checkpoint(values):
   write(journal,dict(identity=identity,elements={f'{i},{j}':value for (i,j),value in values.items()}));print(name,'CI orbital couplings complete',len(values),flush=True)
  response=MatrixResponse(ctx,reference,c,cs,mus,y,labels,h,w,derivatives=False,auxiliary_records=records,scalar_elements=elements,scalar_checkpoint=checkpoint,fixed_bra=True);hh,ss,eg=response.matrices(np.zeros(s.coordinates_bohr.size));np.savez(matrices,h=hh.v,s=ss.v,ground_energy=float(eg.v));del response,w
 with np.load(matrices) as z:h=z['h'];metric=z['s'];eg=float(z['ground_energy'])
 projection=np.eye(len(basis))[:,1:];projection[0]-=metric[0,1:]/metric[0,0];sp=projection.T@metric@projection;smallest=float(np.linalg.eigvalsh(sp).min());assert smallest>1e-8,('unresolved CI metric dependence',smallest)
 energies,z=eigh(projection.T@h@projection,sp);ci=projection@z
 # A multi-promotion pigment is tracked as one retained target subspace.
 targets=np.eye(len(basis))[:,1:1+len(cs)].copy();targets[0]-=(metric[0]@targets)/metric[0,0];tv,tu=eigh(targets.T@metric@targets);assert tv.min()>1e-8;targets=targets@((tu*tv**-.5)@tu.T);weights=np.sum((targets.T@metric@ci)**2,axis=0)
 native_targets=[]
 for row in roks['states']:
  hole,particle=row['hole'],row['particle'];order=[hole]+[j for j in range(no) if j!=hole]+[particle]+[j for j in range(no,n) if j!=particle];native_targets.append(opened(y[:,order],no))
 native_cross=np.array([[overlap(target,b) for b in basis] for target in native_targets]);native_weights=np.sum((native_cross@ci)**2,axis=0);selected=np.sort(np.argsort(native_weights)[-len(native_targets):]);assert np.min(native_weights[selected])>=.49,('unresolved native-frontier target assignment',native_weights)
 dipole_ao=calc._model(s.coordinates_bohr)[2]
 if dipole_ao is None:
  pairs=calc._post_scf_pair_integrals;dipole_ao=np.zeros((3,n,n));dipole_ao[:,pairs.rows,pairs.columns]=pairs.dipole.T;dipole_ao[:,pairs.columns,pairs.rows]=pairs.dipole.T
 if dipole_ao.shape!=(3,n,n):raise ValueError(('native dipole shape',dipole_ao.shape))
 frame=ctx.charge_correction.transport.u;dipoles=np.array([-frame.T@si@a@si@frame for a in dipole_ao]);transition=np.array([transition_dipole(basis[0],b,dipoles) for b in basis]);moment=ci.T@transition;gap=energies-eg;osc=(2/3)*gap*np.sum(moment**2,axis=1)
 rows=[dict(root=i+1,excitation_ev=float(gap[i]*27.211386245988),oscillator_strength=float(osc[i]),transition_dipole_au=moment[i].tolist(),retained_subspace_weight=float(weights[i]),native_frontier_subspace_weight=float(native_weights[i]),selected=bool(i in selected)) for i in range(len(gap))]
 np.savez(out/'eigenstates.npz',energies=energies,ci=ci,transition_dipoles=moment,metric=metric)
 report=dict(molecule=name,host=socket.getfqdn(),states=rows,selected_states=[rows[i] for i in selected],basis_size=len(basis),auxiliary_records=records,retained_ROKS_ordinals=ordinals,minimum_projected_metric_eigenvalue=smallest,spin_squared=0.,weights=[.5,.5],lambda_exchange=.9,bath_efficiency_eta=.6,energy_offset_ev=0,DFT_imports=0,TDA_solves=0,current_process_wall_seconds=time.monotonic()-start,matrix_checkpoint_reused=reused,timing_scope='Current process only; excludes previous matrix construction when resumed and all DFT preparation',model_sha256=hashlib.sha256((model/'manifest.json').read_bytes()).hexdigest(),scope='Vertical examples on source geometries; no excited geometry or pigment nuclear derivative qualification',target_rule='Largest collective overlap with the frozen native-frontier singlet subspace, minimum weight 0.49; ROKS weights reported separately',numerical_quadrature='Full frozen XC quadrature; RI used only in orbital Newton step')
 report['frontier_occupied']=frontier;report['frontier_virtual']=frontier;report['screened_interaction_reused']=True;report['screening_active_frontier']=screening_frontier;report['role']='Baseline' if frontier==2 else 'Post-exposure active-space convergence diagnostic; no parameter retuning'
 write(out/'result.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--wait-for-screening',action='store_true');p.add_argument('--frontier',type=int,choices=range(2,13),default=2);p.add_argument('--screening-frontier',type=int,default=5);a=p.parse_args()
 while not (screening_path(a.name,a.screening_frontier)/'certificate.json').exists():
  if not a.wait_for_screening:raise RuntimeError('Screening incomplete')
  time.sleep(20)
 with threadpool_limits(8):run(a.name,a.frontier,a.screening_frontier)
