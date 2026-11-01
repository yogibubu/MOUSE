"""Independent displaced-root energy checks of the analytic nuclear Hessian."""
from nuclear_assembly import *
from slater_condon import csf_element as literal_csf
import argparse

def literal_closed(c,indices):return [(1.,(c[:,indices],c[:,indices]))]
def literal_open(c,no):
    a=c[:,list(range(1,no))+[no]];b=c[:,:no]
    return [(2**-.5,(a,b)),(2**-.5,(b,a))]

def displaced(model,delta):
    xyz=model.case['xyz']+delta.reshape(-1,3)/ANGSTROM_TO_BOHR
    calc=calculator(model.s);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC forbidden in independent nuclear validation'))
    context=prepare_gfn2_deltascf(calc,xyz,orbital_reference=model.s,spin_reference=model.reference)
    cg,eg,gres,_=refine.ground(context,model.ground.c)
    states=[];residuals=[]
    for response in model.states:
        q=np.zeros((model.no,len(cg)-model.no));q[0,0]=1
        # Character guard follows the already fixed target orbital subspaces.
        q=np.outer(cg[:,:model.no].T@context.state.overlap@response.c[:,0],cg[:,model.no:].T@context.state.overlap@response.c[:,model.no]);q/=np.linalg.norm(q)
        c,mu,e,res,it=refine.roks(context,model.case['eri'],response.c,cg,q,response.multiplier)
        states.append(c);residuals.append(res)
    u=TransportedOperators(context.state,model.s.coordinates_bohr,np.zeros((0,len(cg),len(cg)))).u
    root=metric_power(context.state.overlap,.5);yg=u.T@root@cg;ys=[u.T@root@c for c in states]
    basis=[literal_closed(yg,list(range(model.no)))];diagonal=[eg]
    def energy(c,o):return context.fock_and_energy(np.array([(c*x)@c.T for x in o]))[1]
    for c,y,response in zip(states,ys,model.states):
        basis.append(literal_open(y,model.no));diagonal.append(2*energy(c,response.occupations)-energy(c,response.triplet))
    occ=model.ground.occupations.copy();occ[:,0]=0;occ[:,model.no]=1
    for c,y in zip(states,ys):
        basis.append(literal_closed(y,list(range(1,model.no))+[model.no]));diagonal.append(energy(c,occ))
    count=len(basis);s=np.zeros((count,count));h=np.zeros_like(s)
    for i in range(count):
        for j in range(i+1):
            ov,value=literal_csf(basis[i],basis[j],model.h,model.w)
            s[i,j]=s[j,i]=ov;h[i,j]=h[j,i]=value
    residual=np.asarray(diagonal)-np.diag(h)/np.diag(s)
    h+=.5*s*(residual[:,None]+residual[None,:])
    p=np.eye(count)[:,1:];p[0]-=s[0,1:]/s[0,0]
    e=eigh(p.T@h@p,p.T@s@p,eigvals_only=True)
    return dict(ground=eg,energies={identity:float(e[root]) for identity,root in model.roots.items()},ground_residual=gres,roks_residuals=residuals)

def validate(cohort,name):
    model=NuclearCandidate(cohort,name);destination=model.destination
    with np.load(destination/'nuclear_response.npz') as z:arrays={k:z[k].copy() for k in z.files}
    certificate=json.loads((destination/'hessian_certificate.json').read_text())
    nr=model.nr;vectors=[];unit=np.zeros(nr);unit[3]=1.;vectors.append(unit)
    other=np.sin(np.arange(nr)+.37);other/=np.linalg.norm(other);vectors.append(other)
    results={identity:[] for identity in model.roots}
    for index,direction in enumerate(vectors):
        for eps in [1e-3,5e-4]:
            values=[]
            for sign in [-1,1]:
                folder=destination/'independent_energy_validation';folder.mkdir(exist_ok=True)
                path=folder/f'direction_{index}_eps_{eps:.8f}_sign_{sign}.json'
                if path.exists():value=json.loads(path.read_text())
                else:
                    value=displaced(model,sign*eps*direction);write(path,value)
                values.append(value)
            for identity in model.roots:
                center=certificate['states'][identity]['energy_hartree'];g=arrays[identity+'_gradient'];h=arrays[identity+'_hessian']
                em=values[0]['energies'][identity];ep=values[1]['energies'][identity]
                gradient=(ep-em)/(2*eps);curvature=(ep-2*center+em)/eps**2
                ge=abs(gradient-g@direction);he=abs(curvature-direction@h@direction)
                results[identity].append(dict(direction=index,epsilon_bohr=eps,gradient_error=float(ge),hessian_direction_error=float(he),
                    independent_gradient=float(gradient),analytic_gradient=float(g@direction),independent_curvature=float(curvature),analytic_curvature=float(direction@h@direction)))
                print(name,identity,index,eps,ge,he,flush=True)
    # The raw central differences have a visible O(h^2) error in the strongly
    # curved mixed direction. Validate its convergence and independently
    # extrapolate that truncation term, without changing the analytic Hessian.
    for identity,checks in results.items():
        for index in range(len(vectors)):
            coarse,fine=[x for x in checks if x['direction']==index]
            eg=abs((4*fine['independent_gradient']-coarse['independent_gradient'])/3-fine['analytic_gradient'])
            eh=abs((4*fine['independent_curvature']-coarse['independent_curvature'])/3-fine['analytic_curvature'])
            assert eg<2e-7,(identity,index,eg)
            assert eh<2e-5,(identity,index,eh)
            fine.update(richardson_gradient_error=eg,richardson_hessian_error=eh)
    write(destination/'independent_nuclear_validation.json',dict(host=HOST,states=results,
        implementation='analytic molecular Hessian; displaced calculations used only for independent validation',SCC_runs=0,
        source_sha256=sha(__file__),response_sha256=sha(destination/'nuclear_response.npz'),
        protocol_sha256=sha(Path(__file__).parent/'nuclear_protocol.json')))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cohort');parser.add_argument('molecule');args=parser.parse_args()
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():validate(args.cohort,args.molecule)
