"""Balanced two-root singlet pair-CAS diagnostic; exposed development only."""
from pair_diagnostic import *
from pair_cas_response import pair_cas_response
from orthogonal_singlet import exponential_pullback
from scipy.linalg import expm
from scipy.optimize import minimize

def solve(context,no,ground,q,out):
    out.mkdir(parents=True,exist_ok=True)
    u,sv,vh=np.linalg.svd(q,full_matrices=True)
    c0=np.column_stack([ground[:,:no]@u,ground[:,no:]@vh.T]);n=len(c0)
    left,right=np.triu_indices(n,1);cache={}
    history=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
    def evaluate(x):
        if 'x' in cache and np.array_equal(x,cache['x']):return cache
        k=np.zeros((n,n));k[left,right]=x;k[right,left]=-x
        rotation=expm(k);c=c0@rotation
        values,z,gradients,matrix,_,_=pair_cas_response(context,c,no)
        local=.5*(gradients[0]+gradients[1])
        g=exponential_pullback(k,rotation,left,right,local)
        h=ground[:,:no].T@context.state.overlap@c[:,0]
        p=ground[:,no:].T@context.state.overlap@c[:,no]
        character=abs(float(h@q@p))/max(np.linalg.norm(h)*np.linalg.norm(p)*np.linalg.norm(q),1e-300)
        cache.clear();cache.update(x=x.copy(),c=c,energy=float(.5*(values[0]+values[1])),g=g,
            local=local,values=values,z=z,matrix=matrix,character=character)
        return cache
    def callback(x):
        a=evaluate(x);history.append(dict(iteration=len(history),average_energy=a['energy'],gap_ev=(a['values'][1]-a['values'][0])*EV,
            orbital_residual=float(np.max(abs(a['local']))),open_singlet_weight=float(a['z'][1,1]**2),transition_character=a['character']))
        write(out/'history.json',history);np.savez_compressed(out/'checkpoint.npz',x=x,coefficients=a['c'])
    x=np.zeros(len(left))
    if (out/'checkpoint.npz').exists():
        with np.load(out/'checkpoint.npz') as a:x=a['x'].copy()
    initial=evaluate(x)
    solved=minimize(lambda x:evaluate(x)['energy'],x,jac=lambda x:evaluate(x)['g'],method='BFGS',callback=callback,
        options=dict(maxiter=150,gtol=1e-8))
    a=evaluate(solved.x);res=float(np.max(abs(a['local'])));weight=float(a['z'][1,1]**2)
    np.savez_compressed(out/'final.npz',coefficients=a['c'],matrix=a['matrix'],energies=a['values'],ci=a['z'])
    result=dict(qualified=bool(res<1e-7 and weight>.5 and a['character']>=.7),optimizer_success=bool(solved.success),
        message=solved.message,gap_ev=float((a['values'][1]-a['values'][0])*EV),energies=a['values'],
        orbital_residual=res,open_singlet_weight=weight,transition_character=a['character'],iterations=len(history),
        ci_orthogonality_error=float(np.max(abs(a['z'].T@a['z']-np.eye(3)))),
        active_pair=[0,no],average_weights=[.5,.5],source_sha256=sha(__file__),
        response_source_sha256=sha(Path(__file__).parent/'pair_cas_response.py'),energy_labels_used=False)
    write(out/'result.json',result);return result

def run(cohort,name,state_id):
    src=R/cohort/'predictions'/name;dest=DEST/'state_average'/cohort/name/state_id
    raw=loadpoint(src/'native_ground.npz');s=canonical_state(raw);no=s.electron_count//2
    calc=calculator(s);xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR;context=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)
    joint=load_relative_pair_roks(src/state_id/'stationary.npz',request_hash='independent:relative',require_gradient=False)
    parents=[]
    for i,profile in enumerate(['full_response','qualified_spectral_response']):
        p=src/profile/'anchor/model.npz'
        with np.load(p) as a:d={k:a[k].copy() for k in ['operators','charge','spin','linear','q0']}
        response=FrozenNonlocalResponseReference.from_vertical_model(s,context.kernel.partition,d,sha(p))
        assert (src/f'grounds/{i}/certificate.json').exists()
        g=ground(src/f'grounds/{i}',calc,xyz,s,response)
        ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=response)
        old=joint.components[i].common_coefficients
        q=np.outer(g.coefficients[0][:,:no].T@s.overlap@old[:,0],g.coefficients[0][:,no:].T@s.overlap@old[:,no]);q/=np.linalg.norm(q)
        calc.evaluate=lambda *args,**kw:(_ for _ in ()).throw(AssertionError('Unexpected SCC in prepared diagnostic'))
        parents.append(solve(ctx,no,g.coefficients[0],q,dest/f'parent_{i}'))
    write(dest/'result.json',dict(id=state_id,molecule=name,parents=parents,qualified=all(p['qualified'] for p in parents),
        excitation_ev=.5*sum(p['gap_ev'] for p in parents),weights=[.5,.5],lambda_exchange=.9,offset=0,
        host=HOST,threads=2,energy_labels_used=False,scope='state-specific active-pair, two-state averaged orbital optimization; diagnostic only'))
    print(name,state_id,'balanced pair diagnostic complete',flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        for cohort,name,state in [('independent','Criegee_Intermediate','I002'),('independent_challenge','Furofuran','C002')]:
            run(cohort,name,state)
