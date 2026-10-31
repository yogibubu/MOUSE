"""Development-only constrained orbital optimization, verified Oracle only."""
from pair_diagnostic import *
from orthogonal_singlet import ground_overlap_jet,exponential_pullback
from matrix_gfn2.multicsf import MultiCSFObjective
from scipy.linalg import expm,expm_frechet
from scipy.optimize import minimize
import argparse

def solve(context,eri,no,c0,ground,transition,out):
    out.mkdir(parents=True,exist_ok=True)
    obj=MultiCSFObjective(context,eri,no,1,ground,transition)
    left,right=obj.left,obj.right;n=len(c0);cache={};history=[]
    def evaluate(x):
        if 'x' in cache and np.array_equal(x,cache['x']):return cache
        k=np.zeros((n,n));k[left,right]=x;k[right,left]=-x
        rotation=expm(k);c=c0@rotation;e,g,_=obj.evaluate(c,False)
        ov,dov=ground_overlap_jet(ground,c,context.state.overlap,obj.occupations[0],left,right)
        def pullback(local):
            return exponential_pullback(k,rotation,left,right,local)
        cache.clear();cache.update(x=x.copy(),c=c,e=e,g=pullback(g),overlap=ov,d_overlap=pullback(dov),fidelity=obj.fidelity(c)[1])
        return cache
    def callback(x):
        a=evaluate(x);history.append(dict(iteration=len(history),energy=a['e'],overlap=a['overlap'],fidelity=a['fidelity']))
        np.savez_compressed(out/'checkpoint.npz',x=x,coefficients=a['c'])
        write(out/'history.json',history)
    x=np.zeros(len(left))
    if (out/'checkpoint.npz').exists():
        with np.load(out/'checkpoint.npz') as a:x=a['x'].copy()
    solved=minimize(lambda x:evaluate(x)['e'],x,jac=lambda x:evaluate(x)['g'],method='SLSQP',
        constraints=[dict(type='eq',fun=lambda x:evaluate(x)['overlap'],jac=lambda x:evaluate(x)['d_overlap'])],
        callback=callback,options=dict(maxiter=180,ftol=1e-15,disp=False))
    a=evaluate(solved.x);gg=a['g'];j=a['d_overlap'];multiplier=float(j@gg/max(j@j,1e-300));res=gg-multiplier*j
    np.savez_compressed(out/'final.npz',coefficients=a['c'],x=solved.x)
    result=dict(success=bool(solved.success),message=solved.message,iterations=int(solved.nit),energy_hartree=a['e'],
        ground_overlap_squared=a['overlap']**2,fidelity=a['fidelity'],kkt_max=float(np.max(abs(res))),
        multiplier=multiplier,qualified=bool(solved.success and abs(a['overlap'])<1e-8 and a['fidelity']>=.7 and np.max(abs(res))<1e-7))
    write(out/'result.json',result);return result

def run(cohort,name,state_id):
    src=R/cohort/'predictions'/name;dest=DEST/'orthogonal'/cohort/name/state_id
    raw=loadpoint(src/'native_ground.npz');s=canonical_state(raw);no=s.electron_count//2
    calc=calculator(s);xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR;context=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)
    joint=load_relative_pair_roks(src/state_id/'stationary.npz',request_hash='independent:relative',require_gradient=False)
    with np.load(src/'full_response/anchor/projected_eri.npz') as a:eri=a['eri'].copy()
    parents=[]
    for i,profile in enumerate(['full_response','qualified_spectral_response']):
        p=src/profile/'anchor/model.npz'
        with np.load(p) as a:d={k:a[k].copy() for k in ['operators','charge','spin','linear','q0']}
        response=FrozenNonlocalResponseReference.from_vertical_model(s,context.kernel.partition,d,sha(p))
        assert (src/f'grounds/{i}/certificate.json').exists()
        g=ground(src/f'grounds/{i}',calc,xyz,s,response)
        ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=response)
        # Follow the retained OO transition, with no excitation-energy labels.
        old=joint.components[i].common_coefficients
        q=np.outer(g.coefficients[0][:,:no].T@s.overlap@old[:,0],g.coefficients[0][:,no:].T@s.overlap@old[:,no])
        q/=np.linalg.norm(q)
        result=solve(ctx,eri,no,joint.components[i].common_coefficients,g.coefficients[0],q,dest/f'parent_{i}')
        result['gap_ev']=(result['energy_hartree']-g.energy_hartree)*EV;parents.append(result)
    write(dest/'result.json',dict(id=state_id,molecule=name,parents=parents,excitation_ev=.5*sum(p['gap_ev'] for p in parents),
        qualified=all(p['qualified'] for p in parents),energy_labels_used=False,weights=[.5,.5],lambda_exchange=.9,offset=0,
        scope='separately optimized orthogonal parents; relative-parent frame and nuclear response not yet integrated'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('cohort');a.add_argument('name');a.add_argument('state');args=a.parse_args()
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():run(args.cohort,args.name,args.state)
