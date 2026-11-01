"""Oracle molecular finite-difference validation; implementation is analytic."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/enzo/MOUSE-m06-response-20261101')
from predict import *
from mixed_fock import MixedFock,Jet

def run():
    cohort='independent';name='Criegee_Intermediate';case=prepare(cohort,name);out=DEST/cohort/name;s=case['s'];no=case['no']
    model=out/'coherent_common/model.npz'
    with np.load(model) as z:data={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
    reference=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(model))
    context=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=reference)
    with np.load(out/'coherent_states/I002/final/final.npz') as z:c=z['coefficients'].copy()
    occ=np.zeros((2,len(c)));occ[:,:no]=1;occ[0,0]=0;occ[0,no]=1
    dm=np.array([(c*x)@c.T for x in occ]);nr=s.coordinates_bohr.size
    rng=np.random.default_rng(19);first=rng.normal(size=(nr,*dm.shape))*.001;first=.5*(first+first.transpose(0,1,3,2))
    direction=np.zeros(nr);direction[3]=1
    jets=[Jet(dm[i],first[:,i],first[3,i]) for i in range(2)]
    f=MixedFock(context).evaluate(jets,direction)
    exact=context.fock_and_energy(dm)[0];primal=float(np.max(abs(np.array([x.v for x in f])-exact)))
    assert primal<1e-7,primal
    checks=[]
    for eps in [2e-4,1e-4]:
        values=[]
        for sign in [-1,1]:
            xyz=case['xyz']+sign*eps*direction.reshape(-1,3)/ANGSTROM_TO_BOHR
            calc=calculator(s);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC forbidden in validation'))
            displaced=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=reference)
            values.append(displaced.fock_and_energy(dm+sign*eps*first[3])[0])
        derivative=(values[1]-values[0])/(2*eps);curvature=(values[1]-2*exact+values[0])/eps**2
        gerror=float(np.max(abs(derivative-np.array([x.a[3] for x in f]))));herror=float(np.max(abs(curvature-np.array([x.ab[3] for x in f]))))
        assert gerror<1e-6,(eps,gerror)
        assert herror<2e-5,(eps,herror)
        checks.append(dict(epsilon_bohr=eps,first_error=gerror,second_error=herror))
    path=Path('/home/enzo/MOUSE-nosi-20261101/mixed_fock_validation.json')
    write(path,dict(host=HOST,primal_error=primal,checks=checks,source_sha256=sha(__file__),implementation_sha256=sha(path.parent/'mixed_fock.py'),model_sha256=sha(model),SCC_runs=0))
    print(primal,checks,flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():run()
