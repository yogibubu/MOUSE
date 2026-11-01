"""Oracle molecular exactness and full orbital-Hessian benchmark for native batch."""
from native_batch import *
sys.path.insert(0,'/home/enzo/MOUSE-transferability-20261101')
from matrix_gfn2.relative_pair_roks import load_relative_pair_roks
from matrix_gfn2.multicsf import MultiCSFObjective

def validate(cohort,name,state_id):
    out=DEST/cohort/name;case=prepare(cohort,name);s=case['s'];no=case['no']
    point=load_relative_pair_roks(case['src']/state_id/'stationary.npz',request_hash='independent:relative',require_gradient=False)
    original,fast=install_native_batch(case,out);rng=np.random.default_rng(47);states=[]
    with np.load(out/'embedded_ground.npz') as a:c=a['coefficients'].copy()
    states.append((c,np.array([s.alpha_occupations,s.beta_occupations])))
    for component in point.components:states.append((component.common_coefficients,component.occupations))
    errors=[]
    for c,occ in states:
        state=SimpleNamespace(**vars(s));state.coefficients=c;state.alpha_occupations,state.beta_occupations=occ
        dp=rng.normal(size=(5,len(c),len(c)));dp=.5*(dp+dp.transpose(0,2,1))
        literal=np.array([original(state)(x) for x in dp]);batched=fast(state).response_batch(dp)
        errors.append(float(np.max(abs(literal-batched))))
    assert max(errors)<1e-10,errors
    with np.load(out/'F/model.npz') as a:data={k:a[k].copy() for k in ['operators','charge','spin','q0','linear']}
    response=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(out/'F/model.npz'))
    ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=response)
    c=point.components[0].common_coefficients;ground_c=states[0][0];q=np.zeros((no,len(c)-no));q[0,0]=1
    timings=[];values=[]
    for kernel in [original,fast]:
        case['calc'].density_response_kernel=kernel;obj=MultiCSFObjective(ctx,case['eri'],no,1,ground_c,q)
        started=time.monotonic();values.append(obj.evaluate(c,True));timings.append(time.monotonic()-started)
    differences=[float(np.max(abs(a-b))) for a,b in zip(values[0],values[1])]
    assert max(differences)<1e-10,differences
    write(out/'native_batch_validation.json',dict(host=HOST,threads=2,reference_density_kernel_max_errors=errors,
        full_energy_gradient_orbital_hessian_errors=differences,literal_seconds=timings[0],batched_seconds=timings[1],
        speedup=timings[0]/timings[1],source_sha256=sha(__file__),batch_source_sha256=sha(Path(__file__).parent/'native_batch.py'),
        scope='fixed geometry, exact ordinary-GFN2 native response; not a new physical approximation'))
    print(name,'native batch verified',timings,flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        validate('independent','Criegee_Intermediate','I002')
        validate('independent_challenge','Furofuran','C001')
