"""Global separation of DFT orbital-level and response references; no fitting."""
from pathlib import Path
import sys
CAM=Path('/home/enzo/MOUSE-cam-response-20261101');M06=Path('/home/enzo/MOUSE-m06-response-20261101')
LEVELS=Path('/home/enzo/MOUSE-native-reference-20261101');ROOT=Path('/home/enzo/MOUSE-mixed-reference-20261101')
sys.path.insert(0,str(CAM))
from predict import *
from native_batch import install_native_batch

def link(source,target):
    assert source.exists(),source;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(source.resolve())

def prepare_mixed(cohort,name):
    out=ROOT/cohort/name;out.mkdir(parents=True,exist_ok=True);source=M06/cohort/name;levels=LEVELS/cohort/name
    for filename in ['F/model.npz','native_response.npz','embedded_ground.npz']:link(source/filename,out/filename)
    model=out/'coherent_common/model.npz';model.parent.mkdir(exist_ok=True)
    if not model.exists():
        with np.load(source/'F/model.npz') as z:response={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
        with np.load(levels/'coherent_common/model.npz') as z:onebody={k:z[k].copy() for k in response}
        assert np.array_equal(response['operators'],onebody['operators'])
        np.savez_compressed(model,**dict(onebody,charge=response['charge'],spin=response['spin']))
        write(out/'coherent_common/reference_certificate.json',dict(one_body_source_sha256=sha(levels/'coherent_common/model.npz'),response_source_sha256=sha(source/'F/model.npz'),
            source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocol.json'),energy_labels_used=False,weights=[.5,.5],lambda_exchange=.9,offset_ev=0))
    case=prepare(cohort,name);s=case['s'];c=s.coefficients;no=case['no'];install_native_batch(case,out)
    with np.load(model) as z:d={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
    reference=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,d,sha(model))
    ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=reference)
    p=c[:,:no]@c[:,:no].T;f,e=ctx.fock_and_energy(np.array([p,p]))
    seed=replace(case['raw'],coefficients=np.array([c,c]),fock=f,energy_hartree=e,spin_reference_fingerprint=reference.fingerprint,reference_state=s,converged=True)
    g=ground(out/'coherent_common/ground',case['calc'],case['xyz'],s,reference,seed=seed)
    error=float(np.max(abs(g.density_ao-2*p)));assert error<1e-7
    write(out/'coherent_common/native_ground_certificate.json',dict(native_density_error=error,minimum_ground_curvature=g.minimum_orbital_curvature_hartree,orbital_residual=g.orbital_residual))

if __name__=='__main__':
    lock=(ROOT/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        import heldout_predict as hp
        hp.DEST=ROOT
        import nosi_coherent as nc
        nc.DEST=ROOT;nc.HERE=ROOT/'nosi_coherent'
        import nosi_projected as npj
        npj.HERE=nc.HERE
        for cohort,name in json.loads((ROOT/'protocol.json').read_text())['cases']:
            if not (LEVELS/cohort/name/'coherent_states').exists():
                print('Awaiting native orbital-level model',cohort,name,flush=True);continue
            prepare_mixed(cohort,name);hp.predict_molecule(cohort,name);nc.run(cohort,name);npj.projected(cohort,name)
