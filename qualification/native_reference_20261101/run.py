"""Native-orbital-preserving DFT reference; no new electronic DFT jobs."""
from pathlib import Path
import sys,os
PARENT=Path(os.environ.get('MOUSE_PARENT_REFERENCE_ROOT','/home/enzo/MOUSE-cam-response-20261101'))
sys.path.insert(0,str(PARENT))
from predict import *
from native_batch import install_native_batch
NATIVE=Path('/home/enzo/MOUSE-native-reference-20261101')

def link(source,target):
    assert source.exists(),source;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(source.resolve())

def prepare_native(cohort,name):
    case=prepare(cohort,name);s=case['s'];no=case['no'];c=s.coefficients;n=len(c)
    source=PARENT/cohort/name;out=NATIVE/cohort/name;out.mkdir(parents=True,exist_ok=True)
    for filename in ['F/model.npz','native_response.npz','embedded_ground.npz']:link(source/filename,out/filename)
    common=out/'coherent_common';common.mkdir(exist_ok=True);path=common/'model.npz'
    if not path.exists():
        with np.load(source/'F/model.npz') as z:d={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
        with np.load(source/'embedded_ground.npz') as z:ff=z['fock'].copy()
        with np.load(source/'CAM_full_reference/reference.npz') as z:cd=z['coefficients'].copy();od=z['occupations'].copy();ed=z['orbital_energies'].copy()
        ef=np.diag(c.T@ff@c).copy();eq=ef.copy();nc=np.count_nonzero(od==2)-no
        cross=c.T@case['lift'].T@case['sd']@cd;selected=np.zeros(n,int);score=np.zeros(n);margin=np.zeros(n)
        for ids,refs in [(np.arange(no),np.flatnonzero(od==2)[nc:]),(np.arange(no,n),np.flatnonzero(od==0))]:
            ov=cross[np.ix_(ids,refs)];_,columns=linear_sum_assignment(-ov**2)
            for i,index in enumerate(ids):
                selected[index]=refs[columns[i]];score[index]=abs(ov[i,columns[i]])
                margin[index]=score[index]-np.max(np.delete(abs(ov[i]),columns[i]),initial=0.)
        good=(score>=.7)&(margin>=.05);origin=np.mean(ed[selected[:no]])
        eq[good]=ed[selected[good]]-origin+np.mean(ef[:no]);fallback=bool(np.max(eq[:no])>=np.min(eq[no:]))
        if fallback:eq=ef.copy()
        levels=.5*(ef+eq);assert np.max(levels[:no])<np.min(levels[no:])
        target=(s.overlap@c*levels)@c.T@s.overlap;p=2*c[:,:no]@c[:,:no].T
        native_f,_=case['ctx'].fock_and_energy(np.array([.5*p,.5*p]));b=.5*(s.overlap@d['operators']+d['operators']@s.overlap)
        dual=np.array([2*solve_sylvester(s.overlap,s.overlap,a) for a in d['operators']])
        data=dict(d,q0=np.einsum('aij,ij->a',b,p),linear=np.einsum('aij,ij->a',dual,target-native_f[0]))
        np.savez_compressed(path,**data)
        write(common/'reference_certificate.json',dict(native_orbitals=True,F_levels=ef,Q_levels=eq,common_levels=levels,mapping=selected,overlaps=score,margins=margin,
            qualified_mapping=good,molecular_Q_fallback=fallback,parent_model_sha256=sha(source/'F/model.npz'),DFT_reference_sha256=sha(source/'CAM_full_reference/reference.npz'),
            source_sha256=sha(__file__),protocol_sha256=sha(NATIVE/'protocol.json'),weights=[.5,.5],lambda_exchange=.9,offset_ev=0))
    install_native_batch(case,out)
    with np.load(path) as z:d={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
    r=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,d,sha(path))
    ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=r)
    p= c[:,:no]@c[:,:no].T;f,e=ctx.fock_and_energy(np.array([p,p]))
    seed=replace(case['raw'],coefficients=np.array([c,c]),fock=f,energy_hartree=e,spin_reference_fingerprint=r.fingerprint,reference_state=s,converged=True)
    g=ground(common/'ground',case['calc'],case['xyz'],s,r,seed=seed)
    error=float(np.max(abs(g.density_ao-2*p)));assert error<1e-7,(name,error)
    write(common/'native_ground_certificate.json',dict(native_density_error=error,minimum_ground_curvature=g.minimum_orbital_curvature_hartree,
        orbital_residual=g.orbital_residual,source_sha256=sha(__file__)))
    return case

if __name__=='__main__':
    # This is an independent, prepared study on the explicitly verified host.
    lock=(NATIVE/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        import heldout_predict as hp
        hp.DEST=NATIVE
        import nosi_coherent as nc
        nc.DEST=NATIVE;nc.HERE=NATIVE/'nosi_coherent'
        import nosi_projected as npj
        npj.HERE=nc.HERE
        protocol=json.loads((NATIVE/'protocol.json').read_text())
        for cohort,name in protocol['cases']:
            if not (PARENT/'nosi_coherent'/cohort/name/'projected_result.json').exists():
                print('Awaiting completed parent',cohort,name,flush=True);continue
            prepare_native(cohort,name)
            hp.predict_molecule(cohort,name)
            nc.run(cohort,name)
            npj.projected(cohort,name)
