"""Calibrate the prepared response at its actual native GFN2 reference density."""
from pathlib import Path
import sys,importlib.util
PARENT=Path('/home/enzo/MOUSE-cam-response-20261101');sys.path.insert(0,str(PARENT))
from predict import *
from calibrate import features
from native_batch import install_native_batch
ROOT=Path('/home/enzo/MOUSE-native-density-reference-20261101')

def link(source,target):
    assert source.exists(),source;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(source.resolve())

def calibrate_native(cohort,name):
    out=ROOT/cohort/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'native_calibration_complete.json').exists():return
    start=time.monotonic();case=prepare(cohort,name);s=case['s'];c=s.coefficients;no=case['no'];n=len(c)
    install_native_batch(case,PARENT/cohort/name)
    functional=CompactDFT(case,out/'grid');p=2*c[:,:no]@c[:,:no].T;target,energy=functional.evaluate(p)
    np.savez_compressed(out/'embedded_ground.npz',coefficients=c,density=p,fock=target,orbital_energies=np.diag(c.T@target@c),
        energy_up_to_fixed_core_constant=energy,reference_kind='native GFN2 density, not stationary embedded DFT')
    a=case['old']['operators'];b=.5*(s.overlap@a+a@s.overlap);dual=np.array([2*solve_sylvester(s.overlap,s.overlap,x) for x in a]);nf=len(a)
    packed=dual.reshape(nf,n*n);eri=case['eri'];lr=case['eri_lr']
    j=packed@eri.reshape(n*n,n*n)@packed.T;k=packed@eri.transpose(0,2,1,3).reshape(n*n,n*n)@packed.T
    klr=packed@lr.transpose(0,2,1,3).reshape(n*n,n*n)@packed.T
    charge=j-.0855*k-.207*klr;spin=-.0855*k-.207*klr;completed=0
    progress=out/'response_progress.npz'
    if progress.exists():
        with np.load(progress) as z:charge=z['charge'].copy();spin=z['spin'].copy();completed=int(z['completed'])
    for block,(phi,core,weight) in enumerate(functional.chunks):
        if block<completed:continue
        rho=.5*(core+functional.rho(phi,p));fxc=functional.mf._numint.eval_xc_eff('CAM-B3LYP',np.array([rho,rho]),deriv=2,xctype='GGA')[2]
        fc=.25*(fxc[0,:,0]+fxc[0,:,1]+fxc[1,:,0]+fxc[1,:,1]);fs=.25*(fxc[0,:,0]-fxc[0,:,1]-fxc[1,:,0]+fxc[1,:,1])
        for lo in range(0,len(weight),4096):
            hi=min(lo+4096,len(weight));f=features(phi[:,lo:hi],dual);flat=f.transpose(1,0,2).reshape(nf,-1)
            for matrix,fx in [(charge,fc),(spin,fs)]:
                weighted=(np.einsum('ijg,jag->aig',fx[:,:,lo:hi],f,optimize=True)*weight[None,None,lo:hi]).reshape(nf,-1)
                matrix+=flat@weighted.T
        np.savez_compressed(progress,charge=charge,spin=spin,completed=block+1)
        print(name,'native-density calibration',block+1,'/',len(functional.chunks),flush=True)
    state=SimpleNamespace(**vars(s));state.coefficients=c;native_kernel=case['ctx'].density_response_kernel(state)
    native=np.einsum('aij,bij->ab',dual,native_kernel.response_batch(dual))
    folder=out/'F';folder.mkdir(exist_ok=True);native_f,_=case['ctx'].fock_and_energy(np.array([.5*p,.5*p]))
    np.savez_compressed(folder/'model.npz',operators=a,charge=.5*(charge+charge.T)-native,spin=.5*(spin+spin.T),
        q0=np.einsum('aij,ij->a',b,p),linear=np.einsum('aij,ij->a',dual,target-native_f[0]))
    np.savez_compressed(out/'native_response.npz',matrix=native)
    link(PARENT/cohort/name/'CAM_full_reference/reference.npz',out/'CAM_full_reference/reference.npz')
    write(out/'native_calibration_complete.json',dict(host=HOST,reference_density='native GFN2',xc='CAM-B3LYP',
        weights=[.5,.5],lambda_exchange=.9,offset_ev=0,new_DFT_SCF=0,wall_seconds=time.monotonic()-start,
        source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocol.json'),density_sha256=hashlib.sha256(p.tobytes()).hexdigest()))

if __name__=='__main__':
    lock=(ROOT/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        engine_path=Path('/home/enzo/MOUSE-native-reference-20261101/run.py')
        spec=importlib.util.spec_from_file_location('native_reference_engine',engine_path);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
        engine.PARENT=ROOT;engine.NATIVE=ROOT
        import heldout_predict as hp
        hp.DEST=ROOT
        import nosi_coherent as nc
        nc.DEST=ROOT;nc.HERE=ROOT/'nosi_coherent'
        import nosi_projected as npj
        npj.HERE=nc.HERE
        for cohort,name in json.loads((ROOT/'protocol.json').read_text())['cases']:
            if not (PARENT/'nosi_coherent'/cohort/name/'projected_result.json').exists():
                print('Awaiting completed parent',cohort,name,flush=True);continue
            calibrate_native(cohort,name)
            engine.prepare_native(cohort,name)
            hp.predict_molecule(cohort,name)
            nc.run(cohort,name)
            npj.projected(cohort,name)
