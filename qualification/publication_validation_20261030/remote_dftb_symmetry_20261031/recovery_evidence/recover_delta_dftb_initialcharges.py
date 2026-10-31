"""Bounded solver-only recovery of the two incomplete Delta-DFTB cases.

Same geometry, SCC tolerance, mio-1-1 Hamiltonian and integer occupations.
Reuse completed TD ground charges through InitialCharges (binary restart is
unsupported by the backend); never rerun a completed Delta calculation.
"""
from pathlib import Path
import socket,json,shutil,fcntl,subprocess,os,time,signal,re
assert socket.getfqdn()=='oracle.sns.it'
import run_dftbplus as base
P=Path('/home/enzo/MOUSE-publication-validation-20261030')
O=P/'external/dftbplus24/attempt_06';O.mkdir(exist_ok=True)
trials=[('anderson',2000,'Mixer = Anderson { MixingParameter = 0.01\n InitMixingParameter = 0.01\n Generations = 4 }'),
        ('simple_002',20000,'Mixer = Simple { MixingParameter = 0.02 }'),
        ('simple_0005',50000,'Mixer = Simple { MixingParameter = 0.005 }')]
protocol=dict(host=socket.getfqdn(),source_sha256=base.sha(Path(__file__)),
              jobs=['cyclopentadienone','uracil'],solver_trials=[dict(name=n,maximum_scc_iterations=i,mixer=m) for n,i,m in trials],
              timeout_per_trial_seconds=60,scf_tolerance=1e-10,threads=2,unchanged_hamiltonian=True,
              energy_inputs_to_solver_choice=False,ground_charge_restart='atomic InitialCharges from completed TD ground output; binary charge restart is explicitly unsupported by Delta DFTB',
              failed_prior_attempts_retained=True,method='same spin-purified HOMO-LUMO NonAufbau, no smearing or empirical change')
path=O/'protocol.json'
if path.exists():assert json.loads(path.read_text())==protocol
else:base.write(path,protocol)
lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
for name in protocol['jobs']:
    prior=P/'external/dftbplus24/attempt_04'/name/'delta'
    assert json.loads((prior/'result.json').read_text())['returncode']!=0
    dest=O/name/'delta';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists() and json.loads((dest/'result.json').read_text()).get('returncode')==0:continue
    completed=None
    for variant,niter,mixer in trials:
        d=dest/variant;d.mkdir(exist_ok=True)
        if (d/'result.json').exists():
            if json.loads((d/'result.json').read_text()).get('returncode')==0:completed=d;break
            continue
        geometry=prior/'geometry.xyz';shutil.copy2(geometry,d/'geometry.xyz')
        ground=P/'external/dftbplus24/attempt_03'/name/'td/charges.bin'
        textground=(ground.parent/'detailed.out').read_text().split('Atomic gross charges (e)',1)[1].split('Nr. of electrons',1)[0]
        charges=[float(q) for q in re.findall(r'^\s*\d+\s+([-+0-9.Ee]+)\s*$',textground,re.M)]
        assert len(charges)==int(geometry.read_text().splitlines()[0])
        # Remove only the rounding error in the printed neutral charge guess.
        shift=sum(charges)/len(charges);charges=[q-shift for q in charges]
        initial=' InitialCharges = { AllAtomCharges = { '+' '.join(f'{q:.14g}' for q in charges)+' } }'

        text=base.inp(geometry,'delta').replace('Mixer = Broyden { MixingParameter = 0.1 }',mixer)
        text=text.replace('MaxSCCIterations = 300',f'MaxSCCIterations = {niter}').replace('SCC = Yes','SCC = Yes\n'+initial,1)
        (d/'dftb_in.hsd').write_text(text);start=time.monotonic()
        with (d/'stdout.log').open('w') as log:
            proc=subprocess.Popen([str(base.B)],cwd=d,env=base.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=proc.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=-9
        result=dict(host=socket.getfqdn(),molecule=name,method='delta',returncode=rc,
                    wall_seconds=time.monotonic()-start,variant=variant,protocol_sha256=base.sha(path),
                    geometry_sha256=base.sha(geometry),input_sha256=base.sha(d/'dftb_in.hsd'),
                    initial_ground_charge_sha256=base.sha(ground.parent/'detailed.out'),initial_atomic_charges=charges,prior_checkpoint_retained=str(prior/'charges.bin'))
        base.write(d/'result.json',result);print(name,variant,rc,flush=True)
        if rc==0:completed=d;break
    if completed is not None:
        # A common accepted directory is consumed only after a successful run.
        for filename in ['geometry.xyz','eigenvec.out','results.tag','detailed.out','dftb_in.hsd','stdout.log','result.json','charges.bin']:
            if (completed/filename).exists():shutil.copy2(completed/filename,dest/filename)
    else:base.write(dest/'result.json',dict(host=socket.getfqdn(),molecule=name,method='delta',returncode=1,status='bounded_solver_recoveries_incomplete',protocol_sha256=base.sha(path)))
