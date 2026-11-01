"""Same-functional TDA on native GFN2 spatial ground orbitals."""
from pathlib import Path
import sys,time
ROOT=Path('/home/enzo/MOUSE-frontier-cisd-20261101');NATIVE=Path('/home/enzo/MOUSE-native-reference-20261101')
sys.path.insert(0,'/home/enzo/MOUSE-cam-response-20261101')
from predict import *
from native_batch import install_native_batch

def run(cohort,name):
    destination=ROOT/cohort/name/'tda_control.json'
    if destination.exists():return
    if not (ROOT/cohort/name/'result.json').exists():return
    case=prepare(cohort,name);source=NATIVE/cohort/name;s=case['s'];no=case['no'];install_native_batch(case,source)
    model=source/'coherent_common/model.npz'
    with np.load(model) as z:d={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
    reference=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,d,sha(model));ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=reference)
    with np.load(source/'coherent_common/ground/stationary.npz') as z:g=SimpleNamespace(fock=z['result_fock'].copy(),coefficients=z['result_coefficients'].copy())
    _,c=eigh(g.fock[0],s.overlap);cross=abs(c.T@s.overlap@s.coefficients);ii,jj=linear_sum_assignment(-cross**2);overlap=float(np.min(cross[ii,jj]));assert overlap>1-1e-7
    rows=[r for r in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if r['molecule']==name];predictions=[]
    exchange=.9*(.19*case['eri']+.46*case['eri_lr'])
    for row in rows:
        transition=target_transition(case,row);start=time.monotonic();result=tda(ctx,g,case['eri'],s,no,transition,exchange_eri=exchange);result.update(id=row['id'],wall_seconds=time.monotonic()-start);predictions.append(result)
    write(destination,dict(predictions=predictions,minimum_native_MO_overlap=overlap,DFT_spatial_orbitals_used=False,lambda_exchange=.9,weights=[.5,.5],model_sha256=sha(model),source_sha256=sha(__file__),host=HOST))
if __name__=='__main__':
    with threadpool_limits():
        p=json.loads((ROOT/'protocol.json').read_text())
        for cohort,name in p['cases']+[['prospective_cisd',r['molecule']] for r in p['prospective_cases']]:run(cohort,name)
