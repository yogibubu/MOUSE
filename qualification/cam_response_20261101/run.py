from anchor import *
from anchor import anchor
from calibrate import calibrate
from full_reference import full_reference
from heldout_predict import predict_molecule
from nosi_coherent import run as nosi
from nosi_projected import projected

def dataset():
    root=DEST/'data';root.mkdir(exist_ok=True)
    for cohort in ['independent','independent_challenge']:
        path=root/cohort
        if not path.exists():path.symlink_to(R/cohort,target_is_directory=True)
    path=root/'extra_npi'
    if not path.exists():path.symlink_to(Path('/home/enzo/MOUSE-m06-response-20261101/extra_dataset/extra_npi'),target_is_directory=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        dataset()
        protocol=json.loads((DEST/'protocol.json').read_text())
        for cohort,name in protocol['cases']:
            anchor(cohort,name)
            full_reference(cohort,name,prepare_only=True)
            calibrate(cohort,name)
            full_reference(cohort,name)
            predict_molecule(cohort,name)
            nosi(cohort,name)
            projected(cohort,name)
