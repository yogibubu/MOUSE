"""Resume all stages; completed molecular and state checkpoints are retained."""
from anchor import *
from anchor import anchor
from calibrate import calibrate
from predict_common import run

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        for cohort,name in [('independent','Criegee_Intermediate'),('independent_challenge','Furofuran')]:
            anchor(cohort,name)
            calibrate(cohort,name)
        for cohort,name in [('independent','Criegee_Intermediate'),('independent_challenge','Furofuran')]:
            run(cohort,name)
