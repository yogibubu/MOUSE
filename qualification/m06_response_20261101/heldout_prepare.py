from anchor import *
from anchor import anchor
from calibrate import calibrate
from full_reference import full_reference

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        for name in ['Benzoxadiazole','Hexatriene']:
            anchor('independent',name)
            calibrate('independent',name)
            full_reference('independent',name)
