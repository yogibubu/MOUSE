from nosi_coherent import *
from nosi_projected import projected

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        for name in ['Benzoxadiazole','Hexatriene']:
            run('independent',name)
            projected('independent',name)
