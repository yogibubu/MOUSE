"""At most three two-thread Oracle science jobs; cost jobs retain exclusive access."""
from pathlib import Path
import fcntl,time,socket
assert socket.getfqdn()=='oracle.sns.it'
R=Path(__file__).resolve().parent

def science_locks():
 while True:
  for i in range(3):
   slot=(R/f'science_slot_{i}.lock').open('a')
   try:fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:slot.close();continue
   shared=Path('/home/enzo/MOUSE-publication-validation-20261030/blind/references/compute.lock').open('a');fcntl.flock(shared,fcntl.LOCK_SH)
   return slot,shared
  time.sleep(1)
