"""Resume one additional frontier, then qualify its subtracted excitation roots."""
from pathlib import Path
import argparse,socket,sys,subprocess
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--frontier',type=int,required=True);p.add_argument('--screening-frontier',type=int);p.add_argument('--threads',type=int,default=4);a=p.parse_args();w=a.screening_frontier or a.frontier
out=HERE/'static_subtraction_frontiers'/a.name/f'N{a.frontier}_W{w}';out.mkdir(parents=True,exist_ok=True)
commands=[['build_transition_screening.py',a.name,'--frontier',str(w),'--threads',str(a.threads)],['run_pigment_cid.py',a.name,'--frontier',str(a.frontier),'--screening-frontier',str(w),'--screening-mode','coherent_transitions','--roots','40','--threads',str(a.threads)],['check_static_subtraction.py',a.name,'--frontier',str(a.frontier),'--screening-frontier',str(w),'--eigensolver','davidson','--roots','20','--threads',str(a.threads),'--output',str(out/'result.json')],['qualify_subtracted_observables.py','--diagnostic',str(out/'result.json')]]
for i,command in enumerate(commands):
 with (out/f'stage_{i+1}.log').open('a') as log:subprocess.run([sys.executable,'-u',*command],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True)
