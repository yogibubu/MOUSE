"""Resume a protocol-defined coherent-screening validation without new native jobs."""
from pathlib import Path
import argparse,socket,json,subprocess,sys
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--protocol',required=True);p.add_argument('--threads',type=int,default=4);a=p.parse_args();definition=json.loads((HERE/a.protocol).read_text());ns=definition['coherent_frontiers'][a.name];w=definition['fixed_screening_frontier'];prefix=definition['artifact_prefixes'][a.name]
def command(args,log):
 with (HERE/log).open('a') as f:subprocess.run([sys.executable,'-u',*args],cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True)
for n,screen in dict.fromkeys([(n,n) for n in ns]+[(n,w) for n in ns]):
 command(['build_transition_screening.py',a.name,'--frontier',str(screen),'--threads',str(a.threads)],f'{prefix}_screening_W{screen}.log')
 command(['run_pigment_cid.py',a.name,'--frontier',str(n),'--screening-frontier',str(screen),'--screening-mode','coherent_transitions','--roots','40','--threads',str(a.threads)],f'{prefix}_N{n}_W{screen}.log')
command(['summarize_transition_spectra.py',a.name,'--protocol',a.protocol],f'{prefix}_assessment_pipeline.log')
