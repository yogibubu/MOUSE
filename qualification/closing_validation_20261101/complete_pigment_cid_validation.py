"""Resume a data-defined pigment spectral-convergence protocol remotely."""
from pathlib import Path
import socket,subprocess,sys,time,json,os,contextlib,functools,argparse
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
assert subprocess.check_output(['scutil','--get','ComputerName'],text=True).strip()=='Mac Studio di Vincenzo'
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);args=parser.parse_args()
protocol=json.loads((HERE/args.protocol).read_text());name=protocol['molecule'];prefix=protocol['artifact_prefix']
baseline_ns=protocol['baseline_frontiers'];baseline_w=protocol['baseline_screening_frontier'];fixed_w=protocol['fixed_screening_frontier'];history_count=protocol['successive_increments_required']+1
def folder(n,w):
 return HERE/('pigment_cid/'+name if n==2 and w==5 else f'pigment_cid_expansion/{name}/'+('' if w==5 else f'screening_{w}/')+f'frontier_{n}')
cid_module=None
def command(args,log):
 global cid_module
 with (HERE/log).open('a') as f:
  if args[0]=='run_pigment_cid.py':
   # Reuse the immutable prepared context across CI windows. Every window
   # replaces its registered density span; full quadrature remains unchanged.
   if cid_module is None:
    import run_pigment_cid as cid_module
    cid_module.load=functools.lru_cache(maxsize=1)(cid_module.load)
   with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f),cid_module.threadpool_limits(8):
    cid_module.run(name,int(args[args.index('--frontier')+1]),int(args[args.index('--screening-frontier')+1]),int(args[args.index('--roots')+1]),checkpoint_batch_size=protocol.get('checkpoint_batch_size',512))
  else:subprocess.run([sys.executable,'-u',*args],cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True)
def calc(n,w):
 screen=HERE/(f'pigment_screening/{name}' if w==5 else f'pigment_screening_expansion/{name}/frontier_{w}')
 if not (screen/'certificate.json').exists():command(['prepare_pigment_screening.py',name,'--frontier',str(w)],f'{prefix}_screening_W{w}.log')
 out=folder(n,w)
 if not (out/'result.json').exists():
  for count in protocol['eigensolver_root_limits']:
   try:command(['run_pigment_cid.py',name,'--frontier',str(n),'--screening-frontier',str(w),'--roots',str(count)],f'{prefix}_N{n}_W{w}.log');break
   except (subprocess.CalledProcessError,ValueError) as error:
    text=(HERE/f'{prefix}_N{n}_W{w}.log').read_text()
    if 'unresolved target subspace' not in str(error) and 'unresolved target subspace' not in text[-3000:]:raise
  else:raise RuntimeError('Native-target coverage unresolved within configured root limits')
 r=json.loads((out/'result.json').read_text());print(json.dumps(dict(N=n,W=w,basis=r['basis_size'],lowest=r['states'][:max(protocol['roots'])],coverage=r['target_coverage'])),flush=True);return r
def gaps(r):return [r['states'][i-1]['excitation_ev'] for i in protocol['roots']]
def increments(rows):return [[b-a for a,b in zip(gaps(x),gaps(y))] for x,y in zip(rows,rows[1:])]
def passed(rows):return len(rows)>=protocol['successive_increments_required']+1 and all(abs(x)<protocol['energy_tolerance_ev'] for row in increments(rows[-(protocol['successive_increments_required']+1):]) for x in row)
def save(data):
 p=HERE/f'{prefix}_progress.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
# A pre-existing baseline worker may own the first checkpoint.
first=baseline_ns[0]
if protocol.get('wait_for_existing_baseline',False):
 while not (folder(first,baseline_w)/'result.json').exists():
  log=HERE/f'{prefix}_N{first}_W{baseline_w}.log'
  if log.exists() and 'Traceback' in log.read_text():raise RuntimeError('Baseline worker failed; inspect preserved log')
  time.sleep(10)
baseline=[calc(n,baseline_w) for n in baseline_ns]
coherent=[baseline[-1]];windows=[baseline_ns[-1]]
for n in range(baseline_ns[-1]+1,protocol['maximum_frontier']+1):
 coherent.append(calc(n,n));windows.append(n);save(dict(phase='coherent',windows=windows,increments_ev=increments(coherent),coherent_passed=passed(coherent)))
 if passed(coherent):break
if not passed(coherent):raise RuntimeError('Coherent convergence threshold not reached within frozen frontier limit')
fixed=[calc(n,fixed_w) for n in windows[-(protocol['successive_increments_required']+1):]]
result=dict(protocol=protocol,host=socket.getfqdn(),windows=windows[-history_count:],coherent_results=[str(folder(n,n)) for n in windows[-history_count:]],fixed_results=[str(folder(n,fixed_w)) for n in windows[-history_count:]],coherent_increments_ev=increments(coherent[-history_count:]),fixed_Hamiltonian_increments_ev=increments(fixed),coherent_converged=passed(coherent),fixed_Hamiltonian_converged=passed(fixed),numerical_audit_pending=True)
save(result);(HERE/f'{prefix}_convergence.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
