"""Finite dependent archive step for the already running cost campaign."""
from pathlib import Path
import socket,time,json,subprocess,sys,datetime
assert socket.getfqdn()=='oracle.sns.it'
r=Path(__file__).resolve().parent;deadline=time.monotonic()+3600
while True:
 progress=json.loads((r/'progress.json').read_text())
 rows=list((r/'cost').rglob('result.json'))
 if len(rows)==72 and progress.get('status')=='attempt_pass_complete' and (r/'hessian_acceleration_benchmark_complete.json').exists():break
 if time.monotonic()>deadline:raise TimeoutError('Finite campaign did not close within archive wait budget')
 time.sleep(5)
subprocess.run([sys.executable,str(r/'collect_final.py')],check=True)
d=json.loads((r/'final_results.json').read_text());assert all(d['operational_closure'].values())
assert d['geometries']['Cyclopentadiene']['minimum_mode_check']['status']=='pass'
result=subprocess.check_output([sys.executable,str(r/'archive.py')],text=True)
(r/'final_archive.json').write_text(result);print(result,flush=True)
