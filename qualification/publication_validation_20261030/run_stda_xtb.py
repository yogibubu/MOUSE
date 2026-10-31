"""Published dedicated sTDA-xTB on exact archived geometries, remote only.

This stage preserves all spectra/vectors. State matching is a separate mandatory
stage; a completed spectrum is never automatically counted as a matched state.
"""
from pathlib import Path
import socket,os,subprocess,json,time,hashlib,signal,re
assert socket.getfqdn()=='oracle.sns.it'
P=Path(__file__).resolve().parent;BIN=P/'software/bin/xtb4stda';STD=Path('/home/enzo/.local/bin/std2');OUT=P/'external/stda_xtb/attempt_03';OUT.mkdir(parents=True,exist_ok=True)
ENV=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',OMP_STACKSIZE='512M',XTB4STDAHOME='/home/enzo/mouse-pub-20261030-xtbdata')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def command(args,cwd,name,timeout=180):
 stamp=time.perf_counter();log=cwd/(name+'.log');res=cwd/(name+'.resources.txt')
 with log.open('w') as stream:
  process=subprocess.Popen(['/usr/bin/time','-v','-o',str(res)]+list(map(str,args)),cwd=cwd,env=ENV,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=process.wait(timeout=timeout);status='completed' if rc==0 else 'execution_error'
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait();rc=None;status='timeout'
 text=res.read_text() if res.exists() else '';m=re.search(r'Maximum resident set size \(kbytes\):\s+(\d+)',text)
 return dict(command=list(map(str,args)),returncode=rc,status=status,wall_seconds=time.perf_counter()-stamp,maximum_rss_bytes=int(m[1])*1024 if m else None)
manifest=dict(host=socket.getfqdn(),threads=2,geometry_source='same archived vertical anchors as MOUSE',executable_sha256={str(p):sha(p.resolve()) for p in [BIN,STD]},method='dedicated xtb4stda then std2 -xtb -e 10 -vectm 30',state_matching='pending; spectra alone are not matched results',parameter_sha256={f.name:sha(f) for f in sorted((P/'software/xtb4stda-data').glob('.param_stda*.xtb'))})
f=OUT/'protocol.json'
if f.exists():assert json.loads(f.read_text())==manifest
else:write(f,manifest)
for xyz in sorted((P/'inputs/anchors').glob('*_anchor.xyz')):
 name=xyz.stem.removesuffix('_anchor');d=OUT/name;d.mkdir(exist_ok=True);request=dict(geometry_sha256=sha(xyz),software=manifest['executable_sha256']);done=d/'result.json'
 if done.exists():assert json.loads(done.read_text())['request']==request;continue
 target=d/'geometry.xyz'
 if not target.exists():target.write_bytes(xyz.read_bytes())
 stages=[];ground=d/'ground_result.json'
 if ground.exists():g=json.loads(ground.read_text());assert g['request']==request;g=g['stage']
 else:
  g=command([BIN,'geometry.xyz'],d,'ground');write(ground,dict(request=request,stage=g))
 stages.append(g)
 if g['status']=='completed' and (d/'wfn.xtb').exists():
  r=command([STD,'-xtb','-e','10','-vectm','30'],d,'response');stages.append(r);status='completed_unassigned' if r['status']=='completed' else r['status']
 else:status=g['status'] if g['status']!='completed' else 'execution_error'
 result=dict(host=socket.getfqdn(),molecule=name,request=request,status=status,stages=stages,files={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name not in ['result.json']})
 write(done,result);print(name,status,flush=True)
 if not (d/'wfn.xtb').exists():raise RuntimeError('Dedicated ground-state preflight failed; see '+str(d))
