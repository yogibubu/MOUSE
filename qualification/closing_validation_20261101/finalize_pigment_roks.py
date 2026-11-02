"""Adopt certified, distinct ROKS roots; preserve every completed calculation."""
from pathlib import Path
import argparse,hashlib,json,shutil,socket,time,os
import numpy as np
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
p=argparse.ArgumentParser();p.add_argument('name',nargs='?',default='Chlorophyllide_a');p.add_argument('--wait',action='store_true');args=p.parse_args();HERE=Path(__file__).resolve().parent;model=HERE/'pigment_models'/args.name;official=HERE/'pigment_roks'/args.name;model_hash=hashlib.sha256((model/'manifest.json').read_bytes()).hexdigest();settings=json.loads((model/'manifest.json').read_text());count=1 if args.name=='Beta_carotene' else 4
with np.load(model/'state.npz') as z:s=z['overlap'].copy();no=int(z['electron_count'])//2
ia=list(range(1,no))+[0];ib=list(range(1,no))+[no]
def overlap(a,b):
 return float(np.linalg.det(a[:,ia].T@s@b[:,ia])*np.linalg.det(a[:,ib].T@s@b[:,ib])+np.linalg.det(a[:,ia].T@s@b[:,ib])*np.linalg.det(a[:,ib].T@s@b[:,ia]))
def read(folder,index):
 path=folder/'certificate.json'
 if not path.exists():return None
 result=json.loads(path.read_text());r=result['report']
 if (result['ordinal']!=index or result['model_sha256']!=model_hash or not r['converged'] or r['residual']>=1e-10 or abs(r['ground_overlap'])>=1e-10 or r['fidelity']<.7 or result['spin_squared']!=0. or any(result[k] for k in ['DFT_imports','new_SCC_calls','TDA_solves'])):return None
 with np.load(folder/'checkpoint.npz') as z:c=z['coefficients'].copy();assert str(z['model_sha256'])==model_hash
 assert abs(overlap(c,c)-1.)<1e-8
 return dict(result=result,coefficients=c,folder=folder)
def write(path,data):
 tmp=path.with_suffix(f'.tmp.{os.getpid()}.json');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
rejected={};start=time.monotonic()
while True:
 accepted={}
 # Preserve the earliest completed official root if a later target collapses onto it.
 existing=[(official/f'state_{i}',i) for i in range(1,count+1) if (official/f'state_{i}/certificate.json').exists()];existing.sort(key=lambda item:(item[0]/'certificate.json').stat().st_mtime)
 for folder,i in existing:
  candidate=read(folder,i)
  if candidate is None:continue
  collisions=[(j,overlap(candidate['coefficients'],other['coefficients'])) for j,other in accepted.items()]
  if any(abs(value)>1-1e-7 for j,value in collisions):raise RuntimeError(f'Official roots are duplicated: {i}, {collisions}')
  accepted[i]=candidate
 for i in range(1,count+1):
  if i in accepted:continue
  folders=sorted((HERE/'pigment_roks_variants').glob(f'*/{args.name}/state_{i}'),key=lambda x:(x/'certificate.json').stat().st_mtime if (x/'certificate.json').exists() else float('inf'))
  for folder in folders:
   candidate=read(folder,i)
   if candidate is None:continue
   collisions=[dict(reference=j,overlap=overlap(candidate['coefficients'],other['coefficients'])) for j,other in accepted.items()]
   if any(abs(item['overlap'])>1-1e-7 for item in collisions):
    rejected[str(folder)]=dict(reason='duplicate spin-pure CSF',overlaps=collisions);continue
   dst=official/f'state_{i}';dst.mkdir(parents=True,exist_ok=True);backup=HERE/f'pigment_stagnation_repair_20261102/before_final_adoption/state_{i}';backup.mkdir(parents=True,exist_ok=True)
   for filename in ['checkpoint.npz','progress.json','certificate.json']:
    if (dst/filename).exists() and not (backup/filename).exists():shutil.copy2(dst/filename,backup/filename)
   shutil.copy2(folder/'checkpoint.npz',dst/'checkpoint.npz');progress=json.loads((folder/'progress.json').read_text());write(dst/'progress.json',dict(molecule=args.name,state=i,row=progress.get('row',progress),adopted_from=str(folder)));candidate['result']['adopted_from']=str(folder);write(dst/'certificate.json',candidate['result']);candidate['folder']=dst;accepted[i]=candidate;print('ADOPTED',i,candidate['result']['excitation_ev'],candidate['result']['report']['residual'],flush=True);break
 if len(accepted)==count:
  ordered=[accepted[i] for i in range(1,count+1)];metric=np.array([[overlap(a['coefficients'],b['coefficients']) for b in ordered] for a in ordered]);metric=.5*(metric+metric.T);eigenvalues=np.linalg.eigvalsh(metric);assert eigenvalues[0]>1e-8,eigenvalues
  ground=[item['result']['ground_energy_hartree'] for item in ordered];assert max(ground)-min(ground)<1e-10
  audit=dict(host=socket.getfqdn(),complete=True,model_sha256=model_hash,residual_tolerance=1e-10,states=[dict(ordinal=i,energy_ev=accepted[i]['result']['excitation_ev'],residual=accepted[i]['result']['report']['residual'],fidelity=accepted[i]['result']['report']['fidelity'],source=str(accepted[i]['folder'])) for i in range(1,count+1)],CSF_metric=metric.tolist(),CSF_metric_eigenvalues=eigenvalues.tolist(),rejected_duplicate_candidates=rejected,audit_wall_seconds=time.monotonic()-start)
  destination=HERE/'pigment_stagnation_repair_20261102';destination.mkdir(exist_ok=True);write(destination/'stationarity_audit.json',audit)
  write(official/'result.json',dict(molecule=args.name,host=socket.getfqdn(),states=[item['result'] for item in ordered],weights=settings['weights'],lambda_exchange=settings['lambda_exchange'],offset_ev=settings['energy_offset_ev'],bath_efficiency_eta=settings['bath_efficiency_eta'],model_sha256=model_hash,scope='Distinct spin-pure ground-orthogonal stationary ROKS roots before screened CID; not a final correlated spectrum or a pigment nuclear-Hessian qualification',stationarity_audit=str(destination/'stationarity_audit.json')))
  print(json.dumps(audit,indent=2),flush=True);break
 if not args.wait:
  print(json.dumps(dict(complete=False,certified=sorted(accepted),missing=[i for i in range(1,count+1) if i not in accepted],rejected_duplicate_candidates=rejected),indent=2));break
 time.sleep(20)
