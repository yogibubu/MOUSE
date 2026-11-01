"""Read-only source/runtime/model inventory; no electronic calculations."""
from pathlib import Path
import socket,sys,json,hashlib,platform,subprocess
import numpy,scipy,pyscf
ROOT=Path('/home/enzo/MOUSE-frontier-cisd-20261101');NATIVE=Path('/home/enzo/MOUSE-native-reference-20261101');CAM=Path('/home/enzo/MOUSE-cam-response-20261101')
def record(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
protocol=json.loads((ROOT/'protocol.json').read_text());files=[]
for cohort,name in protocol['cases']+[['prospective_cisd',r['molecule']] for r in protocol['prospective_cases']]:
 for root,patterns in [(NATIVE/cohort/name,['coherent_common/model.npz','coherent_common/ground/stationary.npz','coherent_common/reference_certificate.json','coherent_states/*/final/final.npz']), (CAM/cohort/name,['CAM_full_reference/reference.npz','F/model.npz','native_response.npz','embedded_ground.npz','range_eri.npz'])]:
  for pattern in patterns:
   for p in root.glob(pattern):files.append(record(p))
sources=[]
for root in [ROOT,Path('/home/enzo/MOUSE-nosi-20261101'),CAM]:
 for p in root.glob('*.py'):sources.append(record(p))
report=dict(host=socket.getfqdn(),platform=platform.platform(),python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,pyscf=pyscf.__version__,sources=sources,inputs=files)
(ROOT/'runtime_and_input_identity.json').write_text(json.dumps(report,indent=2)+'\n')
