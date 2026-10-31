"""Frozen numerical recoveries; retain successful calculations and every prior attempt."""
from pathlib import Path
import socket,json,re,shutil,hashlib
assert socket.getfqdn()=='oracle.sns.it'
import run_dftbplus as base
P=Path(__file__).resolve().parent
jobs=[('nitroxyl','td'),('cyclopentadienone','delta'),('uracil','delta')]
plan={'host':socket.getfqdn(),'jobs':jobs,'td':'reduce ARPACK roots to floor((ntrans-1)/2); no Hamiltonian change; output SPX for identity audit','delta':'DIIS mixer, 300 SCC iterations; same Hamiltonian and spin constants; rerun failed calculation only','source_sha256':base.sha(Path(__file__))}
root=P/'external/dftbplus24/attempt_04';root.mkdir(exist_ok=True)
f=root/'recovery_protocol.json'
if f.exists():assert json.loads(f.read_text())==json.loads(json.dumps(plan))
else:base.write(f,plan)
for name,method in jobs:
 old=P/'external/dftbplus24/attempt_03'/name/method
 prior=json.loads((old/'result.json').read_text());assert prior['status']!='completed_unassigned'
 d=root/name/method;d.mkdir(parents=True,exist_ok=True)
 if (d/'result.json').exists():continue
 shutil.copy2(old/'geometry.xyz',d/'geometry.xyz');text=(old/'dftb_in.hsd').read_text()
 if method=='td':
  atoms=[x.split()[0] for x in (d/'geometry.xyz').read_text().splitlines()[2:]]
  no=sum(base.VAL[x] for x in atoms)//2;nv=sum(base.NAO[x] for x in atoms)-no
  text=re.sub(r'NrOfExcitations = \d+',f'NrOfExcitations = {max(1,min(30,(no*nv-1)//2))}',text).replace('WriteTransitions = Yes','WriteTransitions = Yes\n WriteSPTransitions = Yes')
 else:text=text.replace('Mixer = Broyden { MixingParameter = 0.1 }','Mixer = DIIS {\n InitMixingParameter = 0.1\n Generations = 6\n }')
 (d/'dftb_in.hsd').write_text(text)
 r=base.run([base.B],d);r.update(molecule=name,method=method,host=socket.getfqdn(),previous_attempt=str(old),input_sha256=base.sha(d/'dftb_in.hsd'),plan_sha256=base.sha(f))
 if (d/'detailed.out').exists():
  m=re.search(r'S0\s*->\s*S1:\s+([-+0-9.Ee]+)\s+H\s+([-+0-9.Ee]+)\s+eV',(d/'detailed.out').read_text())
  if m:r['lowest_singlet_gap_ev']=float(m[2])
 base.write(d/'result.json',r);print(name,method,r,flush=True)
