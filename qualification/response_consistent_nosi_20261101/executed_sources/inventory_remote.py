"""Read-only provenance inventory; no electronic calculation."""
from pathlib import Path
import hashlib,json,socket,subprocess
ROOT=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-response-consistent-nosi-20261101')
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
assert subprocess.check_output(['scutil','--get','ComputerName'],text=True).strip()=='Mac Studio di Vincenzo'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
operators={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('operators.npz'))}
source=ROOT.parents[1]/'source/packages'
files={str(p.relative_to(source)):sha(p) for p in sorted(source.glob('*/src/*/*.py')) if 'nosi' in p.name or p.name in ['qmqm_component.py','scan.py','optimizer.py','composite_roks_component.py']}
(ROOT/'remote_inventory.json').write_text(json.dumps(dict(host=socket.getfqdn(),computer_name='Mac Studio di Vincenzo',large_remote_operators=operators,public_source_hashes=files),indent=2)+'\n')
