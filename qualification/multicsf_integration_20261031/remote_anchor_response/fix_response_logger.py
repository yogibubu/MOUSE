from pathlib import Path
import json,hashlib,shutil,socket
assert socket.getfqdn()=="oracle.sns.it"
w=Path("/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration")
p=w/"anchor_rank_symmetry/protocol.json"
d=json.loads(p.read_text())
assert d["source_sha256"]=='9eccfcc0a37bb6814733a7cfc87388a72f402730a14e842c40a6a04762dce182'
assert not list((w/"anchor_rank_symmetry").glob("*/*/result.json"))
shutil.copy2(p,p.with_name("protocol_before_logger_fix.json"))
f=w/"anchor_rank_symmetry/Phthalimide/0.98/failure.json"
if f.exists():shutil.copy2(f,f.with_name("failure_before_logger_fix.json"))
log=w/"anchor_rank_symmetry.log"
shutil.copy2(log,w/"anchor_rank_symmetry_logger_setup.log")
d["source_sha256"]=hashlib.sha256((w/"anchor_rank_symmetry.py").read_bytes()).hexdigest()
p.write_text(json.dumps(d,indent=2)+"\n")
