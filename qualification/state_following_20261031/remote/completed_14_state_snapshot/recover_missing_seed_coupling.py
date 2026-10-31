from pathlib import Path
import subprocess,sys,shutil,json,socket
assert socket.getfqdn()=='oracle.sns.it'
v=Path('/home/enzo/MOUSE-orbital-performance-20261031')
src=v/'results/Phthalimide/B005/grounds';dst=v/'results/Phthalimide/B024/grounds'
if not dst.exists():
 shutil.copytree(src,dst)
 (dst/'reuse_provenance.json').write_text(json.dumps(dict(source=str(src),reason='identical geometry and model; worker verifies response fingerprints; certificates and checkpoints unchanged'),indent=2)+'\n')
(v/'progress.json').write_text(json.dumps(dict(host=socket.getfqdn(),stage='missing_seed_coupling',state='B024',status='running',labels_read=False),indent=2)+'\n')
for args in [['assigned_pair_worker.py','Phthalimide','B024'],['assigned_pair_audit.py']]:
 with (v/'recover_missing_seed_coupling.log').open('a') as f:subprocess.run([sys.executable,str(v/args[0])]+args[1:],stdout=f,stderr=subprocess.STDOUT,timeout=240,check=True)
sys.path.insert(0,str(v));import assigned_pipeline
a=assigned_pipeline.ledger()
(v/'progress.json').write_text(json.dumps(dict(host=socket.getfqdn(),stage='OO_assignments_complete' if a else 'repair_required',status='controls_completeness_check_required',matched_states=json.loads((v/'ledger.json').read_text())['matched_states'],labels_read=False),indent=2)+'\n')
