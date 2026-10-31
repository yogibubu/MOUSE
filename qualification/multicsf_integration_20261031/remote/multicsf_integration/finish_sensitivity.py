from pathlib import Path
import subprocess,sys,socket,json
assert socket.getfqdn()=='oracle.sns.it'
w=Path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration')
for name,sid in [('Cyclopentadiene','B016'),('Phthalimide','B024')]:subprocess.run([sys.executable,str(w/'export_model.py'),name,sid],check=True)
subprocess.run([sys.executable,str(w/'lambda_sensitivity.py')],check=True)
(w/'sensitivity_complete.json').write_text(json.dumps(dict(host=socket.getfqdn(),complete=True),indent=2)+'\n')
