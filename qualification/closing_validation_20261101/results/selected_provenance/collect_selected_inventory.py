from pathlib import Path
import socket,json,hashlib,shutil,subprocess
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
base=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');root=base/'oracle/MOUSE-closing-validation-20261101';out=root/'selected_provenance';out.mkdir(exist_ok=True)
files=[]
for folder in [root/'disentangled_bath_eta_0.600000',root/'disentangled_bath',root/'downfold_reference']:
 for p in sorted(folder.rglob('*')):
  if p.is_file() and p.suffix in ['.json','.npz']:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
   files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=h.hexdigest()))
for p in (base/'source/packages/matrix-gfn2/src/matrix_gfn2').glob('_nosi*.py'):shutil.copy2(p,out/p.name)
for name in ['nosi_surface.py','_mixed_jet.py','_einsum_cached.py']:shutil.copy2(base/'source/packages/matrix-gfn2/src/matrix_gfn2'/name,out/name)
for p in root.glob('*.py'):shutil.copy2(p,out/p.name)
report=dict(host=socket.getfqdn(),computer_name=subprocess.check_output(['scutil','--get','ComputerName'],text=True).strip(),files=files,scope='Selected empirical-bath candidate, its complete parameter-free parent and reusable Coulomb-vertex caches. Large arrays remain remote; corrected ethylene portable bundle copied to repository.')
(out/'remote_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
