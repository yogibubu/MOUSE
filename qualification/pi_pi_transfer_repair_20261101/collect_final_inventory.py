"""Archive executed sources and hash completed remote artifacts; no calculation."""
from pathlib import Path
import sys,json,socket,runpy,hashlib,shutil,subprocess,importlib.metadata,platform
assert socket.getfqdn()=='oracle.sns.it'
HERE=Path('/home/enzo/MOUSE-pi-pi-transfer-repair-20261101')
runpy.run_path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration/source/qualification/remote_bootstrap.py')
import matrix_gfn2,matrix_apoc,matrix_link,numpy,scipy
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources={};snapshot=HERE/'executed_sources/final_public'
for package,module in [('matrix-gfn2',matrix_gfn2),('matrix-apoc',matrix_apoc),('matrix-link',matrix_link)]:
 root=Path(module.__file__).parent
 if package=='matrix-gfn2':files=[p for p in root.glob('*.py') if p.name.startswith('_nosi') or p.name in ['_mixed_jet.py','_einsum_cached.py','nosi_surface.py','nonlocal_response.py','deltascf_analytic_hessian.py','_nosi_surface_response.py']]
 elif package=='matrix-apoc':files=[root/'nonorthogonal_ci.py']
 else:files=[root/'nosi_component.py',root/'scan.py']
 for path in files:
  relative=f'packages/{package}/src/{root.name}/{path.name}';target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);sources[relative]=sha(path)
executed=HERE/'executed_sources/research';executed.mkdir(parents=True,exist_ok=True)
for path in HERE.glob('*.py'):shutil.copyfile(path,executed/path.name)
environment=dict(host=socket.getfqdn(),computer_name=subprocess.check_output(['scutil','--get','ComputerName'],text=True).strip() if sys.platform=='darwin' else socket.gethostname(),platform=platform.platform(),pyscf_preparation_version=importlib.metadata.version('pyscf'),native_extension_sha256={str(module.__file__):sha(Path(module.__file__)) for name,module in sys.modules.items() if name.startswith('matrix_gfn2') and str(getattr(module,'__file__','')).endswith('.so')},python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,threads_per_process=2,maximum_parallel_column_processes=12)
(HERE/'final_source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n');(HERE/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
files={}
for path in sorted(HERE.rglob('*')):
 if path.is_file() and path.name not in ['remote_inventory_final.json']:
  files[str(path.relative_to(HERE))]=dict(sha256=sha(path),bytes=path.stat().st_size)
(HERE/'remote_inventory_final.json').write_text(json.dumps(dict(environment=environment,files=files),indent=2)+'\n');print(len(files),'completed files hashed',len(sources),'public sources copied')
