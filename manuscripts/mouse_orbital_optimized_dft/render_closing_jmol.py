"""Render three fixed QUEST anchors with Jmol; no electronic calculations."""
from pathlib import Path
import subprocess,json,hashlib,shutil,argparse
import numpy as np
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--jar',type=Path,required=True);a=p.parse_args();root=P/'data/closing_validation';root.mkdir(exist_ok=True);records=[]
for name in ['Formaldehyde','Ethylene','Butadiene']:
 source=P/'coordinates/closing_validation'/(name+'.xyz');lines=source.read_text().splitlines();n=int(lines[0]);atoms=[line.split() for line in lines[2:2+n]];x=np.array([[float(v) for v in row[1:4]] for row in atoms]);center=x.mean(0);_,_,axes=np.linalg.svd(x-center,full_matrices=True);rotation=axes.T
 if np.linalg.det(rotation)<0:rotation[:,-1]*=-1
 view=(x-center)@rotation;cache=P/'tmp/closing_jmol';cache.mkdir(parents=True,exist_ok=True);xyz=cache/(name+'.xyz');xyz.write_text(str(n)+'\nRigid display rotation, Angstrom\n'+''.join(f'{row[0]} {r[0]:.12f} {r[1]:.12f} {r[2]:.12f}\n' for row,r in zip(atoms,view)));png=root/(name+'.png');script=cache/(name+'.spt');script.write_text(f'''load "{xyz}"
background white
set frank off
set antialiasDisplay true
set antialiasImages true
set perspectiveDepth false
select all
spacefill 23%
wireframe 0.09
color atoms CPK
select carbon
color atoms [x535B66]
select hydrogen
color atoms [xF0F0EC]
select all
set ambientPercent 42
set diffusePercent 78
set specularPercent 28
rotate x 18
rotate y -12
rotate z -5
zoom 85
write image 1000 650 PNG "{png}"
exit
''')
 if not png.exists():
  result=subprocess.run(['java','-jar',str(a.jar),'-n','-g1000x650','-s',str(script)],capture_output=True,text=True,check=True);(cache/(name+'.log')).write_text(result.stdout+result.stderr)
 assert png.exists(),name
 records.append(dict(molecule=name,geometry=str(source.relative_to(P)),geometry_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),render=str(png.relative_to(P)),render_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),display_rotation=rotation.tolist(),display_center=center.tolist()))
(root/'jmol_manifest.json').write_text(json.dumps(dict(renderer='Jmol 14.31.10',jar_sha256=hashlib.sha256(a.jar.read_bytes()).hexdigest(),molecules=records),indent=2)+'\n')
