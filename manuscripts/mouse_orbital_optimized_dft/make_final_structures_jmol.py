"""Actual Jmol renders of the six new frozen molecular anchors, no optimization."""
from pathlib import Path
import json,hashlib,subprocess,argparse
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
P=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--jmol-jar',type=Path,required=True);a=ap.parse_args();cache=P/'tmp/jmol_final';cache.mkdir(parents=True,exist_ok=True)
 rows=json.loads((P/'data/structure_geometry_sources_final.json').read_text());records=[]
 names={'Criegee_Intermediate':'Criegee intermediate','Cyclobutadiene':'Cyclobutadiene','Benzoxadiazole':'Benzoxadiazole','Hexatriene':'Hexatriene','Ammonia':'Ammonia','Furofuran':'Furofuran'}
 for row in rows:
  source=P/row['path'];assert hashlib.sha256(source.read_bytes()).hexdigest()==row['geometry_sha256'];lines=source.read_text().splitlines();nat=int(lines[0]);atom=[s.split() for s in lines[2:2+nat]];elements=[s[0] for s in atom];x=np.array([[float(v) for v in s[1:4]] for s in atom]);heavy=np.array(elements)!='H';center=x[heavy].mean(0);x-=center;_,_,vt=np.linalg.svd(x[heavy],full_matrices=True);rotation=vt.T
  if np.linalg.det(rotation)<0:rotation[:,-1]*=-1
  x=x@rotation;name=row['molecule'];view=cache/f'{name}.xyz';view.write_text(f'{nat}\nRigid display rotation only; Angstrom\n'+''.join(f'{el} {xx:.12f} {yy:.12f} {zz:.12f}\n' for el,(xx,yy,zz) in zip(elements,x)))
  png=cache/f'{name}.png';script=cache/f'{name}.spt';script.write_text(f'''load "{view}"
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
''')
  if not png.exists():
   run=subprocess.run(['java','-jar',str(a.jmol_jar),'-n','-g1000x650','-s',str(script)],capture_output=True,text=True,check=True);(cache/f'{name}.log').write_text(run.stdout+run.stderr)
  records.append(dict(row,display_center=center.tolist(),display_rotation=rotation.tolist(),jmol_script=script.read_text().replace(str(P),'$MANUSCRIPT')))
 pdf=canvas.Canvas(str(P/'figures/independent_structures_jmol.pdf'),pagesize=(540,390));pdf.setTitle('Frozen validation and state-character challenge molecules')
 pdf.setFillColorRGB(.12,.20,.29);pdf.setFont('Helvetica-Bold',13);pdf.drawString(12,369,'New validation and state-character challenges')
 for i,row in enumerate(rows):
  rr,cc=divmod(i,3);left=cc*180;bottom=344-(rr+1)*162;im=Image.open(cache/f"{row['molecule']}.png").convert('RGB');pixels=np.asarray(im);yy,xx=np.where(np.any(pixels<225,axis=2));im=im.crop((max(0,xx.min()-18),max(0,yy.min()-18),min(im.width,xx.max()+19),min(im.height,yy.max()+19)));scale=min(165/im.width,132/im.height);w,h=im.width*scale,im.height*scale;pdf.drawImage(ImageReader(im),left+(180-w)/2,bottom+25+(132-h)/2,width=w,height=h);pdf.setFont('Helvetica',9);pdf.setFillColorRGB(.12,.16,.21);pdf.drawCentredString(left+90,bottom+9,names[row['molecule']])
 pdf.setFont('Helvetica',7);pdf.setFillColorRGB(.35,.39,.44);pdf.drawCentredString(270,9,'Jmol ball-and-stick | Archived frozen anchors | Rigid display rotations only | Panel scales vary');pdf.showPage();pdf.save()
 (P/'data/independent_structures_jmol.json').write_text(json.dumps(dict(renderer='Jmol 14.31.10',jar_sha256=hashlib.sha256(a.jmol_jar.read_bytes()).hexdigest(),molecules=records,scope='five primary holdout molecules and one subsequent fresh molecular challenge; presence does not certify convergence or accuracy'),indent=2)+'\n')
 print('Rendered six archived molecular structures with Jmol')
if __name__=='__main__':main()
