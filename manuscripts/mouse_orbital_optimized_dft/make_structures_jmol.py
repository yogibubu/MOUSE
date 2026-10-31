"""Render archived anchors with Jmol; no molecular optimization or electronic jobs."""
from pathlib import Path
import argparse, csv, hashlib, json, subprocess
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

P=Path(__file__).resolve().parent
NAMES={'nitroxyl':'Nitroxyl (HNO)','nitrosomethane':'Nitrosomethane','cyanoformaldehyde':'Cyanoformaldehyde','cyclopentadienone':'Cyclopentadienone','cyclopropenone':'Cyclopropenone','Bimane_Syn':'syn-Bimane','Nitropyridine_N-oxide':'Nitropyridine N-oxide','Aza-naphthalene':'Aza-naphthalene'}
def title(name):return NAMES.get(name,name.capitalize() if name.islower() else name)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--jmol-jar',type=Path,required=True);args=ap.parse_args()
 cache=P/'tmp/jmol';cache.mkdir(parents=True,exist_ok=True)
 first=list(dict.fromkeys(r['molecule'] for r in csv.DictReader((P/'data/vertical_energies.csv').open())))
 second=json.loads((P/'data/structure_geometry_sources.json').read_text())
 groups=[[(n,P/'coordinates'/f'{n}_anchor.xyz') for n in first],[(r['molecule'],P/r['path']) for r in second]]
 assert [len(g) for g in groups]==[18,10]
 records=[]
 for group in groups:
  for name,source in group:
   key=name.lower().replace(' ','_');lines=source.read_text().splitlines();n=int(lines[0]);rows=[s.split() for s in lines[2:] if s.strip()];assert len(rows)==n
   atoms=[r[0] for r in rows];q=np.array([[float(v) for v in r[1:4]] for r in rows]);heavy=np.array(atoms)!='H'
   center=q[heavy].mean(0);q=q-center
   _,_,axes=np.linalg.svd(q[heavy],full_matrices=True);rot=axes.T
   if np.linalg.det(rot)<0:rot[:,-1]*=-1
   q=q@rot
   view=cache/f'{key}_view.xyz';view.write_text(f'{n}\nRigid display rotation only; Angstrom\n'+''.join(f'{a} {x:.12f} {y:.12f} {z:.12f}\n' for a,(x,y,z) in zip(atoms,q)))
   png=cache/f'{key}.png';spt=cache/f'{key}.spt'
   spt.write_text(f'''load "{view}"
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
    done=subprocess.run(['java','-jar',str(args.jmol_jar),'-n','-g1000x650','-s',str(spt)],capture_output=True,text=True,check=True)
    (cache/f'{key}.log').write_text(done.stdout+done.stderr)
   assert png.exists(),name
   records.append(dict(molecule=name,geometry=str(source.relative_to(P)),geometry_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),atoms=n,display_rotation=rot.tolist(),display_center_angstrom=center.tolist(),jmol_script=spt.read_text().replace(str(P),'$MANUSCRIPT')))
 width,height=540,686;pdf=canvas.Canvas(str(P/'figures/molecular_structures_jmol.pdf'),pagesize=(width,height));pdf.setTitle('MOUSE OO-DFT: molecular structures rendered with Jmol')
 def heading(y,letter,text,count):
  pdf.setFillColorRGB(.12,.20,.29);pdf.roundRect(5,y-8,20,20,4,fill=1,stroke=0);pdf.setFillColorRGB(1,1,1);pdf.setFont('Helvetica-Bold',11);pdf.drawCentredString(15,y-2,letter)
  pdf.setFillColorRGB(.12,.20,.29);pdf.setFont('Helvetica-Bold',11);pdf.drawString(33,y-1,text);pdf.setFont('Helvetica',8);pdf.setFillColorRGB(.4,.45,.5);pdf.drawRightString(533,y,f'{count} molecules')
 def draw_group(group,cols,top,cellh):
  cellw=width/cols
  for i,(name,_) in enumerate(group):
   row,col=divmod(i,cols);x=col*cellw;y=top-(row+1)*cellh
   png=cache/(name.lower().replace(' ','_')+'.png')
   im=Image.open(png).convert('RGB');a=np.asarray(im);mask=np.any(a<225,axis=2);yy,xx=np.where(mask);pad=18
   im=im.crop((max(0,xx.min()-pad),max(0,yy.min()-pad),min(im.width,xx.max()+pad+1),min(im.height,yy.max()+pad+1)))
   boxw,boxh=cellw-12,cellh-25;scale=min(boxw/im.width,boxh/im.height);iw,ih=im.width*scale,im.height*scale
   pdf.drawImage(ImageReader(im),x+(cellw-iw)/2,y+23+(boxh-ih)/2,width=iw,height=ih)
   pdf.setFillColorRGB(.12,.16,.21);pdf.setFont('Helvetica',7.3)
   label=title(name)
   if pdf.stringWidth(label,'Helvetica',7.3)>cellw-4 and ' ' in label:
    words=label.split();cut=len(words)//2;pdf.drawCentredString(x+cellw/2,y+13,' '.join(words[:cut]));pdf.drawCentredString(x+cellw/2,y+4,' '.join(words[cut:]))
   else:pdf.drawCentredString(x+cellw/2,y+9,label)
 heading(666,'A','Initial vertical-validation cohort',18);draw_group(groups[0],6,645,114)
 pdf.setStrokeColorRGB(.85,.88,.91);pdf.line(8,292,532,292)
 heading(270,'B','Reduced validation extension',10);draw_group(groups[1],5,249,110)
 pdf.setFillColorRGB(.35,.39,.44);pdf.setFont('Helvetica',7);pdf.drawCentredString(270,12,'Jmol 3D ball-and-stick views | C: grey   H: white   N: blue   O: red | panel scales vary')
 pdf.showPage();pdf.save()
 provenance=dict(renderer='Jmol',jar_sha256=hashlib.sha256(args.jmol_jar.read_bytes()).hexdigest(),number_of_molecules=len(records),geometry_changes='Rigid translation and proper rotation only; original coordinates archived unchanged',projection='Orthographic; fixed 18/-12/-5 degree display tilts; panels scaled independently for legibility',groups=[dict(name='Initial vertical cohort',molecules=first),dict(name='Reduced extension',molecules=[r['molecule'] for r in second])],molecules=records)
 (P/'data/molecular_structures_jmol.json').write_text(json.dumps(provenance,indent=2)+'\n')
 print('Rendered',len(records),'molecular structures')
if __name__=='__main__':main()
