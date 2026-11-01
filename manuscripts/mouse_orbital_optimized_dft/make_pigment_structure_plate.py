"""Actual Jmol pigment renders, labelled as preparation rather than validation."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
P=Path(__file__).resolve().parent;records=json.loads((P/'data/pigment_preparation/jmol_manifest.json').read_text())['molecules'];c=canvas.Canvas(str(P/'figures/pigment_test_structures_jmol.pdf'),pagesize=(540,620));c.setTitle('Pigment stress-test starting structures, not validated excited states');c.setFont('Helvetica-Bold',14);c.drawString(20,594,'Pigment test structures: preparation')
for record,box in zip(records,[(18,344,504,208),(104,66,332,244)]):
 source=P/record['render'];assert hashlib.sha256(source.read_bytes()).hexdigest()==record['render_sha256'];im=Image.open(source).convert('RGB');a=np.asarray(im);yy,xx=np.where(np.any(a<225,axis=2));im=im.crop((max(0,xx.min()-15),max(0,yy.min()-15),min(im.width,xx.max()+16),min(im.height,yy.max()+16)));x,y,w,h=box;scale=min(w/im.width,h/im.height);ww,hh=im.width*scale,im.height*scale;c.drawImage(ImageReader(im),x+(w-ww)/2,y+(h-hh)/2,width=ww,height=hh);c.setFillColorRGB(.12,.22,.29);c.setFont('Helvetica-Bold',11);c.drawCentredString(270,y-14,record['molecule'].replace('_',' '))
c.setFont('Helvetica',8);c.drawCentredString(270,20,'Jmol 3D | Source geometries | Native ground-state preflight only | No OO excitation validation claimed');c.showPage();c.save()
