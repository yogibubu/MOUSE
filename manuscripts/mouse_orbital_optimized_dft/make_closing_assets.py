"""Rebuild closing-validation figures/tables solely from bundled evidence."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
P=Path(__file__).resolve().parent;D=P/'data/closing_validation'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
summary=json.loads((D/'prospective_summary.json').read_text());rows=summary['rows'];fig,axes=plt.subplots(1,2,figsize=(9.1,4.0),gridspec_kw={'width_ratios':[1,1.1]});y=np.arange(len(rows));ax=axes[0];ax.axvspan(-.5,.5,color='#edf4f6');ax.axvline(0,color='#91a2ae',lw=.8)
for i,r in enumerate(rows):
 ax.plot([r['error_ev'],r['TDA_error_ev']],[i,i],c='#aeb9c4',lw=1.2)
ax.scatter([r['TDA_error_ev'] for r in rows],y,s=35,facecolor='white',edgecolor='#8796a5',label='Matched native-MO TDA');ax.scatter([r['error_ev'] for r in rows],y,s=35,c='#007d85',label='Frozen OO-CID');ax.set(yticks=y,yticklabels=[r['molecule'] for r in rows],xlabel='Signed vertical error / eV',ylim=(2.5,-.5),xlim=(-.6,3.8));ax.set_title('(a) Three prospective valence states',loc='left',fontweight='bold',fontsize=10);ax.grid(axis='x',alpha=.14);ax.legend(loc='lower right',fontsize=7,frameon=False)
anchors=json.loads((D/'anchor_summary.json').read_text());ax=axes[1];ax.axhline(0,c='#91a2ae',lw=.8)
for name,col,marker in [('Criegee_Intermediate','#007d85','o'),('Formaldehyde','#315d91','s')]:
 for direction,style in [('in_plane_stretch','-'),('out_of_plane','--')]:
  group=sorted([r for r in anchors['frozen_membership'] if r['molecule']==name and r['direction']==direction],key=lambda r:r['step_bohr']);ax.plot([r['step_bohr'] for r in group],[r['anchor_shift_ev'] for r in group],marker=marker,color=col,ls=style,label=(name.replace('_Intermediate','')+' '+('stretch' if style=='-' else 'out of plane')))
ax.set(xlabel='Anchor displacement / bohr',ylabel='Gap change on re-preparation / eV',xticks=[.01,.03],ylim=(-.052,.06));ax.set_title('(b) Same transported CSF membership',loc='left',fontweight='bold',fontsize=10);ax.legend(fontsize=7,frameon=False);ax.grid(alpha=.13)
fig.text(.5,.01,'Fixed 50/50 weights, lambda = 0.9, zero offset. No model refit; state-seed amendment disclosed.',ha='center',fontsize=7,color='#455668');fig.tight_layout(rect=(0,.06,1,1),w_pad=2)
for ext in ['pdf','png']:fig.savefig(P/'figures'/('closing_validation.'+ext),dpi=230,bbox_inches='tight')
lines=[r'\begin{tabular}{lrrrr}\toprule',r'Molecule & Reference & OO-CID & TDA & OO error\\\midrule']
for r in rows:lines.append(f"{r['molecule']} & {r['reference_ev']:.3f} & {r['prediction_ev']:.3f} & {r['TDA_ev']:.3f} & {r['error_ev']:+.3f}"+r'\\')
lines.append(r'\bottomrule\end{tabular}');(P/'tables/closing_vertical.tex').write_text('\n'.join(lines)+'\n')
physical=json.loads((D/'physical_comparison.json').read_text());lines=[r'\begin{tabular}{llrrr}\toprule',r'Molecule / coordinate & Unit & Model & Reference & Difference\\\midrule']
for r in physical['geometries']:lines.append(f"{r['molecule']} / {r['metric']} & {r['unit']} & {r['prediction']:.3f} & {r['reference']:.3f} & {r['error']:+.3f}"+r'\\')
lines.append(r'\bottomrule\end{tabular}');(P/'tables/closing_geometry.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{tabular}{llrrr}\toprule',r'Molecule / mode & Index & Harmonic & Fundamental & Difference\\\midrule']
for r in physical['frequencies']:lines.append(f"{r['molecule']} / {r['mode']} & {r['mode_index']} & {r['prediction_harmonic_cm']:.1f} & {r['reference_fundamental_cm']:.1f} & {r['difference_cm']:+.1f}"+r'\\')
lines.append(r'\bottomrule\end{tabular}');(P/'tables/closing_frequencies.tex').write_text('\n'.join(lines)+'\n')
# Include all 21 molecular anchors, preserving every original Jmol pixel source.
manifest=json.loads((P/'data/screened_cid/jmol_manifest.json').read_text());extra=json.loads((D/'jmol_manifest.json').read_text());records=manifest['molecules']+extra['molecules'];pdf=canvas.Canvas(str(P/'figures/screened_cid_structures_jmol.pdf'),pagesize=(540,710));pdf.setTitle('Twenty-one molecular anchors: development and prospective extension');pdf.setFillColorRGB(.10,.20,.28);pdf.setFont('Helvetica-Bold',13);pdf.drawString(14,688,'Twenty-three transitions in twenty-one molecules')
for index,record in enumerate(records):
 source=P/record['render'];assert hashlib.sha256(source.read_bytes()).hexdigest()==record['render_sha256'];im=Image.open(source).convert('RGB');pixels=np.asarray(im);yy,xx=np.where(np.any(pixels<225,axis=2));im=im.crop((max(0,xx.min()-18),max(0,yy.min()-18),min(im.width,xx.max()+19),min(im.height,yy.max()+19)));row,col=divmod(index,3);left=col*180;bottom=671-(row+1)*91;scale=min(162/im.width,66/im.height);w,h=im.width*scale,im.height*scale;pdf.drawImage(ImageReader(im),left+(180-w)/2,bottom+20+(66-h)/2,width=w,height=h);pdf.setFillColorRGB(.12,.20,.28);pdf.setFont('Helvetica',8);label=record['molecule'].replace('_Intermediate',' intermediate').replace('_',' ');label+= ' [prospective]' if index>=18 else '';pdf.drawCentredString(left+90,bottom+7,label)
pdf.setFont('Helvetica',7);pdf.setFillColorRGB(.35,.42,.47);pdf.drawCentredString(270,15,'Jmol 3D renders | Fixed geometries | Display scales vary | Last row: prospective extension');pdf.showPage();pdf.save()
