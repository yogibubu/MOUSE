"""Rebuild final error panels, twenty-state table and eighteen-molecule Jmol plate."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
P=Path(__file__).resolve().parent
rows=json.loads((P/'data/screened_cid/predictions.json').read_text());summary=json.loads((P/'data/screened_cid/summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(8.4,5.8),gridspec_kw={'width_ratios':[1,1.08]})
for ax,family,color,title in zip(axes,['pi_pi','n_pi'],['#008b8b','#285582'],[r'(a) All seven $\pi\to\pi^*$ states',r'(b) Thirteen $n\to\pi^*$ controls']):
 group=[r for r in rows if r['family']==family];new=np.array([r['error_ev'] for r in group]);old=np.array([r['previous_oo_ev']-r['reference_ev'] for r in group]);y=np.arange(len(group));ax.axvspan(-.5,.5,color='#edf4f6');ax.axvline(0,color='#8b9aa8',lw=.8)
 for i in range(len(group)):ax.plot([old[i],new[i]],[i,i],color='#acb8c2',lw=1.2,zorder=2)
 ax.scatter(old,y,s=29,facecolor='white',edgecolor='#8f9da8',lw=1,label='Previous compact space',zorder=3)
 ax.scatter(new,y,s=32,c=color,edgecolor='white',lw=.5,label='Screened symmetry augmentation',zorder=4)
 labels=[r['id']+' '+r['molecule'].replace('_Intermediate','').replace('_',' ') for r in group]
 ax.set(yticks=y,yticklabels=labels,xlabel='Signed excitation error / eV',xlim=(-.82,.82) if family=='pi_pi' else (-.55,.55));ax.set_ylim(len(group)-.45,-.65);ax.tick_params(axis='y',length=0,labelsize=8);ax.grid(axis='x',alpha=.14);ax.set_title(title,loc='left',fontsize=10,fontweight='bold',pad=12)
 score=summary[family];ax.text(.02,-.17,f"MAE {score['MAE_ev']:.3f} eV; max {score['max_error_ev']:.3f} eV",transform=ax.transAxes,fontsize=8,color=color)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.025),frameon=False,ncol=2,fontsize=8)
fig.text(.5,.006,'Fixed 50/50 reference weights; lambda = 0.9; no energy offset. All twenty states are exposed development.',ha='center',fontsize=7,color='#455668')
fig.tight_layout(rect=(0,.12,1,1),w_pad=1.7)
for ext in ['pdf','png']:fig.savefig(P/'figures'/f'screened_cid_validation.{ext}',dpi=230,bbox_inches='tight')
lines=[r'\begin{tabular}{llrrr}\toprule',r'ID & Molecule & Reference & Screened OO-CID & TDA \\ \midrule']
for r in rows:lines.append(f"{r['id']} & {r['molecule'].replace('_',' ')} & {r['reference_ev']:.3f} & {r['prediction_ev']:.3f} & {r['TDA_ev']:.3f}"+r' \\')
lines.append(r'\bottomrule\end{tabular}');(P/'tables/screened_cid_validation.tex').write_text('\n'.join(lines)+'\n')
manifest=json.loads((P/'data/screened_cid/jmol_manifest.json').read_text());pdf=canvas.Canvas(str(P/'figures/screened_cid_structures_jmol.pdf'),pagesize=(540,710));pdf.setTitle('Eighteen molecular anchors in the final MOUSE OO-CID test')
pdf.setFillColorRGB(.10,.20,.28);pdf.setFont('Helvetica-Bold',13);pdf.drawString(14,688,'Twenty transitions in eighteen molecules')
for index,record in enumerate(manifest['molecules']):
 source=P/record['render'];assert hashlib.sha256(source.read_bytes()).hexdigest()==record['render_sha256'];im=Image.open(source).convert('RGB');pixels=np.asarray(im);yy,xx=np.where(np.any(pixels<225,axis=2));im=im.crop((max(0,xx.min()-18),max(0,yy.min()-18),min(im.width,xx.max()+19),min(im.height,yy.max()+19)))
 row,col=divmod(index,3);left=col*180;bottom=671-(row+1)*106;scale=min(162/im.width,80/im.height);width,height=im.width*scale,im.height*scale;pdf.drawImage(ImageReader(im),left+(180-width)/2,bottom+20+(80-height)/2,width=width,height=height);pdf.setFillColorRGB(.12,.20,.28);pdf.setFont('Helvetica',8);pdf.drawCentredString(left+90,bottom+7,record['molecule'].replace('_Intermediate',' intermediate').replace('_',' '))
pdf.setFont('Helvetica',7);pdf.setFillColorRGB(.35,.42,.47);pdf.drawCentredString(270,15,'Jmol 3D ball-and-stick renders | Frozen molecular anchors | Display scales vary');pdf.showPage();pdf.save()
