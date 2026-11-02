"""Plot a completed fixed-state functional comparison; no electronic work."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
j=json.loads((root/'functional_comparison.json').read_text());assert len(j['rows'])==18,'Incomplete campaign'
ids=['B001','B002','B003','V010','N001','V001']
labels=['Formaldehyde\n$n\\to\\pi^*$','Ethylene\n$\\pi\\to\\pi^*$','Butadiene\n$\\pi\\to\\pi^*$','Furan\n$\\pi\\to\\pi^*$','Acetaldehyde\n$n\\to\\pi^*$','Acetone\n$n\\to\\pi^*$']
styles=[('PBE0','#64829a','o'),('CAM-B3LYP','#d0803c','s'),('WB97X','#217b65','D')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.2,3.8),layout='constrained')
ax.axvspan(3.5,5.5,color='#f0f2f4',zorder=0);ax.axhline(0,color='#444',lw=.9)
for f,color,marker in styles:
 values=[next(r['error_ev'] for r in j['rows'] if r['id']==i and r['functional']==f) for i in ids]
 ax.plot(range(6),values,marker=marker,color=color,lw=1.3,ms=5,label=f.replace('WB97X',r'$\omega$B97X'))
ax.set_xticks(range(6),labels);ax.set_ylabel('ROKS − reference energy (eV)');ax.set_xlim(-.3,5.3);ax.grid(axis='y',alpha=.16)
ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01))
ax.text(.23,.94,'Selection: four pilot states',transform=ax.transAxes,ha='center',fontsize=8,color='#555')
ax.text(.82,.94,'Two confirmation states',transform=ax.transAxes,ha='center',fontsize=8,color='#555')
fig.savefig(root/'functional_comparison.pdf');fig.savefig(root/'functional_comparison.png',dpi=200)
