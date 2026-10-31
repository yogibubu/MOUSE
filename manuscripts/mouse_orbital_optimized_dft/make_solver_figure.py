"""Plot archived Oracle measurements only; no electronic calculation."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
d=json.loads((P/'data/solver_acceleration.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
fig,(a,b)=plt.subplots(1,2,figsize=(8.5,3.5),layout='constrained')
colors=['#D98745','#168A8A']
bars=a.bar(['Scalar response','Batched response'],[d['old_seconds'],d['new_seconds']],color=colors,width=.55)
a.bar_label(bars,fmt='%.3f s',padding=4);a.set_ylim(0,5)
a.set_ylabel('Wall time (s)');a.set_title('(a) Identical analytic Jacobian',loc='left',fontsize=11)
a.text(.97,.88,f"{d['kernel_speedup']:.1f} times faster",transform=a.transAxes,ha='right',color=colors[1],fontweight='bold')
b.axis('off');b.set_title('(b) Numerical equivalence and scope',loc='left',fontsize=11)
b.text(.02,.82,'44 basis functions\n510 orbital rotations\nTwo Oracle threads',transform=b.transAxes,linespacing=1.7)
b.text(.02,.43,r'Maximum matrix difference'+'\n'+r'$8.88\times10^{-16}$',transform=b.transAxes,fontsize=12,color='#168A8A',linespacing=1.8)
b.text(.02,.10,'One Jacobian at the same retained checkpoint.\nReference preparation and derivatives are excluded.\nThis is not an end-to-end comparison with TDA.',transform=b.transAxes,fontsize=8.5,color='#56636A',linespacing=1.5)
for ext in ['pdf','svg','png']:
 fig.savefig(P/'figures'/('solver_acceleration.'+ext),dpi=220)
plt.close(fig)
provenance=P/'figures/FIGURE_PROVENANCE.json'
meta=json.loads(provenance.read_text())
meta['figures']['solver_acceleration']='Measured identical-checkpoint 510-rotation orbital Jacobian; excludes preparation and nuclear derivatives; no end-to-end TDA comparison.'
meta['solver_acceleration_source']=dict(path='data/solver_acceleration.json',sha256=hashlib.sha256((P/'data/solver_acceleration.json').read_bytes()).hexdigest(),script='make_solver_figure.py',host='oracle.sns.it')
provenance.write_text(json.dumps(meta,indent=2)+'\n')
