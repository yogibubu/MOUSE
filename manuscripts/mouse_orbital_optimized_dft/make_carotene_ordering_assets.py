"""Plot archived scalar CI results only; no electronic work is performed."""
from pathlib import Path
import json, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
Q=P.parents[1]/'qualification/closing_validation_20261101/results/pigments'
D=P/'data/pigments';D.mkdir(exist_ok=True)
cases=[('N=2, W=5','pigment_cid/Beta_carotene',2,5),('N=4, W=5','pigment_cid_expansion/Beta_carotene/frontier_4',4,5),('N=5, W=5','pigment_cid_expansion/Beta_carotene/frontier_5',5,5),('N=5, W=6','pigment_cid_expansion/Beta_carotene/screening_6/frontier_5',5,6),('N=6, W=6','pigment_cid_expansion/Beta_carotene/screening_6/frontier_6',6,6)]
rows=[]
for label,folder,n,w in cases:
 source=Q/folder; dest=D/f'n{n}_w{w}';dest.mkdir(exist_ok=True)
 for filename in ['result.json','analysis.json']:
  if (source/filename).exists():shutil.copyfile(source/filename,dest/filename)
 report=json.loads((source/'result.json').read_text())
 states=report['states'] if 'states' in report else report['spectrum']
 rows.append(dict(label=label,N=n,W=w,basis_size=report['basis_size'],states=states))
shutil.copyfile(Q/'pigment_spectral_references.json',D/'spectral_references.json')
(D/'ordering_summary.json').write_text(json.dumps(dict(parameters_retuned=False,weights=[.5,.5],lambda_exchange=.9,eta=.6,geometry_unchanged=True,scope='Exposed vertical development test; pigments excluded from 26-state accuracy ledger',rows=rows),indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(10.5,4.7));colors=['#275f9b','#d47722','#647975'];markers=['o','s','^']
for j in range(3):
 for i,row in enumerate(rows):
  state=row['states'][j]
  axs[0].scatter(i,state['excitation_ev'],color=colors[j],marker=markers[j],s=48,label=f'Energy root {j+1}' if i==0 else None,zorder=3)
  axs[1].bar(i+(j-1)*.23,state['oscillator_strength'],width=.21,color=colors[j],label=f'Root {j+1}' if i==0 else None)
for ax in axs:
 ax.set_xticks(range(len(rows)),[r['label'].replace(', ','\n')+f"\n{r['basis_size']} CSFs" for r in rows],fontsize=8);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axs[0].set_ylabel('Vertical excitation / eV');axs[0].set_ylim(2.25,4.5);axs[0].set_title('(a) Lowest energy-ordered states',loc='left',fontweight='bold');axs[0].legend(frameon=False,fontsize=8,loc='upper right')
axs[1].set_yscale('log');axs[1].set_ylim(.001,12);axs[1].set_ylabel('Oscillator strength');axs[1].set_title('(b) Transition intensities',loc='left',fontweight='bold');axs[1].legend(frameon=False,fontsize=8,loc='upper right')
fig.suptitle('All-trans β-carotene: configuration expansion restores weak S₁ below intense S₂',fontsize=11,fontweight='bold');fig.subplots_adjust(left=.07,right=.98,bottom=.20,top=.84,wspace=.28)
fig.savefig(P/'figures/beta_carotene_state_ordering.pdf');fig.savefig(P/'figures/beta_carotene_state_ordering.png',dpi=180);plt.close(fig)
lines=[r'\begin{table}[!htbp]\centering\small',r'\caption{Carotene vertical configuration-expansion control. Energies in eV; dimensionless oscillator strengths.}',r'\begin{tabular}{rrrrrrr}\toprule $N$ & $W$ & CSFs & $E_1$ & $f_1$ & $E_2$ & $f_2$ \\ \midrule']
for row in rows:
 a,b=row['states'][:2];lines.append(f"{row['N']} & {row['W']} & {row['basis_size']} & {a['excitation_ev']:.3f} & {a['oscillator_strength']:.4f} & {b['excitation_ev']:.3f} & {b['oscillator_strength']:.3f} "+r'\\')
lines += [r'\bottomrule\end{tabular}\end{table}'];(P/'tables/beta_carotene_expansion.tex').write_text('\n'.join(lines)+'\n')
