"""Plot archived scalar CI results only; no electronic work is performed."""
from pathlib import Path
import json, shutil, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
Q=P.parents[1]/'qualification/closing_validation_20261101/results/pigments'
D=P/'data/pigments';D.mkdir(exist_ok=True)
cases=[('N=2, W=5','pigment_cid/Beta_carotene',2,5),('N=4, W=5','pigment_cid_expansion/Beta_carotene/frontier_4',4,5),('N=5, W=5','pigment_cid_expansion/Beta_carotene/frontier_5',5,5),('N=5, W=6','pigment_cid_expansion/Beta_carotene/screening_6/frontier_5',5,6),('N=6, W=6','pigment_cid_expansion/Beta_carotene/screening_6/frontier_6',6,6)]
for n in [6,7,8]:
 folder=f'pigment_cid_expansion/Beta_carotene/screening_8/frontier_{n}'
 if (Q/folder/'result.json').exists():cases.append((f'N={n}, W=8',folder,n,8))
for n in [8,9,10]:
 folder=f'pigment_cid_expansion/Beta_carotene/screening_10/frontier_{n}'
 if (Q/folder/'result.json').exists():cases.append((f'N={n}, W=10',folder,n,10))
rows=[]
for label,folder,n,w in cases:
 source=Q/folder; dest=D/f'n{n}_w{w}';dest.mkdir(exist_ok=True)
 for filename in ['result.json','analysis.json']:
  if (source/filename).exists():
   data=json.loads((source/filename).read_text());data.pop('auxiliary_records',None)
   data['states']=data['states'][:10];data['compact_view']='Lowest ten roots; full spectrum and configurations in qualification archive'
   data['source_sha256']=hashlib.sha256((source/filename).read_bytes()).hexdigest()
   (dest/filename).write_text(json.dumps(data,indent=2)+'\n')
 report=json.loads((source/'result.json').read_text())
 states=report.get('states',report.get('spectrum'))[:10]
 rows.append(dict(label=label,N=n,W=w,basis_size=report['basis_size'],states=states))
shutil.copyfile(Q/'pigment_spectral_references.json',D/'spectral_references.json')
for name in ['beta_carotene_convergence.json','beta_carotene_convergence_w10.json']:
 if (Q/name).exists():shutil.copyfile(Q/name,D/name)
(D/'ordering_summary.json').write_text(json.dumps(dict(parameters_retuned=False,weights=[.5,.5],lambda_exchange=.9,eta=.6,geometry_unchanged=True,scope='Exposed vertical development test; pigments excluded from 26-state accuracy ledger',rows=rows),indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(10.5,4.7));colors=['#275f9b','#d47722','#647975'];markers=['o','s','^']
for j in range(3):
 for i,row in enumerate(rows[:5]):
  state=row['states'][j]
  axs[0].scatter(i,state['excitation_ev'],color=colors[j],marker=markers[j],s=48,label=f'Energy root {j+1}' if i==0 else None,zorder=3)
  axs[1].bar(i+(j-1)*.23,state['oscillator_strength'],width=.21,color=colors[j],label=f'Root {j+1}' if i==0 else None)
for ax in axs:
 ax.set_xticks(range(5),[r['label'].replace(', ','\n')+f"\n{r['basis_size']} CSFs" for r in rows[:5]],fontsize=11);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axs[0].set_ylabel('Vertical excitation / eV');axs[0].set_ylim(2.25,4.8);axs[0].set_title('(a) Lowest energy-ordered states',loc='left',fontweight='bold');axs[0].legend(frameon=False,fontsize=11,loc='upper right')
axs[1].set_yscale('log');axs[1].set_ylim(.001,12);axs[1].set_ylabel('Oscillator strength');axs[1].set_title('(b) Transition intensities',loc='left',fontweight='bold');axs[1].legend(frameon=False,fontsize=11,loc='upper right')
fig.suptitle('β-carotene: initial configuration-space ordering repair',fontsize=13,fontweight='bold');fig.subplots_adjust(left=.07,right=.98,bottom=.20,top=.84,wspace=.28)
fig.savefig(P/'figures/beta_carotene_state_ordering.pdf');fig.savefig(P/'figures/beta_carotene_state_ordering.png',dpi=180);plt.close(fig)
lines=[r'\begin{table}[!htbp]\centering\small',r'\caption{Carotene vertical configuration-expansion control. Energies in eV; dimensionless oscillator strengths.}',r'\begin{tabular}{rrrrrrrrr}\toprule $N$ & $W$ & CSFs & $E_1$ & $f_1$ & $E_2$ & $f_2$ & $E_3$ & $f_3$ \\ \midrule']
for row in rows:
 a,b,c=row['states'][:3];lines.append(f"{row['N']} & {row['W']} & {row['basis_size']} & {a['excitation_ev']:.3f} & {a['oscillator_strength']:.4f} & {b['excitation_ev']:.3f} & {b['oscillator_strength']:.3f} & {c['excitation_ev']:.3f} & {c['oscillator_strength']:.3f} "+r'\\')
lines += [r'\bottomrule\end{tabular}\end{table}'];(P/'tables/beta_carotene_expansion.tex').write_text('\n'.join(lines)+'\n')
# Separate the fixed-screening convergence sequence from the original repair.
complete_windows=[w for w in [8,10] if len([r for r in rows if r['W']==w])==3]
window=max(complete_windows) if complete_windows else 8
convergence=[r for r in rows if r['W']==window]
if len(convergence)==3:
 fig,axs=plt.subplots(1,2,figsize=(10.0,4.2))
 for j in range(3):
  axs[0].plot([r['N'] for r in convergence],[r['states'][j]['excitation_ev'] for r in convergence],color=colors[j],marker=markers[j],label=f'Root {j+1}',lw=1.2)
  for i,row in enumerate(convergence):axs[1].bar(i+(j-1)*.23,row['states'][j]['oscillator_strength'],width=.21,color=colors[j])
 axs[0].set_xticks([r['N'] for r in convergence],[f"N={r['N']}\n{r['basis_size']} CSFs" for r in convergence]);axs[0].set_ylabel('Vertical excitation / eV');axs[0].set_title('(a) Fixed-screening energy convergence',loc='left',fontweight='bold');axs[0].legend(frameon=False,fontsize=11)
 axs[1].set_xticks(range(3),[f"N={r['N']}" for r in convergence]);axs[1].set_yscale('log');axs[1].set_ylabel('Oscillator strength');axs[1].set_title('(b) Weak / intense ordering',loc='left',fontweight='bold')
 for ax in axs:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 fig.suptitle(f'β-carotene: larger CI windows at fixed screening W = {window}',fontweight='bold',fontsize=13);fig.subplots_adjust(left=.08,right=.98,bottom=.18,top=.82,wspace=.30)
 fig.savefig(P/'figures/beta_carotene_convergence.pdf');fig.savefig(P/'figures/beta_carotene_convergence.png',dpi=180);plt.close(fig)
