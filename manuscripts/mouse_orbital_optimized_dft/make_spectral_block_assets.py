"""Rebuild the 26-state development figure from archived scalar evidence."""
from pathlib import Path
import json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;Q=P.parents[1]/'qualification/closing_validation_20261101';D=P/'data/spectral_blocks';D.mkdir(exist_ok=True)
for name in ['spectral_block_summary.json','spectral_block_protocol.json','additional_scored.json','pigment_protocol.json']:shutil.copyfile(Q/name,D/name)
d=json.loads((D/'spectral_block_summary.json').read_text());plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(10.8,5.5),gridspec_kw={'wspace':.48})
for ax,family,title in zip(axs,['pi_pi','n_pi'],[r'$\pi\rightarrow\pi^*$: 12 states',r'$n\rightarrow\pi^*$: 14 states']):
 rows=[r for r in d['rows'] if r['family']==family];y=list(range(len(rows)));ax.axvspan(-.5,.5,color='#e8f0ea',zorder=0);ax.axvline(0,color='.4',lw=.7)
 for i,r in enumerate(rows):ax.plot([r['error_ev'],r['TDA_error_ev']],[i,i],color='.78',lw=1,zorder=1)
 ax.scatter([r['TDA_error_ev'] for r in rows],y,color='#cf8849',marker='x',s=30,label='Matched native-MO TDA',zorder=2);ax.scatter([r['error_ev'] for r in rows],y,color='#126d85',s=30,label='OO-CID, spectral blocks',zorder=3)
 ax.set_yticks(y,[r['molecule'].replace('_',' ')+' '+r['id'] for r in rows]);ax.invert_yaxis();ax.set_title(title,weight='bold');ax.set_xlabel('Signed excitation error / eV');ax.grid(axis='x',alpha=.15)
axs[0].legend(loc='lower right',fontsize=7,frameon=False);fig.subplots_adjust(left=.19,right=.985,top=.89,bottom=.10);fig.suptitle('Complete exposed development ledger; fixed 50/50, λ = 0.9, η = 0.60',fontsize=11);fig.savefig(P/'figures/spectral_block_validation.pdf');plt.close(fig)
lines=[r'{\small\setlength{\tabcolsep}{4pt}\begin{longtable}{lllrrrr}',r'\caption{Spectral-block development amendment. All 26 labels are exposed. Native-orbital TDA is recalculated for each corrected model; energies and signed errors are in eV.}\label{tab:spectralblocks}\\',r'\toprule ID & Molecule & Family & Reference & OO-CID & Error & TDA \\ \midrule\endfirsthead',r'\toprule ID & Molecule & Family & Reference & OO-CID & Error & TDA \\ \midrule\endhead']
for r in d['rows']:
 family=r'$\pi\pi^*$' if r['family']=='pi_pi' else r'$n\pi^*$';lines.append(f"{r['id']} & {r['molecule'].replace('_',' ')} & {family} & {r['reference_ev']:.3f} & {r['prediction_ev']:.3f} & {r['error_ev']:+.3f} & {r['TDA_ev']:.3f} "+r'\\')
lines += [r'\bottomrule\end{longtable}}'];(P/'spectral_block_table.tex').write_text('\n'.join(lines)+'\n')
