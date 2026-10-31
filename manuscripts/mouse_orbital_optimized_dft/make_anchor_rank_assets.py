"""Plot actual displaced-reference rank checks, without recomputing electronics."""
from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
Q=R.parents[1]/'qualification/multicsf_integration_20261031/remote_anchor_response/anchor_rank_symmetry'
D=R/'data/multicsf_integration'
for name,source in [('anchor_rank_actual.json',Q/'summary.json'),('anchor_projection_audit.json',Q/'projection_audit.json')]:
    if source.exists():shutil.copy2(source,D/name)
data=json.loads((D/'anchor_rank_actual.json').read_text());assert data['complete'] and len(data['rows'])==6
audit=json.loads((D/'anchor_projection_audit.json').read_text())
plt.rcParams.update({'pdf.fonttype':42,'font.size':9})
fig,axes=plt.subplots(1,2,figsize=(9.8,3.55),layout='constrained')
names=['Cyclopentadiene','Furan','Phthalimide'];colors=['#2079A5','#D07820','#618D39']
lines=[r'\begin{table}[!htbp]',r'\centering\small',
       r'\caption{Actual displaced-reference rank test. PBE0/aug-cc-pVDZ/grid-3 ground states and two same-symmetry TDA roots are recomputed at uniform Cartesian scales 0.98 and 1.02. A fresh restricted native GFN2 S0 defines the compact projection. $N_{\rm pair}$ and rank refer to that normalized compact transition. $f$ is coherent cross-geometry transition overlap with the original fixed DFT root; $p$ is the compact projection norm. This does not recalibrate the complete F/Q energy model.}\label{tab:actualanchorrank}',
       r'\begin{tabular}{lrrrrr}',r'\toprule',r'Molecule & Scale & $N_{\rm pair}$ & Rank & $f$ & $p$ \\',r'\midrule']
for name,color in zip(names,colors):
    a=next(r for r in audit['rows'] if r['molecule']==name)
    rows=sorted([r for r in data['rows'] if r['molecule']==name],key=lambda r:r['scale'])
    x=[.98,1.,1.02]
    native=[rows[0]['compact_transition']['effective_pairs'],a['original_relaxed_native_compact']['effective_pairs'],rows[1]['compact_transition']['effective_pairs']]
    transported=[a['rows'][0]['transported_anchor_compact']['effective_pairs'],a['original_compact']['effective_pairs'],a['rows'][1]['transported_anchor_compact']['effective_pairs']]
    axes[0].plot(x,native,'o-',color=color,label=name,ms=4,lw=1.3)
    axes[0].plot(x,transported,'--',color=color,lw=.9,alpha=.8)
    overlap=[max(rows[0]['coherent_reference_overlaps']),1.,max(rows[1]['coherent_reference_overlaps'])]
    norm=[rows[0]['compact_projection_norm'],a['original_projection_norm'],rows[1]['compact_projection_norm']]
    axes[1].plot(x,overlap,'o-',color=color,ms=4,lw=1.2)
    axes[1].plot(x,norm,'s--',color=color,ms=3,lw=.9)
    for row in rows:
        lines.append(name+f" & {row['scale']:.2f} & {row['compact_transition']['effective_pairs']:.4f} & {row['compact_transition']['rank']} & {max(row['coherent_reference_overlaps']):.4f} & {row['compact_projection_norm']:.4f}"+r' \\')
for ax in axes:
    ax.set_xlabel('Uniform internal Cartesian scale');ax.set_xticks(x);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
axes[0].axhline(1.5,color='#555555',lw=.8,ls=':')
axes[0].set_ylabel('Effective number of compact NTO pairs')
axes[0].set_title('(a) Actual reference and subspace sensitivity',loc='left',fontsize=9)
axes[0].legend(frameon=False,fontsize=8)
axes[1].set_ylabel('Transition overlap or projection norm');axes[1].set_ylim(.69,1.01)
axes[1].axhline(.7,color='#555555',lw=.8,ls=':')
axes[1].set_title('(b) Identity and compact representability',loc='left',fontsize=9)
axes[1].text(.981,.72,'Circles: transition overlap; squares: compact norm',fontsize=7)
fig.savefig(R/'figures/actual_anchor_rank.pdf');fig.savefig(R/'figures/actual_anchor_rank.png',dpi=180)
lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
(R/'tables/actual_anchor_rank.tex').write_text('\n'.join(lines)+'\n')
