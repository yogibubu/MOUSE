"""Rebuild complete two-molecule derivative validation assets, no electronics."""
from pathlib import Path
import json,shutil,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
Q=R.parents[1]/'qualification/multicsf_integration_20261031/remote_anchor_response'
D=R/'data/multicsf_integration'
for name,source in [('cpd_full_columns.json',Q/'cpd_hessian/report.json'),('cpd_analytic_cost.json',Q/'cpd_hessian/analytic_cost.json'),('rigid_body.json',Q/'rigid_body/report.json')]:
    if source.exists():shutil.copy2(source,D/name)
cpd=json.loads((D/'cpd_full_columns.json').read_text());assert cpd['complete'] and len(cpd['checks'])==33
furan=json.loads((D/'full_columns.json').read_text());assert furan['complete'] and len(furan['checks'])==27
cost=json.loads((D/'cpd_analytic_cost.json').read_text())
times=[t for row in cpd['checks'] for t in row['displacement_seconds']]
summary=dict(analytic_full_hessian_seconds=cost['seconds'],gradient_calls=len(times),
             gradient_seconds_sum=sum(times),gradient_seconds_median=statistics.median(times),
             gradient_seconds_range=[min(times),max(times)],finite_difference_to_analytic_ratio=sum(times)/cost['seconds'],
             scope='same prepared fixed CPD model; 66 independently reoptimized displaced public energy/gradient calls versus one cold analytic Hessian call; excludes DFT/model preparation and transfer/I/O; not TDA or size-scaling comparison')
(D/'cpd_property_cost.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'pdf.fonttype':42,'font.size':9})
fig,axes=plt.subplots(1,2,figsize=(9.7,3.45),layout='constrained')
for report,name,color,marker in [(furan,'Furan','#2079A5','o'),(cpd,'Cyclopentadiene','#D07820','s')]:
    checks=sorted(report['checks'],key=lambda r:r['coordinate'])
    for ax,key in zip(axes,['hessian_column_max_error','energy_gradient_error']):
        ax.plot([r['coordinate'] for r in checks],np.maximum([r[key] for r in checks],1e-16),marker+'-',label=name,color=color,ms=3,lw=.9)
for ax in axes:
    ax.set_yscale('log');ax.set_xlabel('Cartesian column (zero based)');ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel(r'Maximum Hessian error / $E_h$ bohr$^{-2}$')
axes[0].set_title('(a) Every analytic Hessian column',loc='left',fontsize=10)
axes[0].legend(frameon=False,fontsize=8)
axes[1].set_ylabel(r'Energy--gradient error / $E_h$ bohr$^{-1}$')
axes[1].set_title('(b) Independent energy differences',loc='left',fontsize=10)
fig.savefig(R/'figures/multicsf_complete_hessians.pdf')
fig.savefig(R/'figures/multicsf_complete_hessians.png',dpi=180)
lines=[r'\begin{longtable}{rrr}',
       r'\caption{Complete 33-column cyclopentadiene total-excited-energy Hessian check through the public multi-CSF surface. The fixed rank-two model includes native S0 restoration and both corrected ground subtractions. Each signed displacement is independently reoptimized from the accepted anchor; central step is $2\times10^{-4}$ bohr. Units are $E_h$/bohr and $E_h$/bohr$^2$.}\label{tab:cpdcolumns}\\',
       r'\toprule',r'Cartesian column (zero based) & Energy--gradient error & Max. Hessian error \\',r'\midrule\endhead']
for row in sorted(cpd['checks'],key=lambda r:r['coordinate']):
    lines.append(f"{row['coordinate']} & {row['energy_gradient_error']:.2e} & {row['hessian_column_max_error']:.2e}"+r' \\')
lines.extend([r'\bottomrule',r'\end{longtable}'])
(R/'tables/cpd_complete_columns.tex').write_text('\n'.join(lines)+'\n')
