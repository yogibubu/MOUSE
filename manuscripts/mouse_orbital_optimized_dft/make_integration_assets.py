"""Rebuild integration/sensitivity tables and static scientific figure."""
from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;Q=R.parents[1]/'qualification/multicsf_integration_20261031';D=R/'data/multicsf_integration';D.mkdir(exist_ok=True)
inputs={'full_columns.json':Q/'Furan_full_columns_report.json','link_qualification.json':Q/'remote/multicsf_integration/link_validation/report.json','lambda_sensitivity.json':Q/'remote/multicsf_integration/lambda_sensitivity/summary.json','rank_weight_sensitivity.json':Q/'rank_weight_sensitivity.json','effective_rank.json':R/'data/reduced_extension/effective_rank_development.json'}
for name,src in inputs.items():
 if src.exists() and src.resolve()!=(D/name).resolve():shutil.copy2(src,D/name)
rows=json.loads((D/'lambda_sensitivity.json').read_text())['rows'];rank=json.loads((D/'rank_weight_sensitivity.json').read_text());eff=json.loads((D/'effective_rank.json').read_text())['rows']
names=['Cyclopentadiene','Furan','Phthalimide'];colors=['#2079A5','#D07820','#618D39'];fig,axes=plt.subplots(1,3,figsize=(10.6,3.45),layout='constrained');plt.rcParams.update({'pdf.fonttype':42})
lines=[r'\begin{table}[!htbp]',r'\centering\small',r'\caption{Vertical sensitivity of the multi-CSF candidate to diagonal exact-exchange response. One-body DFT anchor, semilocal response, bare off-diagonal CSF couplings and 50/50 weights remain fixed; no fitting is performed. Energies and maximum changes from $\lambda=0.9$ are in eV.}\label{tab:multicsfsens}',r'\begin{lcrr}']
# Use an explicit conventional tabular layout.
lines=lines[:-1]+[r'\begin{tabular}{lrrrr}',r'\toprule',r'Molecule & $\lambda=0.85$ & $0.90$ & $0.95$ & Max. change \\',r'\midrule']
for name,color in zip(names,colors):
 rr=sorted([r for r in rows if r['molecule']==name],key=lambda r:r['lambda_exchange']);x=np.array([r['lambda_exchange'] for r in rr]);y=np.array([r['excitation_ev'] for r in rr]);center=y[1]
 axes[0].plot(x,y-center,'o-',color=color,label=name,lw=1.4,ms=4)
 mix=next(r for r in rank['weight_sensitivity'] if r['molecule']==name);axes[1].plot([.4,.5,.6],[mix['vertical_energies_ev'][str(w)]-mix['vertical_energies_ev']['0.5'] for w in [.4,.5,.6]],'o-',color=color,lw=1.4,ms=4)
 row=next(r for r in eff if r['molecule']==name and r['family']=='pi_pi');w=np.array(row['weights']);t=np.linspace(.85,1.15,201);a=w[0];b=w[1:].sum();c=w[1:]@w[1:];number=(a+t*t*b)**2/(a*a+t**4*c);axes[2].plot(t,number,c=color,lw=1.5)
 lines.append(name+' & '+' & '.join(f'{v:.3f}' for v in y)+f' & {max(abs(y-center)):.3f}'+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(R/'tables/multicsf_sensitivity.tex').write_text('\n'.join(lines)+'\n')
for ax in axes:
 ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18);ax.tick_params(labelsize=8)
for ax in axes[:2]:ax.axhline(0,c='#555555',lw=.7);ax.set_ylabel(r'$E-E_{\mathrm{fixed}}$ / eV',fontsize=9)
axes[0].set(xlabel=r'Diagonal exchange-response $\lambda$',xticks=[.85,.9,.95]);axes[0].set_title('(a) Exchange-response sensitivity',fontsize=9,loc='left');axes[0].legend(fontsize=7,frameon=False)
axes[1].set(xlabel='Full-parent vertical weight',xticks=[.4,.5,.6]);axes[1].set_title('(b) Algebraic vertical weights',fontsize=9,loc='left')
axes[2].axhline(1.5,c='#555555',ls='--',lw=.8);axes[2].axvline(.902477964,c=colors[2],ls=':',lw=.9);axes[2].set(xlabel='Subleading NTO singular-value scale',ylabel='Effective number of NTO pairs');axes[2].set_title('(c) Fixed-reference rank sensitivity',fontsize=9,loc='left');axes[2].tick_params(labelsize=8)
fig.savefig(R/'figures/multicsf_sensitivity.pdf');fig.savefig(R/'figures/multicsf_sensitivity.png',dpi=180)
checks=sorted(json.loads((D/'full_columns.json').read_text())['checks'],key=lambda r:r['coordinate'])
lines=[r'\begin{longtable}{rrr}',r'\caption{Complete 27-column furan total excited-state Hessian validation. Analytic constrained orbital/CI response and all three ground-reference derivatives are included. Central differences reoptimize the state at each displaced geometry with step $2\times10^{-4}$ bohr. Units are $E_h$/bohr for the gradient check and $E_h$/bohr$^2$ for the Hessian.}\label{tab:furancolumns}\\',r'\toprule',r'Cartesian column (zero based) & Energy--gradient error & Max. Hessian error \\',r'\midrule\endhead']
for c in checks:lines.append(f"{c['coordinate']} & {c['energy_gradient_error']:.2e} & {c['hessian_column_max_error']:.2e}"+r' \\')
lines += [r'\bottomrule',r'\end{longtable}'];(R/'tables/furan_complete_columns.tex').write_text('\n'.join(lines)+'\n')
