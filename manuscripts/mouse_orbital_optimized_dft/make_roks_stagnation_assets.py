"""Plot archived certification data; performs no electronic calculation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;SOURCE=ROOT.parents[1]/'qualification/closing_validation_20261101/results/chlorophyllide_stagnation_20261102/report.json';data=json.loads(SOURCE.read_text());assert data['complete'];states=data['states'];audit=data['identity_audit']
rows=[]
for item in states:
 cert=item['certificate'];report=cert['report'];rows.append(dict(target=item['state'],energy_ev=cert['excitation_ev'],residual=report['residual'],ground_overlap=report['ground_overlap'],frontier_fidelity=report['fidelity']))
summary=dict(model_sha256=data['model_sha256'],roots=rows,minimum_CSF_metric_eigenvalue=min(audit['CSF_metric_eigenvalues']),scope=data['scope']);(ROOT/'data/chlorophyllide_roks_stationarity.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(8.6,3.7),gridspec_kw={'width_ratios':[1.3,1]},layout='constrained');teal='#007F86';coral='#BD5B36'
# A: the accepted energy-optimization trajectory and its final KKT polishing.
cert=states[1]['certificate'];history=[]
for entry in cert['energy_escape_history']:
 row=dict(entry);error=max(row['residual'],abs(row.get('ground_overlap',0.)))
 if history and abs(error-history[-1][1])<1e-13 and abs(row['energy_hartree']-history[-1][2])<1e-12:continue
 history.append((row.get('stage',''),error,row['energy_hartree']))
x=np.arange(len(history));errors=np.array([item[1] for item in history]);energies=(np.array([item[2] for item in history])-cert['report']['energy_hartree'])*27.211386245988
ax=axes[0];line1=ax.semilogy(x,errors,color=teal,lw=1.7,label='Full KKT residual')[0];ax.axhline(1e-10,color='#777777',ls='--',lw=.9);ax.set(ylim=(1e-15,.2),xlabel='Stored optimization iterates',ylabel=r'Full KKT residual / a.u.',title='A  Escape from residual stagnation');ax.tick_params(axis='y',labelcolor=teal);ax.grid(axis='y',alpha=.16)
right=ax.twinx();right.spines['right'].set_visible(True);line2=right.plot(x,energies,color=coral,lw=1.3,alpha=.9,label=r'$E_{\rm ROKS}-E_{\rm final}$')[0];right.set_ylabel('ROKS energy change / eV',color=coral);right.tick_params(axis='y',labelcolor=coral);ax.legend(handles=[line1,line2],loc='upper right',frameon=False,fontsize=9)
polish=next(i for i,item in enumerate(history) if item[0]=='full KKT polishing');ax.axvline(polish,color='#777777',lw=.8,ls=':');ax.annotate('Full KKT\npolishing',xy=(polish,2e-9),xytext=(-8,0),textcoords='offset points',ha='right',fontsize=9,color='#555555')
# B: all four original target labels, without implying spectral ordering.
ax=axes[1];before=np.array([max(item['before']['residual'],abs(item['before']['ground_overlap'])) for item in states]);after=np.array([item['residual'] for item in rows]);xx=np.arange(1,5)
for a,b,c in zip(xx,before,after):ax.plot([a,a],[b,c],color='#BCC7C9',lw=1.4,zorder=1)
ax.scatter(xx,before,facecolors='white',edgecolors='#7E898B',s=38,label='Before correction',zorder=3);ax.scatter(xx,after,color=teal,s=40,label='Certified',zorder=4);ax.axhline(1e-10,color='#777777',lw=.9,ls='--');ax.set(yscale='log',ylim=(1e-15,.2),xticks=xx,xlabel='Initialization target label',ylabel='Full KKT residual / a.u.',title='B  Four independent ROKS roots');ax.grid(axis='y',alpha=.16);ax.legend(loc='upper right',frameon=False,fontsize=9);ax.text(.97,.62,r'$\lambda_{\min}(S_{\rm CSF})=0.688$',transform=ax.transAxes,ha='right',fontsize=9);ax.annotate('Previously\nconverged',xy=(4,after[3]),xytext=(-10,-27),textcoords='offset points',ha='right',fontsize=9,color='#555555')
fig.savefig(ROOT/'figures/chlorophyllide_roks_convergence.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures/chlorophyllide_roks_convergence.png',dpi=220,bbox_inches='tight');plt.close(fig)
labels=['H$\\to$L','H$-1\\to$L','H$\\to$L$+1$','H$-1\\to$L$+1$']
def sci(value):
 exponent=int(np.floor(np.log10(abs(value))));return f'{value/10**exponent:.2f}\\times10^{{{exponent}}}'
lines=['\\begin{table}[!htbp]\\centering','\\caption{Certified chlorophyllide a ROKS stationary roots on the unchanged prepared surface. Target labels identify the frontier initialization; they are not an energy ordering or a claim of pure single-excitation character. Energies precede screened CID and are not final spectral predictions.}\\label{tab:chlorophyllideroks}','\\begin{tabular}{rlrrrr}\\toprule','Target & Initial label & $\\Delta E$/eV & $\\|R\\|_\\infty$ & $|o|$ & Frontier fidelity\\\\\\midrule']
for row,label in zip(rows,labels):lines.append(f"{row['target']} & {label} & {row['energy_ev']:.6f} & ${sci(row['residual'])}$ & ${sci(abs(row['ground_overlap']))}$ & {row['frontier_fidelity']:.6f}\\\\")
lines+=['\\bottomrule\\end{tabular}','\\end{table}'];(ROOT/'tables/chlorophyllide_roks_stationarity.tex').write_text('\n'.join(lines)+'\n')
