"""Render a protocol-defined saved pigment spectrum; no electronic calculation."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--archive',required=True);args=parser.parse_args();A=Path(args.archive).resolve()
manifest=json.loads((A/'artifact_manifest.json').read_text());assessment_files=list(A.glob('*_assessment.json'));assert len(assessment_files)==1
r=json.loads(assessment_files[0].read_text());p=r['convergence']['protocol'];prefix=p['artifact_prefix'];name=p['molecule'].replace('_',' ');ns=r['convergence']['windows'];fixed=p['fixed_screening_frontier'];roots=p['roots'];states=r['coherent_final_states']
def read(n,w):return json.loads((A/f'spectra/N{n}_W{w}/result.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(9.3,3.8),layout='constrained');colors=['#007F86','#BD5B36','#536A98','#8B704B']
for j,k in enumerate(roots):
 color=colors[j%len(colors)];energies=[read(n,n)['states'][k-1]['excitation_ev'] for n in ns];control=[read(n,fixed)['states'][k-1]['excitation_ev'] for n in ns]
 axs[0].plot(ns,energies,'o-',color=color,lw=1.4,label=f'Root {k}')
 axs[0].plot(ns,control,'s--',color=color,lw=1,ms=4,alpha=.75)
axs[0].set(xlabel='Occupied and virtual frontier size N',ylabel='Vertical excitation / eV',xticks=ns,title='A  Configuration-space convergence');axs[0].grid(axis='y',alpha=.17);axs[0].legend(frameon=False,fontsize=9,ncol=2);axs[0].text(.04,.03,f'Solid: W = N\nDashed: W = {fixed}',transform=axs[0].transAxes,fontsize=9)
shown=[x for x in states if x['excitation_ev']<=max(states[k-1]['excitation_ev'] for k in roots)+.7]
for row in shown:
 color=colors[(row['root']-1)%len(colors)] if row['root'] in roots else '#9CA7A9';axs[1].vlines(row['excitation_ev'],0,row['oscillator_strength'],color=color,lw=2)
 if row['root'] in roots:axs[1].annotate(str(row['root']),(row['excitation_ev'],row['oscillator_strength']),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
axs[1].set(xlabel='Vertical excitation / eV',ylabel='Oscillator strength',title=f'B  Final spin-singlet CID spectrum, N = {ns[-1]}');axs[1].set_ylim(0,max(x['oscillator_strength'] for x in shown)*1.20);axs[1].grid(axis='y',alpha=.17)
fig.savefig(P/f'figures/{prefix}_validation.pdf',bbox_inches='tight');fig.savefig(P/f'figures/{prefix}_validation.png',dpi=200,bbox_inches='tight');plt.close(fig)
summary=dict(molecule=p['molecule'],protocol=p,convergence=r['convergence'],states=states,selected_states=r['coherent_selected_states'],root_tracking=r['root_tracking'],eigenstate_checks=r['eigenstate_checks'],screening_checks=r['screening_checks'],scope=r['scope']);(P/f'data/{prefix}_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=[r'\begin{table}[!htbp]\centering\small',r'\caption{Final spin-singlet screened-CID roots of '+name+f' at $N=W={ns[-1]}$. '+'All energies are vertical gaps on the unchanged source geometry. Native weights refer to the full frozen native-frontier target subspace, not a single orbital promotion.}',r'\label{tab:'+prefix+r'}',r'\begin{tabular}{rrrrr}\toprule',r'Root & $\Delta E$/eV & $f$ & Native target weight & Selected\\\midrule']
for row in states[:12]:lines.append(f"{row['root']} & {row['excitation_ev']:.6f} & {row['oscillator_strength']:.6f} & {row['native_frontier_subspace_weight']:.6f} & {'yes' if row['selected'] else 'no'}\\\\")
lines += [r'\bottomrule\end{tabular}',r'\end{table}'];(P/f'tables/{prefix}_spectrum.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{table}[!htbp]\centering\small',r'\caption{Consecutive frontier increments for '+name+'. Both the coherent $W=N$ sequence and the fixed-interaction sequence must satisfy the frozen 0.02 eV criterion for each monitored root.}',r'\begin{tabular}{ll'+('r'*len(roots))+r'}\toprule','Partition & Increment & '+' & '.join(r'$\delta E_'+str(k)+r'$/eV' for k in roots)+r'\\\midrule']
for label,key in [('$W=N$','coherent_increments_ev'),(f'$W={fixed}$','fixed_Hamiltonian_increments_ev')]:
 for i,values in enumerate(r['convergence'][key]):lines.append(label+f' & {ns[i]}$\\to${ns[i+1]} & '+' & '.join(f'{x:+.6f}' for x in values)+r'\\')
lines += [r'\bottomrule\end{tabular}',r'\end{table}'];(P/f'tables/{prefix}_convergence.tex').write_text('\n'.join(lines)+'\n')
