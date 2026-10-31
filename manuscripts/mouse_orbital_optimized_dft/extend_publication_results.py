"""Lightweight tables/plots from completed Oracle publication-validation outputs."""
from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
Q=P.parents[1]/'qualification/publication_validation_20261030/remote_prechecks'
for name in ['direct_stda_comparison','geometry_reference_comparison','weight_anchor_precheck','hno_harmonic_comparison']:
 if (Q/(name+'.json')).exists():shutil.copy2(Q/(name+'.json'),P/'data'/(name+'.json'))
 else:assert (P/'data'/(name+'.json')).exists()
old=list(csv.DictReader((P/'data/vertical_energies.csv').open()))
lookup={(r['molecule'],r['state']):r for r in old}
stda=json.loads((P/'data/direct_stda_comparison.json').read_text());rows=[]
for x in stda['rows']:
 r=lookup[x['molecule'],x['state']]
 rows.append(dict(molecule=x['molecule'],state=x['state'],cohort=r['cohort'],tbe_ev=float(r['tbe_ev']),oo_dft_ev=float(r['oo_dft_ev']),stda_xtb_ev=x['excitation_ev'],oo_error_ev=float(r['error_ev']),stda_error_ev=x['excitation_ev']-float(r['tbe_ev']),assigned_root=x['assigned_root'],transition_overlap=x['best_overlap'],projection_norm=x['projection_norm'],assignment_margin=x['margin']))
with (P/'data/direct_stda_comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(rr,key):
 a=np.array([x[key] for x in rr]);return dict(count=len(a),mae_ev=float(np.mean(abs(a))),rmse_ev=float(np.sqrt(np.mean(a*a))),mse_ev=float(np.mean(a)),maxabs_ev=float(max(abs(a))))
groups=[('All 19 states',rows),('Exposed regression',[r for r in rows if r['cohort']=='regression']),('Previous prospective',[r for r in rows if r['cohort']!='regression'])]
summary={label:{'MOUSE':stats(rr,'oo_error_ev'),'sTDA-xTB':stats(rr,'stda_error_ev')} for label,rr in groups}
(P/'data/direct_stda_statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Direct comparison with dedicated published sTDA-xTB on identical archived geometries and transition-matched states. Both methods are evaluated against the same TBEs. The cohort is curated and is not the newly frozen blind set.}\label{tab:directstda}',r'\begin{tabular}{lrrrr}',r'\toprule Subset & $N$ & \multicolumn{2}{c}{MAE (eV)} & sTDA max. error\\',r' & & MOUSE & sTDA-xTB & (eV)\\\midrule']
for label,rr in groups:
 s=summary[label];lines.append(f"{label} & {len(rr)} & {s['MOUSE']['mae_ev']:.3f} & {s['sTDA-xTB']['mae_ev']:.3f} & {s['sTDA-xTB']['maxabs_ev']:.3f} "+r'\\')
lines += [r'\bottomrule\end{tabular}',r'\end{table}'];(P/'tables/direct_stda.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{longtable}{llrrrrr}',r'\caption{Every direct sTDA-xTB comparison. Energies and signed errors are in eV; $q$ is the absolute normalized cross-AO transition overlap.}\label{tab:allstda}\\',r'\toprule Molecule & State & TBE & MOUSE & sTDA-xTB & $\Delta_{\rm sTDA}$ & $q$\\\midrule\endfirsthead',r'\toprule Molecule & State & TBE & MOUSE & sTDA-xTB & $\Delta_{\rm sTDA}$ & $q$\\\midrule\endhead']
for r in rows:
 label=r'$n\pi^*$' if r['state']=='n_pi' else r'$\pi\pi^*$'
 lines.append(f"{r['molecule'].capitalize()} & {label} & {r['tbe_ev']:.3f} & {r['oo_dft_ev']:.3f} & {r['stda_xtb_ev']:.3f} & {r['stda_error_ev']:+.3f} & {r['transition_overlap']:.3f} "+r'\\')
lines += [r'\bottomrule\end{longtable}'];(P/'tables/all_stda.tex').write_text('\n'.join(lines)+'\n')
geo=json.loads((P/'data/geometry_reference_comparison.json').read_text())
lines=[r'\begin{longtable}{llrrrr}',r'\caption{Internal coordinates of MOUSE excited-state minima against CCSDR(3)/def2-TZVPP. These comparisons assess physical accuracy separately from derivative correctness.}\\',r'\toprule Molecule & Coordinate & MOUSE & Reference & Error & Unit\\\midrule\endfirsthead',r'\toprule Molecule & Coordinate & MOUSE & Reference & Error & Unit\\\midrule\endhead']
for r in geo['rows']:
 unit=r'\AA' if r['unit']=='Angstrom' else r'$^\circ$'
 metric=r['metric'].replace('_',r'\_')
 lines.append(f"{r['molecule'].capitalize()} & {metric} & {r['prediction']:.3f} & {r['reference']:.3f} & {r['error']:+.3f} & {unit} "+r'\\')
lines += [r'\bottomrule\end{longtable}'];(P/'tables/geometry_accuracy.tex').write_text('\n'.join(lines)+'\n')
w=json.loads((P/'data/weight_anchor_precheck.json').read_text())
lines=[r'\begin{table}[htbp]\centering\small',r'\caption{Algebraic weight sensitivity for the 19 independently stationary vertical-anchor parent gaps. This is not a test of the weighted off-anchor potential or its Hessian. The production weights remain 0.5/0.5.}',r'\begin{tabular}{rrrr}\toprule $w_F$ & MAE (eV) & RMSE (eV) & Max. error (eV)\\\midrule']
for r in w['statistics']:lines.append(f"{r['weight_full']:.2f} & {r['mae_ev']:.3f} & {r['rmse_ev']:.3f} & {r['maxabs_ev']:.3f} "+r'\\')
lines += [r'\bottomrule\end{tabular}\end{table}'];(P/'tables/weight_anchor.tex').write_text('\n'.join(lines)+'\n')
# Six-panel molecule labels remain readable by putting all signed errors on one tall panel.
navy='#173D52';teal='#168A8A';orange='#D98745';gray='#75838E'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':navy,'text.color':navy})
fig,axs=plt.subplots(1,2,figsize=(9.2,6.7),gridspec_kw={'width_ratios':[1.35,1]});fig.subplots_adjust(left=.19,right=.98,bottom=.13,top=.91,wspace=.43)
a,b=axs;yy=np.arange(len(rows))
for j,(key,color,label) in enumerate([('oo_error_ev',teal,'MOUSE 50/50'),('stda_error_ev',orange,'Published sTDA-xTB')]):
 vals=[r[key] for r in rows];a.scatter(vals,yy+(j-.5)*.20,s=22,color=color,label=label,zorder=3);a.hlines(yy+(j-.5)*.20,0,vals,color=color,alpha=.26,lw=1)
a.axvline(0,color=gray,lw=.8);a.set_yticks(yy,[r['molecule'].capitalize()+(' ($\\pi\\pi^*$)' if r['state']=='pi_pi' else '') for r in rows],fontsize=7.8);a.set_ylim(len(rows)-.4,-.7);a.set_xlabel('Signed energy error (eV)');a.grid(axis='x',alpha=.13);a.set_title('a  Identical geometries and matched states',loc='left',fontsize=10,fontweight='bold',pad=14);a.legend(loc='lower center',bbox_to_anchor=(.5,-.16),frameon=False,fontsize=8,ncol=1)
labels=['All 19','Exposed 15','Prospective 4'];xx=np.arange(3)
for j,(key,color) in enumerate([('MOUSE',teal),('sTDA-xTB',orange)]):
 vals=[summary[g][key]['mae_ev'] for g,_ in groups];bars=b.bar(xx+(j-.5)*.32,vals,width=.30,color=color)
 for bar,value in zip(bars,vals):b.text(bar.get_x()+bar.get_width()/2,value+.012,f'{value:.3f}',ha='center',fontsize=8,color=color,fontweight='bold')
b.set_xticks(xx,labels,rotation=20,ha='right',fontsize=8);b.set_ylabel('Mean absolute error (eV)');b.set_ylim(0,.68);b.set_title('b  Preserve the cohort distinction',loc='left',fontsize=10,fontweight='bold',pad=14);b.text(.02,.96,'Curated cohort; not the new blind set',transform=b.transAxes,va='top',fontsize=7.8,color=gray);b.grid(axis='y',alpha=.13);b.set_axisbelow(True)
for ext in ['pdf','svg','png']:fig.savefig(P/'figures'/('direct_stda_validation.'+ext),dpi=220,bbox_inches='tight',pad_inches=.13)
plt.close(fig)
(P/'data/publication_extension_provenance.json').write_text(json.dumps(dict(host_for_electronic_jobs='oracle.sns.it',cohort='19 archived matched states; no blind scores',assignment='exact cross-AO transition overlaps; best >=0.7, projection norm >=0.7, margin >=0.05',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sources={n:hashlib.sha256((P/'data'/n).read_bytes()).hexdigest() for n in ['direct_stda_comparison.json','geometry_reference_comparison.json','weight_anchor_precheck.json','hno_harmonic_comparison.json']}),indent=2)+'\n')
print(json.dumps(summary,indent=2))

f=P/'figures/FIGURE_PROVENANCE.json'
provenance=json.loads(f.read_text())
provenance['publication_extension']={'figure':'direct_stda_validation','source':'data/direct_stda_comparison.csv','electronic_compute_host':'oracle.sns.it','scope':'19 curated states, not the new blind set','assignment':'cross-AO transition identity with explicit norm/overlap/margin gates'}
f.write_text(json.dumps(provenance,indent=2)+'\n')
