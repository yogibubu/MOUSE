"""Rebuild final validation figures/tables from packaged JSON; no electronic jobs."""
from pathlib import Path
import json,statistics,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
D=json.loads((R/'data/final_validation.json').read_text())
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
blue='#2079A5';orange='#D07820';green='#618D39';red='#B34F51'
tex=lambda s:str(s).replace('_',r'\_')
def table(name,caption,cols,head,rows,label):
 lines=[r'\begin{table}[!htbp]',r'\centering\small',r'\caption{'+caption+r'}\label{'+label+'}',r'\begin{tabular}{'+cols+'}',r'\toprule',head+r' \\',r'\midrule']+[' & '.join(map(str,r))+r' \\' for r in rows]+[r'\bottomrule',r'\end{tabular}',r'\end{table}']
 (R/f'tables/{name}.tex').write_text('\n'.join(lines)+'\n')
anchors=[r for r in D['anchors'] if 'change_ev' in r];names=['Furan','Cyclopentadiene','Phthalimide']
fig,axes=plt.subplots(1,2,figsize=(9.8,3.8),layout='constrained')
for name,color in zip(names,[blue,orange,green]):
 rows=sorted([r for r in anchors if r['molecule']==name],key=lambda r:r['scale'])
 if rows:
  points=sorted([(r['scale'],r['change_ev']) for r in rows]+[(1.,0.)]);axes[0].plot([p[0] for p in points],[p[1] for p in points],'o-',color=color,label=name)
axes[0].axhline(0,color='#555555',lw=.7);axes[0].set(xlabel='Reference internal Cartesian scale',ylabel='Change at common target geometry / eV',title='(a) Complete F/Q anchor recalibration');axes[0].set_xticks([.98,1.,1.02]);axes[0].legend(frameon=False,fontsize=8)
allrows=[];positions=[];ticklabels=[];groups=[]
for cohort,label in [('independent','Primary'),('independent_challenge','New molecule'),('independent_state_challenge','New states')]:
 score=D['cohorts'][cohort]['score']
 if score is None:continue
 start=len(allrows)
 for row in score['rows']:
  x=len(allrows);allrows.append(row);ticklabels.append(row['id']);positions.append(x)
  if 'error_ev' in row:axes[1].plot(x-.09,row['error_ev'],'o',color=blue,label='OO 50/50' if not any('error_ev'in z for z in allrows[:-1]) else None,ms=5)
  else:axes[1].plot(x,0,'x',color='#777777',ms=6);axes[1].annotate('outside\ncoverage',(x,0),xytext=(0,9),textcoords='offset points',ha='center',fontsize=6)
  q=next((p for p in (row.get('tda_gfn2') or {}).get('parents',[]) if p['profile']=='Q' and p['status']=='matched'),None)
  if q:axes[1].plot(x+.09,q['error_ev'],'s',color=orange,ms=4,label='TDA-Q / GFN2' if not any(next((p for p in (z.get('tda_gfn2')or{}).get('parents',[]) if p['profile']=='Q' and p['status']=='matched'),None) for z in allrows[:-1]) else None)
 groups.append((start,len(allrows)-1,label))
for start,end,label in groups:
 if start:axes[1].axvline(start-.5,color='#bbbbbb',lw=.7)
 axes[1].text((start+end)/2,1.02,label,transform=axes[1].get_xaxis_transform(),ha='center',fontsize=7)
axes[1].axhline(0,color='#555555',lw=.7);axes[1].set_xticks(positions,ticklabels,rotation=45,ha='right');axes[1].set(ylabel='Signed excitation-energy error / eV',xlabel='Frozen state identifier',title='(b) Prospective tests kept separate');axes[1].legend(frameon=False,fontsize=8);axes[1].set_title('(b) Prospective tests kept separate',pad=24)
fig.savefig(R/'figures/final_anchor_independent.pdf');fig.savefig(R/'figures/final_anchor_independent.png',dpi=180);plt.close(fig)
table('final_anchor','Full F/Q and projected-integral recalibration, evaluated at the original common geometry. The original model definition is reproduced exactly at unit scale. All electronic calculations use Oracle with two threads.','lrrrr','Molecule & Scale & Gap / eV & Change / eV & Residual / $E_h$',[(r['molecule'],f"{r['scale']:.2f}",f"{r['excitation_ev']:.5f}",f"{r['change_ev']:+.5f}",f"{r['residual']:.1e}") for r in anchors],'tab:finalanchor')
rows=[];stats=[]
for cohort,label in [('independent','Primary'),('independent_challenge','Furofuran'),('independent_state_challenge','Benzoxadiazole states')]:
 score=D['cohorts'][cohort]['score']
 if score is None:continue
 state_labels={r['id']:r.get('state_label','') for r in D['cohorts'][cohort]['manifest']['states']}
 qq=[];oo=[]
 for r in score['rows']:
  q=next((p for p in (r.get('tda_gfn2') or {}).get('parents',[]) if p['profile']=='Q' and p['status']=='matched'),None)
  energy=f"{r['excitation_ev']:.3f}" if 'error_ev' in r else '--';err=f"{r['error_ev']:+.3f}" if 'error_ev'in r else '--';qerr=f"{q['error_ev']:+.3f}" if q else '--'
  identifier=r['id']+(r'$^{d}$' if r.get('partial_double_character') else r'$^{u}$' if r.get('reference_safe')=='N' else '')
  rows.append((identifier,r['molecule'].replace('_',' '),'$'+state_labels.get(r['id'],'')+'$',r.get('rank','--'),f"{r['reference_ev']:.3f}",energy,err,qerr))
  if 'error_ev' in r and q:oo.append(abs(r['error_ev']));qq.append(abs(q['error_ev']))
 stats.append(dict(cohort=cohort,label=label,attempted=score['n_attempted'],matched=score['n_matched'],oo=score['overall'],common_n=len(oo),common_oo_mae=statistics.mean(oo) if oo else None,common_q_mae=statistics.mean(qq) if qq else None))
table('final_independent','Frozen primary and subsequent challenges, retained separately by identifier I, C and S. S states share a primary-cohort molecule; they are not new molecular holdouts. Dashes denote no qualified energy, never zero error. $^d$ flags partial double character in the source; $^u$ flags a source reference marked unsafe. Both Criegee states retain the former flag. Errors are against the fixed numerical references.','lllrrrrr',r'ID & Molecule & State & Rank & Ref. & OO & $\delta$OO & $\delta$TDA-Q',rows,'tab:finalindependent')
(R/'data/final_validation_metrics.json').write_text(json.dumps(stats,indent=2)+'\n')
f=D['geometries']['Furan']['reference_comparison']
if f:
 fig,axes=plt.subplots(1,3,figsize=(11,3.55),layout='constrained')
 b=f['bonds'];axes[0].barh(np.arange(len(b)),[r['error_angstrom'] for r in b],color=blue);axes[0].set_yticks(np.arange(len(b)),['O-Cα','Cα-Cβ','Cβ-Cβ','Cα-H','Cβ-H']);axes[0].axvline(0,color='#555555',lw=.7);axes[0].set(xlabel='Bond-length error / Å',title='(a) Symmetric stationary structure')
 for ir,color,mark in [('a1',blue,'o'),('b2',orange,'s')]:
  rr=[r for r in f['in_plane_frequency_comparison'] if r['irrep']==ir];axes[1].scatter([r['eom_ccsd_cm1'] for r in rr],[r['mouse_cm1'] for r in rr],s=25,c=color,marker=mark,label=ir.upper())
 axes[1].plot([600,3600],[600,3600],color='#777777',ls='--',lw=.8);axes[1].set(xlabel='EOM-CCSD / cm$^{-1}$',ylabel='MOUSE / cm$^{-1}$',title='(b) Fifteen in-plane frequencies');axes[1].legend(frameon=False,fontsize=8)
 rr=f['nuclear_imaginary_modes'];axes[2].bar(np.arange(len(rr)),[abs(r['frequency_cm1']) for r in rr],color=red);axes[2].set_xticks(np.arange(len(rr)),[r['irrep'].upper() for r in rr]);axes[2].set(ylabel='Imaginary-frequency magnitude / cm$^{-1}$',xlabel='Out-of-plane symmetry',title='(c) Instabilities retained')
 fig.savefig(R/'figures/furan_physical_validation.pdf');fig.savefig(R/'figures/furan_physical_validation.png',dpi=180);plt.close(fig)
 rows=[(r['irrep'].upper(),r['within_irrep_frequency_order'],f"{r['eom_ccsd_cm1']:.1f}",f"{r['mouse_cm1']:.1f}",f"{r['error_cm1']:+.1f}") for r in f['in_plane_frequency_comparison']]
 table('furan_physical_frequencies','Furan symmetric valence A1 stationary-point in-plane frequencies, in cm$^{-1}$. Reference: EOM-CCSD, Gromov et al., DOI 10.1063/1.1578051, Table VI. Correspondence uses symmetry then descending frequency; cross-method mode-vector overlaps are unavailable. The large B2 discrepancy is retained. The five imaginary out-of-plane model modes have no numerical reference entries in that table.','lrrrr','Irrep & Order & EOM-CCSD & MOUSE & Difference',rows,'tab:furanphysical')
 rows=[]
 for r in f['bonds']:rows.append((tex(r['coordinate']),f"{r['eom_ccsd_angstrom']:.4f}",f"{r['mouse_angstrom']:.4f}",f"{r['error_angstrom']:+.4f}",r'\AA'))
 for r in f['angles']:rows.append((tex(r['coordinate']),f"{r['eom_ccsd_degrees']:.3f}",f"{r['mouse_degrees']:.3f}",f"{r['error_degrees']:+.3f}",r'deg.'))
 table('furan_physical_structure','Furan internal coordinates versus the same EOM-CCSD stationary-point reference (Table V). Symmetry-equivalent coordinates are averaged.','lrrrr','Coordinate & EOM-CCSD & MOUSE & Difference & Unit',rows,'tab:furanstructure')
# Complete prepared-property cost includes E, E+G and E+G+H scopes.
fig,axes=plt.subplots(1,2,figsize=(9.8,3.6),layout='constrained');rows=[]
for prop,color,mark,label in [('tda',orange,'s','Both-parent TDA'),('energy',blue,'o','OO energy'),('gradient',green,'^','OO energy + gradient'),('hessian',red,'D','OO energy + gradient + Hessian')]:
 pts=[]
 for name in ['nitroxyl','acetaldehyde','acetone','Furan','Cyclopentadiene','Phthalimide']:
  rr=[r for r in D['cost'] if r['molecule']==name and r['property']==prop];ok=[r for r in rr if r['status']=='complete']
  if prop=='tda':assert all('true singlet' in r.get('tda_operator','') for r in ok)
  if ok:
   med=statistics.median(r['wall_seconds'] for r in ok);rss=max(r['process_peak_rss_kib'] for r in ok)/1024;pts.append((ok[0]['nao'],med,rss));rows.append((name,prop,len(ok),ok[0]['nao'],ok[0]['rank'],f'{med:.3f}',f'{rss:.1f}'))
  elif any(r['status']=='right_censored' for r in rr):
   r=next(r for r in rr if r['status']=='right_censored');peer=next(p for p in D['cost'] if p['molecule']==name and 'nao' in p);nao=peer['nao'];rss=r['peak_rss_kib']/1024
   rows.append((name,prop,'1*',nao,peer['rank'],r'$\geq600$',f"{rss:.1f}"));axes[0].scatter([nao],[600],marker='^',facecolors='none',edgecolors=red,s=35);axes[1].scatter([nao],[rss],marker='D',facecolors='none',edgecolors=red,s=25)
   axes[0].annotate('H ≥600 s',xy=(nao,600),xytext=(-4,9),textcoords='offset points',ha='right',fontsize=7,color=red)
 if pts:
  pts.sort();axes[0].plot([p[0] for p in pts],[p[1] for p in pts],marker=mark,color=color,label=label,ms=4);axes[1].plot([p[0] for p in pts],[p[2] for p in pts],marker=mark,color=color,ms=4)
axes[0].set(yscale='log',xlabel='Compact AO dimension',ylabel='Prepared-property wall time / s',title='(a) Fresh-process prepared evaluation');axes[0].legend(frameon=False,fontsize=7)
axes[1].set(xlabel='Compact AO dimension',ylabel='Peak whole-process RSS / MiB',title='(b) Memory includes setup and imports')
fig.savefig(R/'figures/final_cost_scaling.pdf');fig.savefig(R/'figures/final_cost_scaling.png',dpi=180);plt.close(fig)
table('final_cost','Prepared-property costs on Oracle, two threads. Median wall time over n completed fresh processes (three for small cases, one for the largest three sizes) and maximum whole-process RSS. OO Hessian scope includes energy and total gradient. The three smaller cases use rank one and the three larger cases rank two; size and CSF-rank effects are not separated. Both-parent all-root TDA and a single OO state are different outputs. DFT/FQ preparation and model loading are excluded from the property timer and audited separately.','llrrrrr','Molecule & Property & n & AOs & Rank & Time / s & RSS / MiB',rows,'tab:finalcost')
print('Rebuilt final validation figures and tables from packaged data')
# Preparation stages from the fresh molecular cohort; unknown is never zero.
rows=[]
for r in (D.get('preparation_audit') or {}).get('whole_prediction_workers',[]):
 ground=r.get('df_t_ground_cost') or {};cal=r.get('full_response_calibration') or {};td=r.get('reference_tda_costs') or []
 fmt=lambda x:f'{x:.2f}' if x is not None else '--'
 rows.append((r['molecule'].replace('_',' '),r['compact_nao'],fmt(ground.get('seconds')),fmt(sum(t['seconds'] for t in td)) if td else '--',fmt(cal.get('seconds')),fmt(r.get('wall_after_lock_and_import_seconds'))))
table('final_preparation','Available preparation stages for the fresh molecular cohort, in seconds. DFT includes the ground calculation; reference TDA supplies transition identity and rank. F calibration is a separate recorded stage. The final column is worker wall time after lock and imports through prediction closure, not a sum of stages or necessarily cold preparation. Ammonia reuses an earlier DFT checkpoint. Dashes denote unavailable/not executed stages. Historical six-size benchmark preparation totals are unavailable and are not assigned zero.','lrrrrr',r'Molecule & AOs & DFT & Ref. TDA & F calib. & Worker',rows,'tab:finalprep')
# Before/after timings use identical property scopes and frozen energy models.
rows=[]
old=(D.get('hessian_acceleration') or {}).get('baseline_cost',[])
for name in ['nitroxyl','acetaldehyde','acetone','Furan','Cyclopentadiene','Phthalimide']:
 values=[]
 for source in [old,D['cost']]:
  for prop in ['gradient','hessian']:
   rr=[r for r in source if r['molecule']==name and r['property']==prop]
   ok=[r for r in rr if r['status']=='complete']
   values.append(f"{statistics.median(r['wall_seconds'] for r in ok):.3f}" if ok else r'$\geq600$' if any(r['status']=='right_censored' for r in rr) else '--')
 rows.append((name,*values))
table('final_hessian_acceleration','Analytic derivative acceleration on the same prepared models: old and current energy-plus-gradient and energy-plus-gradient-plus-Hessian wall times in seconds. Small-case values are medians of three fresh processes; large cases have one. Earlier censored phthalimide timing is retained as a lower bound. Energy models, ranks, weights, scaling and derivative definitions are unchanged.','lrrrr','Molecule & Old E+G & Old E+G+H & New E+G & New E+G+H',rows,'tab:hessianacceleration')
