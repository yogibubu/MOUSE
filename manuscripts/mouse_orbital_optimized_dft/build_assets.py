"""Rebuild manuscript tables and plots from archived results; no electronic jobs."""
from pathlib import Path
import json,csv,shutil,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
Q=ROOT/'qualification/roks_relative_pair_20261030'
D=json.loads(((Q/'SUMMARY.json') if (Q/'SUMMARY.json').exists() else P/'data/SUMMARY.json').read_text())
for name in ['tables','figures','data','coordinates']:(P/name).mkdir(exist_ok=True)
for name in ['SUMMARY.json','PROTOCOL.json','TRANSFER_VERIFICATION.json']:
 if (Q/name).exists():shutil.copyfile(Q/name,P/'data'/name)
 else:assert (P/'data'/name).exists()
refprotocol=ROOT/'qualification/hybrid_spectrum_20261029/frozen_relaxation/protocol.json'
if refprotocol.exists():shutil.copyfile(refprotocol,P/'data/inherited_dft_reference_protocol.json')
names=lambda s:s.replace('_',' ').capitalize()
def table(file,cols,heading,rows,caption,label):
 s='\\begin{table}[!htbp]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+heading+' \\\\\n\\midrule\n'
 s+='\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)
 s+='\n\\bottomrule\n\\end{tabular}\n\\end{table}\n';(P/'tables'/file).write_text(s)
stats=[('Exposed regression',D['regression_statistics']),('Prospective',D['prospective_statistics']),('All states',D['all_statistics']),('$n\\rightarrow\\pi^*$ only',D['npi_statistics'])]
table('statistics.tex','lrrrr','Subset & $N$ & MAE & MSE & Max. abs.',[(n,s['count'],f"{s['mae_ev']:.3f}",f"{s['mean_error_ev']:.3f}",f"{s['maximum_absolute_error_ev']:.3f}") for n,s in stats],'Vertical-energy error statistics in eV; MSE denotes mean signed error, prediction minus TBE.','tab:stats')
pros=D['prospective_states'];rows=[]
for r in pros:rows.append((names(r['molecule']),f"{r['tbe_ev']:.3f}",f"{r['excitation_ev']:.3f}"))
rows.append(('MAE','--','0.304'))
table('prospective.tex','lrr','Molecule & TBE & OO-DFT',rows,'Prospective OO-DFT energies (eV). The four states are valence singlets. Earlier static-response controls are withdrawn as TDA comparisons.','tab:prospective')
mins=D['molecular_minima']
table('minima.tex','lrrr','Molecule & $\\max|g|$ & $\\nu_{\\min}$ & HVP error',[(names(r['molecule']),r"\num{"+f"{r['maximum_gradient']:.2e}"+"}",f"{min(r['frequencies_cm']):.1f}",r"\num{"+f"{r['checks'][-1]['hvp_error']:.2e}"+"}") for r in mins],'Excited-state minima: maximum Cartesian gradient in $E_{\\rm h}$/bohr, minimum projected frequency in cm$^{-1}$, and maximum directional Hessian error at a displacement of $10^{-4}$ bohr in $E_{\\rm h}$/bohr$^2$.','tab:minima')
(P/'data/archival_static_response_timing.json').write_text(json.dumps(dict(status='withdrawn_as_TDA_benchmark',reason='missing antisymmetric nonlocal exchange in static response',archival_measurements=D['timing_with_preparation']),indent=2)+'\n')
allrows=D['regression_states']+pros
errors=[r['excitation_ev']-r['tbe_ev'] for r in allrows]
assert len(allrows)==19 and abs(np.mean(np.abs(errors))-D['all_statistics']['mae_ev'])<1e-12
with (P/'data/vertical_energies.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['molecule','state','cohort','tbe_ev','oo_dft_ev','error_ev','archival_static_response_full_ev','archival_static_response_spectral_ev'])
 for i,r in enumerate(allrows):
  t=r.get('tda_gfn2',r.get('parent_tda_gfn2'))
  w.writerow([r['molecule'],r.get('state','n_pi'),'regression' if i<15 else 'prospective',r['tbe_ev'],r['excitation_ev'],r['excitation_ev']-r['tbe_ev'],t[0]['energy_ev'],t[1]['energy_ev']])
table('all_vertical.tex','llrrr','Molecule & State & TBE & OO-DFT & Error',[(names(r['molecule']),r.get('state','n_pi').replace('n_pi',r'$n\rightarrow\pi^*$').replace('pi_pi',r'$\pi\rightarrow\pi^*$'),f"{r['tbe_ev']:.3f}",f"{r['excitation_ev']:.3f}",f"{r['excitation_ev']-r['tbe_ev']:+.3f}") for r in allrows],'All reported vertical excitation energies and signed errors in eV. The final four rows are prospective tests; the preceding rows are exposed regression examples.','tab:allvertical')
# Recover labels for every optimized geometry from archived accepted CSFs.
symbols={1:'H',6:'C',7:'N',8:'O'}
si=[]
for r in mins:
 joint=next((Q/'remote/molecular'/r['molecule']).rglob('joint.npz'),None)
 if joint is not None:
  with np.load(joint,allow_pickle=False) as a:z=a['parent_0_state_atomic_numbers'].astype(int)
 else:
  lines=(P/'coordinates'/f"{r['molecule']}_minimum.xyz").read_text().splitlines()[2:]
  z=np.array([{v:k for k,v in symbols.items()}[line.split()[0]] for line in lines])
 assert len(z)==len(r['xyz'])
 xyz=str(len(z))+'\n'+r['molecule']+'; constrained OO-DFT minimum; coordinates in Angstrom\n'
 xyz+='\n'.join(f"{symbols[int(zz)]:2s} "+' '.join(f'{v: .10f}' for v in rr) for zz,rr in zip(z,r['xyz']))+'\n'
 (P/'coordinates'/f"{r['molecule']}_minimum.xyz").write_text(xyz)
 si+=['\\par\\noindent\\begin{minipage}{\\linewidth}', '\\subsection{'+names(r['molecule'])+'}', '\\begin{verbatim}',xyz.strip(),'\\end{verbatim}',r'Projected harmonic frequencies (cm$^{-1}$): '+', '.join(f'{v:.2f}' for v in r['frequencies_cm'])+'.\n', '\\end{minipage}\\par\\medskip']
(P/'tables/minimum_coordinates.tex').write_text('\n'.join(si))
provenance=json.loads((P/'reference_provenance.json').read_text())
for name,file in [('uracil','qualification/hybrid_spectrum_20261029/frozen_relaxation/inputs/Uracil.json'),('nitroxyl','qualification/hybrid_spectrum_20261029/frozen_relaxation/inputs/Nitroxyl.json')]:
 if (ROOT/file).exists():provenance['molecules'][name]={'metadata':json.loads((ROOT/file).read_text()),'basis':'aug-cc-pVDZ','basis_source':'inherited_dft_reference_protocol.json','metadata_source':file}
(P/'reference_provenance.json').write_text(json.dumps(provenance,indent=2))
methodrows=[]
for r in allrows:
 choices=[x for x in provenance['molecules'][r['molecule']]['metadata'] if x.get('Spin')==1 and abs(x.get('TBE/AVTZ',-100)-r['tbe_ev'])<1e-6]
 assert len(choices)==1,(r['molecule'],choices)
 c=choices[0]
 methodrows.append({'molecule':r['molecule'],'state':r.get('state','n_pi'),'symmetry':c['State'],'tbe_ev':r['tbe_ev'],'reference_method':c.get('Method'),'safe_flag':c.get('Safe ? (~50 meV)'),'reference_basis':'aug-cc-pVTZ'})
(P/'data/tbe_provenance.json').write_text(json.dumps(methodrows,indent=2))
methods='\n\n'.join('\\noindent\\textbf{'+names(r['molecule'])+' ('+r['state'].replace('_',r'\_')+'):} '+r['reference_method'].replace('_',r'\_')+'.' for r in methodrows)
(P/'tables/tbe_methods.tex').write_text(methods+'\n')
# Bibliographic records: Crossref where accessible, verified publisher/author records otherwise.
meta=json.loads((P/'literature_metadata.json').read_text())
fallback={
'levifaraday':dict(author="Levi, Gianluca and Ivanov, Aleksei V. and J{\\'o}nsson, Hannes",title='Variational calculations of excited states via direct optimization of the orbitals in DFT',journal='Faraday Discussions',year=2020,volume=224,pages='448--466',doi='10.1039/D0FD00064G'),
'ivanov2021':dict(author='Ivanov, Aleksei V. and Levi, Gianluca and Jónsson, Elvar Ö. and Jónsson, Hannes',title='Method for Calculating Excited Electronic States Using Density Functionals and Direct Orbital Optimization with Real Space Grid or Plane Wave Basis Set',journal='Journal of Chemical Theory and Computation',year=2021,volume=17,pages='5034',doi='10.1021/acs.jctc.1c00157'),
'gfn2':dict(author='Bannwarth, Christoph and Ehlert, Sebastian and Grimme, Stefan',title='{GFN2-xTB}: An Accurate and Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method with Multipole Electrostatics and Density-Dependent Dispersion Contributions',journal='Journal of Chemical Theory and Computation',year=2019,volume=15,pages='1652--1671',doi='10.1021/acs.jctc.8b01176'),
'quest2025':dict(author="Loos, Pierre-Fran{\\c c}ois and Boggio-Pasqua, Martial and Blondel, Aymeric and Lipparini, Filippo and Jacquemin, Denis",title='{QUEST} Database of Highly-Accurate Excitation Energies',journal='Journal of Chemical Theory and Computation',year=2025,volume=21,pages='8010--8033',doi='10.1021/acs.jctc.5c00975')}
def latex(s):
 for a,b in [('ö',r'{\"o}'),('Ö',r'{\"O}'),('ó',r"{\'o}"),('é',r"{\'e}"),('ë',r'{\"e}'),('ü',r'{\"u}'),('ä',r'{\"a}'),('ç',r'{\c c}'),('–','--')]:s=s.replace(a,b)
 return s.replace('<scp>','').replace('</scp>','')
records=[]
for key in ['levifaraday','levi2020','ivanov2021','gmf2023','freeze2026','gfn2','roks1998','pbe0','quest','quest2025','tda']:
 m=meta.get(key,{})
 if m.get('title'):
  a=dict(author=' and '.join(x['family']+', '+x.get('given','') for x in m['author']),title=m['title'][0],journal=m['container-title'][0],year=m['published']['date-parts'][0][0],volume=m.get('volume',''),pages=m.get('page',m.get('article-number','')),doi=m['DOI'])
  if key=='quest':a['pages']='e1517'
  if key=='tda':a['pages']='244104'
 else:a=fallback[key]
 a['title']='{'+a['title']+'}'
 records.append('@article{'+key+',\n'+',\n'.join('  '+k+' = {'+latex(str(v))+'}' for k,v in a.items())+'\n}')
for key,a in json.loads((P/'positioning_references.json').read_text()).items():
 import re
 a={k:re.sub(r'\s+',' ',re.sub('<[^>]+>','',str(v))).strip() for k,v in a.items()}
 a['title']='{'+a['title']+'}'
 records.append('@article{'+key+',\n'+',\n'.join('  '+k+' = {'+latex(str(v))+'}' for k,v in a.items())+'\n}')
(P/'references.bib').write_text('\n\n'.join(records)+'\n')
print('Built 19-state tables, figure, 5 optimized XYZ files, TBE method provenance and updated positioning references.')

for r in allrows:
 name=r['molecule'];state=r.get('state','n_pi')
 base=Q/'remote/vertical'/name/state
 path=base/'stationary.npz' if (base/'stationary.npz').exists() else Q/'remote/validation_G'/name/'relative_joint.npz'
 if not path.exists():
  assert (P/'coordinates'/f'{name}_anchor.xyz').exists();continue
 with np.load(path,allow_pickle=False) as a:
  z=a['parent_0_state_atomic_numbers'].astype(int);xyz=a['parent_0_state_coordinates_bohr']/1.8897261254578281
 text=str(len(z))+'\n'+name+'; frozen vertical anchor; Angstrom\n'
 text+='\n'.join(f"{symbols[int(zz)]:2s} "+' '.join(f'{v: .10f}' for v in rr) for zz,rr in zip(z,xyz))+'\n'
 (P/'coordinates'/f'{name}_anchor.xyz').write_text(text)

# Final publication styling; all inputs are archived results or labelled schematics.
import runpy
runpy.run_path(str(P/"make_publication_figures.py"),run_name="__main__")

runpy.run_path(str(P/"extend_publication_results.py"),run_name="__main__")
runpy.run_path(str(P/"make_solver_figure.py"),run_name="__main__")
