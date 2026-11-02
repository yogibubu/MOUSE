"""Render the implementation-only DFT derivative table from archived checks."""
from pathlib import Path
import json
root=Path(__file__).resolve().parent
archive=root.parents[1]/'qualification/dft_roks_response_20261102/results'
rows=[]
for path in sorted(archive.glob('*_validation.json')):
 r=json.loads(path.read_text())
 if 'method' not in r:continue
 g=max(x['gradient_error'] for x in r['checks']) if 'checks' in r else r['gradient_error']
 h=max(x['Hessian_vector_error'] for x in r['checks']) if 'checks' in r else r['Hessian_vector_error']
 rows.append(dict(molecule=r['molecule'],method=r['method'],kind=r.get('kind','roks'),basis=r['basis'],gradient_error=g,Hessian_vector_error=h))
assert len(rows)==6
(root/'data/dft_response_validation.json').write_text(json.dumps(dict(scope='Derivative implementation only, six checks on two molecules, no accuracy benchmark',rows=rows),indent=2)+'\n')
lines=[r'\begin{table}[htbp]\centering\small',r'\caption{Full-DFT derivative qualification. Maximum independent directional gradient and Hessian--vector errors are in Ha/bohr and Ha/bohr$^2$, respectively. All states are initialized by an alpha HOMO--LUMO promotion.}\label{tab:dftresponse}',r'\begin{tabular}{llllrr}',r'\toprule',r'Molecule & Method & Surface & Basis & Gradient & Hessian--vector\\',r'\midrule']
def number(value):
 a,b=f'{value:.2e}'.split('e');return '$'+a+r'\times10^{'+str(int(b))+'}$'
for r in rows:
 surface=r['kind'].upper() if r['kind']!='deltascf' else r'$\Delta$SCF'
 lines.append(f"{r['molecule'].capitalize()} & {r['method']} & {surface} & {r['basis']} & {number(r['gradient_error'])} & {number(r['Hessian_vector_error'])}"+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(root/'tables/dft_response_validation.tex').write_text('\n'.join(lines)+'\n')
