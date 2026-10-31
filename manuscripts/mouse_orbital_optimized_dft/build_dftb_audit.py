"""Rebuild the explicitly limited symmetry-assigned DFTB comparison table."""
from pathlib import Path
import json,shutil
R=Path(__file__).resolve().parent
Q=R.parents[1]/'qualification/publication_validation_20261030/remote_dftb_symmetry_20261031'
D=R/'data/dftb_symmetry';D.mkdir(exist_ok=True)
for name in ['comparison.json','report.json','protocol.json','delta_report.json']:
    if (Q/name).exists():shutil.copy2(Q/name,D/name)
data=json.loads((D/'comparison.json').read_text())
lines=[r'\begin{longtable}{llrrrr}',
       r"""\caption{TD-DFTB2 and spin-purified $\Delta$DFTB2 (mio-1-1) on the identical original 19-state geometries. TD assignments use a pure spatial irrep and the within-irrep ordinal of the fixed DFT reference. $\Delta$DFTB is restricted to the first global reference singlet and requires the final broken-symmetry determinant to have the target spatial irrep. These are symmetry/ordinal comparisons without cross-method transition-overlap qualification. All energies are in eV. ``SCC'' denotes an incomplete archived calculation; ``higher'' denotes an inapplicable higher reference state.}\label{tab:dftbsymmetry}\\""",
       r'\toprule',r'Molecule & State & TBE & MOUSE & TD-DFTB & $\Delta$DFTB \\',r'\midrule\endhead']
for row in data['rows']:
    delta=f"{row['delta_dftb_ev']:.3f}" if row['delta_dftb_ev'] is not None else ('higher' if 'higher' in row['delta_assignment'] else 'SCC')
    state=r'$\pi\pi^*$' if row['state']=='pi_pi' else r'$n\pi^*$'
    lines.append(row['molecule'].replace('_',r'\_')+' & '+state+' & '+f"{row['tbe_ev']:.3f} & {row['oo_dft_ev']:.3f} & {row['td_dftb_ev']:.3f} & {delta}"+r' \\')
lines += [r'\bottomrule',r'\end{longtable}']
(R/'tables/dftb_symmetry_comparison.tex').write_text('\n'.join(lines)+'\n')
