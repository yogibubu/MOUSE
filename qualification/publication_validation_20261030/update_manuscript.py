"""Integrate only completed publication-validation results, preserving blind status."""
from pathlib import Path
import json,unicodedata
P=Path(__file__).resolve().parents[2]/'manuscripts/mouse_orbital_optimized_dft'
p=P/'main.tex';s=p.read_text()
s=s.replace('Its numerical TDA controls isolate this choice within the MOUSE energy models; they are not a direct benchmark of the published sTDA-xTB or XsTD-DFT implementations.','Its matched GFN2 TDA controls isolate this choice within the MOUSE energy models. A separate direct comparison below evaluates published sTDA-xTB; XsTD-DFT is not benchmarked.')
s=s.replace('The manuscript uses the archived final relative-active-pair campaign of 30 October 2026; it introduces no new electronic calculations.','The MOUSE predictions and derivative checks use the archived final relative-active-pair campaign of 30 October 2026. A subsequent publication-validation campaign adds dedicated sTDA-xTB calculations on the identical archived geometries and comparisons with published structural references.')
marker=r'\subsection{Molecular minima and analytic second derivatives}'
if r'\label{fig:directstda}' not in s:
 new=r'''\subsection{Direct comparison with published sTDA-xTB}
Dedicated xtb4stda (release 1.0 with the distributed stda1/stda2 parameter files) supplies the ground-state wavefunction, followed by std2 version 2.0 with \texttt{-xtb -e 10 -vectm 30}. Published method parameters are retained without fitting.\cite{stdaxtb2016,std22026} The archived input XYZs are identical to those used for MOUSE. The recorded contracted Gaussian basis is reconstructed from the xtb4stda wavefunction, checked against its orbital orthonormality, and integrated directly against the fixed DFT reference basis. Each external transition vector is projected into the fixed occupied/virtual reference spaces. Assignments maximize normalized transition overlap, rather than agreement with the TBE. Acceptance requires overlap and projected norm of at least 0.7 and a best-versus-alternative overlap margin of at least 0.05. All 19 requested states satisfy these gates. Executable, parameter, geometry, and transition-vector hashes are retained.

\input{tables/direct_stda.tex}
On the complete curated cohort, the sTDA-xTB MAE is 0.454~eV against 0.183~eV for MOUSE. This average is influenced by sTDA-xTB underestimation for nitroxyl and nitrosomethane; the largest absolute error is 1.506~eV. The result does not establish a general ranking. For the four previous prospective examples, sTDA-xTB has a slightly lower MAE, 0.278 versus 0.304~eV, while the RMSEs are 0.359 and 0.335~eV, respectively. Figure~\ref{fig:directstda} and the complete supporting table show the distribution. These external spectra are distinct from the matched GFN2-parent TDA controls discussed above.

\begin{figure}[!htbp]
\centering\includegraphics[width=\linewidth]{figures/direct_stda_validation.pdf}
\caption{Direct comparison with published dedicated sTDA-xTB on the same geometries and transition-matched states. (a) Every signed error against the common TBE reference. Unmarked states are $n\pi^*$. (b) MAEs with exposed regression and previous prospective subsets kept separate. These are the original curated states; the newly frozen blind benchmark has no scores in this figure.}\label{fig:directstda}
\end{figure}

'''
 s=s.replace(marker,new+marker)
old='A comparison against the released methods on identical geometries and state assignments remains necessary before an accuracy ranking can be made.'
s=s.replace(old,'The separate released sTDA-xTB comparison on identical geometries gives a lower aggregate MAE for MOUSE, but the four previous prospective cases slightly favor sTDA-xTB by MAE. A broader blind comparison is therefore required before a transferable accuracy ranking can be made.')
marker=r'\subsection{Energy-only cost}'
# Locate the timing subsection without assuming its exact title.
idx=s.find(r'\subsection{',s.find(r'\label{fig:hessianchecks}')+1)
if 'CCSDR(3)/def2-TZVPP structural' not in s:
 new=r'''The CCSDR(3)/def2-TZVPP structural comparison is more demanding than verifying derivatives.\cite{geometries2017} For acetaldehyde, acetone, and maleimide, the 11 selected bond lengths differ by $-0.026$ to $+0.007$~\AA. Angular accuracy is less uniform: acetone's carbonyl pyramidalization is underestimated by $11.7^\circ$. Every selected internal coordinate and its reference is reported in the supporting information. For nitroxyl, the three harmonic frequencies ordered from low to high differ from the published (T)+EOM/aug-cc-pVTZ values by $+271$, $+134$, and $+161$~cm$^{-1}$.\cite{hnoharmonic2021} This is a frequency-order precheck pending normal-mode vector confirmation; vibrationally averaged distances and anharmonic fundamentals are excluded. The discrepancies show that analytic correctness alone does not establish spectroscopic accuracy.

'''
 if idx<0:raise RuntimeError('Timing subsection not located')
 s=s[:idx]+new+s[idx:]
marker=r'\section{Conclusions}'
if '44 states in 30 molecules' not in s:
 new=r'''An additional exposure-audited manifest has been frozen before new predictions: 44 states in 30 molecules, comprising 12 $n\pi^*$, 12 $\pi\pi^*$, eight charge-transfer, and 12 Rydberg targets. The four-state CT shortfall relative to the intended quota is retained explicitly. Reference numerical labels remain segregated from the prediction inputs. No scores from this set are included here. An initial algebraic weight precheck on the original 19 vertical anchors gives MAEs of 0.179, 0.183, and 0.239~eV for full-parent weights 0.25, 0.50, and 0.75. The 50/50 candidate remains fixed; these values do not test off-anchor weighted surfaces or Hessians. Full exchange-scaling and anchoring sensitivity, calibrated GFN2 TDA, qualified TD-DFTB/$\Delta$DFTB comparisons, and cold/warm derivative and memory costs remain outstanding.

'''
 s=s.replace(marker,new+marker)
p.write_text(s)
p=P/'supporting_information.tex';s=p.read_text()
if 'tables/all_stda.tex' not in s:
 marker=r'\section{Reference data provenance}'
 new=r'''\section{Direct dedicated sTDA-xTB comparison}
The published dedicated Hamiltonian and parameters are retained. Ground-state xtb4stda wavefunctions and std2 singlet vectors are evaluated at the original XYZs, without geometry optimization or TBE-based root assignment. Cross-basis transition overlaps are normalized only after projection; the norm, overlap, and assignment margin are reported separately in \texttt{data/direct\_stda\_comparison.csv}. All 19 states pass overlap $\geq0.7$, projected norm $\geq0.7$, and margin $\geq0.05$. The small difference between coordinate conversions stored by the legacy executable and the input XYZ is bounded by $2\times10^{-6}$ bohr; no geometry is optimized or refitted. Earlier parameter-path setup failures are preserved separately from the successful electronic attempts.
\input{tables/all_stda.tex}
\section{Physical structural accuracy and weight precheck}
The following internal-coordinate references are from the published CCSDR(3)/def2-TZVPP supporting tables (DOI 10.1021/acs.jctc.7b00921): acetaldehyde S19, acetone S20, and maleimide S27. The geometry definitions and reference provenance are retained in \texttt{data/geometry\_reference\_comparison.json}. No structural reference was used to fit the energy model.
\input{tables/geometry_accuracy.tex}
\input{tables/weight_anchor.tex}
The weight table uses independently stationary vertical parents only. It cannot substitute for reoptimizing a weighted off-anchor coupled problem. The production weight remains 0.5. The separate harmonic nitroxyl precheck is retained in \texttt{data/hno\_harmonic\_comparison.json}; its frequency-order correspondence remains provisional until mode-vector confirmation.

'''
 s=s.replace(marker,new+marker)
p.write_text(s)
p=P/'build_assets.py';s=p.read_text()
if 'extend_publication_results.py' not in s:s+='\nrunpy.run_path(str(P/"extend_publication_results.py"),run_name="__main__")\n'
p.write_text(s)
p=P/'positioning_references.json';data=json.loads(p.read_text())
for v in data.values():
 for k,x in v.items():
  if isinstance(x,str):v[k]=unicodedata.normalize('NFC',x)
p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
