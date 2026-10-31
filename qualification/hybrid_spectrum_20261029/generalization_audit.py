"""Re-evaluate every accepted held-out ROKS spectral correction from checkpoints."""
from remote_bootstrap import HOST
from pathlib import Path
import json,numpy as np
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2.hybrid_spectrum import MatchedOrbitalSpectrum,spectrum_fingerprint,with_occupied_extent_scaling,correct_roks_excitation
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'generalization';EV=27.211386245988
summary=json.loads((OUT/'summary.json').read_text());rows=[]
for row in summary['rows']:
 if not row.get('roks_qualified'):continue
 folder=OUT/row['molecule'];attempt=next(a for a in row['solver_attempts'] if a.get('state_qualified'))
 index=attempt['observed_index'];hole=row['hole'];particle=row['particle'];path=folder/f'{row["id"]}_index_{index}/converged.npz'
 with np.load(path) as f:xyz=f['coordinates_angstrom']
 key=f'generalization_{row["molecule"]}_{row["id"]}_{hole}_{particle}_{index}'
 result=load_derivative_point(path,request_hash=key,coordinates_angstrom=xyz,require_gradient=False);s=result.reference_state
 assert result.converged and result.orbital_residual<=1e-9 and result.orbital_saddle_index==index and result.spin_squared==0
 assert attempt['canonical_csf_overlap']>=.7 and attempt['reference_transition_overlap']>=.7
 with np.load(folder/'mapping.npz') as f:
  m=MatchedOrbitalSpectrum(f['source_energies'],s.alpha_occupations+s.beta_occupations,f['reference_energies'],f['reference_indices'],f['overlaps'],f['margins'],spectrum_fingerprint(s.coefficients,s.overlap,s.alpha_occupations+s.beta_occupations,s.orbital_energies_hartree),'audit/'+row['molecule'],.7,.05,summary['reference_energy_scale'])
  m=with_occupied_extent_scaling(m,f['source_radii'],f['reference_radii'],power=summary['extent_power'])
 corrected=correct_roks_excitation(result,m,hole,particle)
 difference=corrected.excitation_energy_hartree*EV-row['hybrid_roks_ev'];assert abs(difference)<1e-11
 rows.append(dict(molecule=row['molecule'],state=row['id'],index=index,api_replay_difference_ev=difference,spin_squared=result.spin_squared,residual=result.orbital_residual))
assert len(rows)==summary['qualified_states']
output=dict(host=HOST,accepted_states_replayed=len(rows),maximum_replay_difference_ev=max(abs(x['api_replay_difference_ev']) for x in rows),rows=rows)
(OUT/'molecular_audit.json').write_text(json.dumps(output,indent=2)+'\n');print(output)
