"""Archive compact pigment evidence and the qualified source snapshot remotely."""
from pathlib import Path
import json,hashlib,shutil,socket,datetime
import numpy as np
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';out=HERE/'pigment_evidence_snapshot';out.mkdir(exist_ok=True)
files=['beta_convergence_assessment_v1.json','screening_numerical_verification.json','pigment_rescreen_qualification.json','beta_convergence_protocol_v2.json','canonical_kernel_qualification.json','beta_carotene_convergence_w16.json','beta_convergence_protocol.json','frontier_convergence_qualification.json','beta_convergence_assessment.json','beta_carotene_convergence_w12.json','pigment_solver_primitives.json','matrix_free_roks_certificate.json','large_deltascf_certificate.json','density_chart_validation/certificate.json','exact_quadrature_validation/certificate.json','pair_contraction_benchmark.json','pigment_regressions.log','pigment_runtime_qualification.json','pigment_runtime_qualification.log','pigment_status.json','beta_carotene_convergence.json','beta_carotene_convergence_w10.json','pigment_sparse_block_validation.json','pigment_serial_queue.json','fixed_bra_validation.json','pigment_spectral_references.json']
for name in ['Beta_carotene','Chlorophyllide_a']:
 for rel in ['pigment_preflight/{}/stationary_native_certificate.json','pigment_dft_reference/{}/PBE0/certificate.json','pigment_dft_reference/{}/CAM/certificate.json','pigment_embedding/{}/certificate.json','pigment_compact_reference/{}/certificate.json','pigment_full_response/{}/ri_xc_chart/certificate.json','pigment_full_response/{}/ri_xc_chart/reference_mapping.json','pigment_models/{}/manifest.json','pigment_roks/{}/result.json','pigment_screening/{}/certificate.json','pigment_cid/{}/result.json','pigment_cid/{}/result_legacy_observables.json','pigment_cid/{}/observables_phase_corrected.npz','pigment_cid/{}/matrices.npz','pigment_cid/{}/eigenstates.npz']:
  files.append(rel.format(name))
 for pattern in ['state_*/certificate.json','state_*/progress.json','serialized_resume.json']:files.extend(str(path.relative_to(HERE)) for path in (HERE/'pigment_roks'/name).glob(pattern))
for folder in ['pigment_cid_expansion/Beta_carotene','pigment_screening_expansion/Beta_carotene']:
 for pattern in ['**/result.json','**/result_legacy_observables.json','**/observables_phase_corrected.npz','**/analysis.json','**/analysis_coherent.json','**/certificate.json','**/matrices.npz','**/eigenstates.npz']:
  files.extend(str(path.relative_to(HERE)) for path in (HERE/folder).glob(pattern))
manifest={};remote_arrays={}
def digest(path):
 with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def register(path):
 manifest[str(path.relative_to(out))]=dict(bytes=path.stat().st_size,sha256=digest(path))
for file in files:
 source=HERE/file
 if not source.exists():continue
 dest=out/file;dest.parent.mkdir(parents=True,exist_ok=True)
 if source.suffix=='.npz' and source.stat().st_size>90*1024**2:
  receipt=dict(host=socket.getfqdn(),remote_path=str(source),bytes=source.stat().st_size,sha256=digest(source),retention='Complete original checkpoint retained on verified compute host; not replaced or truncated')
  remote_arrays[file]=receipt;receipt_path=Path(str(dest)+'.remote.json');receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');register(receipt_path)
  if source.name=='eigenstates.npz':
   compact=dest.with_name('lowest_eigenstates.npz')
   with np.load(source) as z:
    moments=z['transition_dipoles'][:20]
    corrected=source.parent/'observables_phase_corrected.npz'
    if corrected.exists():
     with np.load(corrected) as a:moments=a['transition_dipoles'][:20]
    np.savez_compressed(compact,energies=z['energies'][:20],ci=z['ci'][:,:20],transition_dipoles=moments,complete_checkpoint_sha256=receipt['sha256'],scope='Lowest twenty states with corrected observable phase; original electronic checkpoints preserved remotely')
   register(compact)
  # Only remove a duplicate from this generated transfer snapshot.
  # The complete numerical checkpoint above is never modified.
  if dest.exists():dest.unlink()
 else:shutil.copy2(source,dest);register(dest)
code={};modules=['_nosi_nuclear.py','_nosi_rescreen.py','_nosi_selected_roots.py','_nosi_canonical_kernel.py','_nosi_configurations.py','_nosi_target_subspace.py','_nosi_canonical.py','_nosi_fixed_bra.py','_angular_transport.py','_quadrature_xc.py','_roks_matrix_free.py','_nosi_mixed_fock.py','_nosi_orthogonality.py','_nosi_slater.py','_nosi_surface_response.py','calculator.py','deltascf.py','multicsf_nuclear.py','nonlocal_response.py','orbital_geometry_reference.py','roks_orbital_response.py']
for filename in modules:
 source=BASE/'source/packages/matrix-gfn2/src/matrix_gfn2'/filename;dest=out/'qualified_sources'/filename;dest.parent.mkdir(exist_ok=True);shutil.copy2(source,dest);code[filename]=hashlib.sha256(source.read_bytes()).hexdigest()
(out/'MANIFEST.json').write_text(json.dumps(dict(host=socket.getfqdn(),captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=manifest,remote_array_files=remote_arrays,qualified_source_sha256=code,source_scope='Qualified final implementation snapshot; numerical jobs preserved checkpoints across development revisions',large_numerical_arrays='Retained on verified Mac Studio, with hashes in model manifests'),indent=2)+'\n');print('Archived',len(manifest),'evidence files and',len(code),'source modules')
