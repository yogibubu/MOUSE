"""Archive compact pigment evidence and the qualified source snapshot remotely."""
from pathlib import Path
import json,hashlib,shutil,socket,datetime
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';out=HERE/'pigment_evidence_snapshot';out.mkdir(exist_ok=True)
files=['pigment_solver_primitives.json','matrix_free_roks_certificate.json','large_deltascf_certificate.json','density_chart_validation/certificate.json','exact_quadrature_validation/certificate.json','pair_contraction_benchmark.json','pigment_regressions.log','pigment_runtime_qualification.json','pigment_runtime_qualification.log','pigment_status.json','beta_carotene_convergence.json','beta_carotene_convergence_w10.json','pigment_sparse_block_validation.json','pigment_serial_queue.json','fixed_bra_validation.json','pigment_spectral_references.json']
for name in ['Beta_carotene','Chlorophyllide_a']:
 for rel in ['pigment_preflight/{}/stationary_native_certificate.json','pigment_dft_reference/{}/PBE0/certificate.json','pigment_dft_reference/{}/CAM/certificate.json','pigment_embedding/{}/certificate.json','pigment_compact_reference/{}/certificate.json','pigment_full_response/{}/ri_xc_chart/certificate.json','pigment_full_response/{}/ri_xc_chart/reference_mapping.json','pigment_models/{}/manifest.json','pigment_roks/{}/result.json','pigment_screening/{}/certificate.json','pigment_cid/{}/result.json','pigment_cid/{}/matrices.npz','pigment_cid/{}/eigenstates.npz']:
  files.append(rel.format(name))
 for pattern in ['state_*/certificate.json','state_*/progress.json','serialized_resume.json']:files.extend(str(path.relative_to(HERE)) for path in (HERE/'pigment_roks'/name).glob(pattern))
for folder in ['pigment_cid_expansion/Beta_carotene','pigment_screening_expansion/Beta_carotene']:
 for pattern in ['**/result.json','**/analysis.json','**/certificate.json','**/matrices.npz','**/eigenstates.npz']:
  files.extend(str(path.relative_to(HERE)) for path in (HERE/folder).glob(pattern))
manifest={}
for file in files:
 source=HERE/file
 if source.exists():
  dest=out/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);manifest[file]=dict(bytes=source.stat().st_size,sha256=hashlib.sha256(source.read_bytes()).hexdigest())
code={};modules=['_nosi_canonical.py','_nosi_fixed_bra.py','_angular_transport.py','_quadrature_xc.py','_roks_matrix_free.py','_nosi_mixed_fock.py','_nosi_orthogonality.py','_nosi_slater.py','_nosi_surface_response.py','calculator.py','deltascf.py','multicsf_nuclear.py','nonlocal_response.py','orbital_geometry_reference.py','roks_orbital_response.py']
for filename in modules:
 source=BASE/'source/packages/matrix-gfn2/src/matrix_gfn2'/filename;dest=out/'qualified_sources'/filename;dest.parent.mkdir(exist_ok=True);shutil.copy2(source,dest);code[filename]=hashlib.sha256(source.read_bytes()).hexdigest()
(out/'MANIFEST.json').write_text(json.dumps(dict(host=socket.getfqdn(),captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=manifest,qualified_source_sha256=code,source_scope='Qualified final implementation snapshot; numerical jobs preserved checkpoints across development revisions',large_numerical_arrays='Retained on verified Mac Studio, with hashes in model manifests'),indent=2)+'\n');print('Archived',len(manifest),'evidence files and',len(code),'source modules')
