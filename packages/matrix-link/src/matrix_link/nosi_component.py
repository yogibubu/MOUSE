"""Prepared OO-CID evaluation, coherent APOC following and atomic acceptance."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,uuid
import numpy as np


def initialize_nosi(component):
    from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result
    b=component.backend;p=b.native_nosi_model_path
    if hashlib.sha256(p.read_bytes()).hexdigest()!=b.native_nosi_model_sha256:raise ValueError('NOSI model changed from backend contract')
    model=PreparedNOSIModel(p);component._nosi_model=model
    if (not 1<=b.electronic_state<=len(model.state_ids) or model.anchor.charge!=b.charge or model.anchor.multiplicity!=b.multiplicity):raise ValueError('NOSI model and requested charge/spin/state differ')
    root=component.run_dir/'nosi_accepted';manifest=root/'accepted.json'
    if manifest.exists():
        row=json.loads(manifest.read_text());path=(root/row['path']).resolve()
        if (row.get('schema')!='mouse.link.nosi.accepted.v1' or not path.is_relative_to(root.resolve()) or row.get('model_fingerprint')!=model.fingerprint or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']):raise ValueError('NOSI accepted restart contract/checksum mismatch')
        result=load_nosi_result(path,model);component.reference=(result,model.apoc_states(result)[b.electronic_state-1])


def _orbital_state(model,result):
    from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
    return SimpleNamespace(basis=model.anchor.basis,coordinates_bohr=result.coordinates_angstrom*ANGSTROM_TO_BOHR)


def evaluate_nosi_component(component,coordinates,compute_hessian):
    from matrix_apoc import follow_nonorthogonal_ci_state
    from .scan import PointEvaluationResult
    b=component.backend;model=component._nosi_model;previous=None if component.reference is None else component.reference[0]
    properties=('energy','gradient','hessian') if compute_hessian else ('energy','gradient')
    result=model.evaluate(component.calculator._calculator,coordinates,properties=properties,reference_result=previous)
    candidates=model.apoc_states(result);selected=candidates[b.electronic_state-1];state=result.states[b.electronic_state-1];score=state.overlap
    if previous is not None:
        cross=component._cross_overlap(_orbital_state(model,previous),_orbital_state(model,result))
        selected,match=follow_nonorthogonal_ci_state(component.reference[1],candidates,cross,minimum_overlap=b.state_tracking_minimum_overlap,ambiguity_margin=b.state_tracking_ambiguity_margin)
        if selected.state_id!=b.electronic_state:raise ValueError('NOSI coherent state identity changed')
        score=match.overlap
    if score<b.state_tracking_minimum_overlap or state.margin<b.state_tracking_ambiguity_margin:raise ValueError('NOSI state continuity is not certified')
    return PointEvaluationResult(0,0.,energy_hartree=state.energy_hartree,gradient_hartree_per_bohr=state.gradient_hartree_per_bohr.ravel(),hessian_hartree_per_bohr2=state.hessian_hartree_per_bohr2,backend_coordinates_angstrom=np.asarray(coordinates),scf_state=(result,selected),source='native prepared spin-pure OO-CID 50/50',execution=dict(response_model='nosi_50_50_screened_even_cid' if model.extended else 'nosi_50_50_selected_cid',prepared_model_schema=model.manifest['schema'],experimental=True,selected_electronic_state=b.electronic_state,gradient_electronic_state=b.electronic_state,weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0.,ground_scc_evaluations=0,tda_evaluations=0,orbital_residual=result.maximum_stationary_residual,state_assignment_overlap=score,state_assignment_margin=state.margin,spin_squared=selected.spin_squared,model_fingerprint=model.fingerprint))


def persist_nosi(component,result):
    from matrix_gfn2.nosi_surface import NOSISurfaceResult,save_nosi_result
    if (not isinstance(result,NOSISurfaceResult) or result.model_fingerprint!=component._nosi_model.fingerprint or result.maximum_stationary_residual>1e-9 or any(s.excitation_hartree<=0 for s in result.states)):raise ValueError('cannot accept uncertified NOSI state')
    component._nosi_model.validate_files();root=component.run_dir/'nosi_accepted';root.mkdir(parents=True,exist_ok=True);path=root/(uuid.uuid4().hex+'.npz');save_nosi_result(path,result)
    data=dict(schema='mouse.link.nosi.accepted.v1',model_fingerprint=result.model_fingerprint,path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest());tmp=root/'accepted.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(root/'accepted.json')
