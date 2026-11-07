"""Opt-in LINK multi-CSF surface, coherent APOC following and atomic restart."""
from pathlib import Path
import json,uuid,hashlib
import numpy as np


def initialize_multicsf(component):
    from matrix_gfn2.multicsf_surface import MultiCSFModel,load_multicsf_result
    backend=component.backend;path=backend.native_multicsf_model_path
    if hashlib.sha256(path.read_bytes()).hexdigest()!=backend.native_multicsf_model_sha256:raise ValueError('multi-CSF model changed from backend contract')
    model=MultiCSFModel(path);component._multicsf_model=model
    if model.state_id!=backend.electronic_state or model.anchor.charge!=backend.charge or model.anchor.multiplicity!=backend.multiplicity:raise ValueError('multi-CSF model and requested charge/spin/state differ')
    root=component.run_dir/'multicsf_accepted';manifest=root/'accepted.json'
    if manifest.exists():
        row=json.loads(manifest.read_text());p=(root/row['path']).resolve()
        if row.get('schema')!='mouse.link.multicsf.accepted.v1' or not p.is_relative_to(root.resolve()) or row['model_fingerprint']!=model.fingerprint or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('multi-CSF accepted restart contract/checksum mismatch')
        result=load_multicsf_result(p,model,component.calculator._calculator);component.reference=(result,result.selected_states)


def evaluate_multicsf_component(component,coordinates,compute_hessian):
    from matrix_gfn2.multicsf_surface import evaluate_multicsf_surface
    from matrix_apoc import multicsf_overlap
    from .scan import PointEvaluationResult
    previous=None if component.reference is None else component.reference[0];backend=component.backend
    result=evaluate_multicsf_surface(component.calculator._calculator,coordinates,component._multicsf_model,reference_result=previous,compute_hessian=compute_hessian,minimum_overlap=backend.state_tracking_minimum_overlap)
    scores=None
    if previous is not None:
        cross=component._cross_overlap(previous.reference_state,result.reference_state);scores=[multicsf_overlap(a,b,cross)[0] for a,b in zip(previous.selected_states,result.selected_states)]
    return PointEvaluationResult(0,0.,energy_hartree=result.energy_hartree,gradient_hartree_per_bohr=result.gradient_hartree_per_bohr.ravel(),hessian_hartree_per_bohr2=result.hessian_hartree_per_bohr2,backend_coordinates_angstrom=np.asarray(coordinates),scf_state=(result,result.selected_states),source='native experimental multi-CSF OO-DFT 50/50',execution=dict(response_model='multicsf_50_50_full_active_frame',experimental=True,selected_electronic_state=backend.electronic_state,gradient_electronic_state=backend.electronic_state,weights=[.5,.5],diagonal_lambda_exchange=.9,energy_offset_ev=0.,ground_scc_evaluations=0,tda_evaluations=0,active_rank=component._multicsf_model.rank,orbital_residual=result.orbital_residual,orbital_saddle_index=result.orbital_saddle_index,minimum_absolute_curvature=result.minimum_absolute_curvature,component_state_overlaps=scores,state_assignment_overlap=None if scores is None else min(scores),transition_fidelity=result.fidelity,parent_gaps_hartree=result.parent_gaps_hartree,model_fingerprint=result.model_fingerprint))


def persist_multicsf(component,result):
    from matrix_gfn2.multicsf_surface import MultiCSFSurfaceResult,save_multicsf_result
    if not isinstance(result,MultiCSFSurfaceResult) or result.orbital_residual>1e-9 or result.minimum_absolute_curvature<1e-7 or min(result.parent_gaps_hartree)<=0:raise ValueError('cannot accept uncertified multi-CSF state')
    root=component.run_dir/'multicsf_accepted';root.mkdir(parents=True,exist_ok=True);path=save_multicsf_result(root/uuid.uuid4().hex,result,component._multicsf_model)
    data=dict(schema='mouse.link.multicsf.accepted.v1',model_fingerprint=result.model_fingerprint,path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest());tmp=root/'accepted.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(root/'accepted.json')
