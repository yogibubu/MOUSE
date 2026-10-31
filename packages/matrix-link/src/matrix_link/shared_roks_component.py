"""Automatic coupled-ROKS evaluation and atomic restart of joint certificates."""
import json,hashlib,uuid
from pathlib import Path
import numpy as np
from .composite_roks_component import _models,_contract,_selected


def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def _coupled_io(backend):
    if backend.native_roks_composite_orbital_coupling=='relative_active_pair':
        from matrix_gfn2.relative_pair_roks import RelativePairROKSResult,save_relative_pair_roks,load_relative_pair_roks
        return RelativePairROKSResult,save_relative_pair_roks,load_relative_pair_roks,'mouse.link.relative_roks.accepted.v1'
    from matrix_gfn2.shared_pair_roks import SharedPairROKSResult,save_shared_pair_roks,load_shared_pair_roks
    return SharedPairROKSResult,save_shared_pair_roks,load_shared_pair_roks,'mouse.link.shared_roks.accepted.v1'


def _relative_anchor(joint,anchor):
    from matrix_gfn2.relative_pair_roks import RelativePairROKSResult,relative_active_targets
    if isinstance(joint,RelativePairROKSResult):
        target=anchor.target_squared_overlaps if isinstance(anchor,RelativePairROKSResult) else relative_active_targets(anchor.components)
        if not np.allclose(joint.target_squared_overlaps,target,atol=1e-12,rtol=0):
            raise ValueError('relative-state targets differ from the original assigned anchor')


def _constraint(backend,joint):
    if backend.native_roks_composite_orbital_coupling=='relative_active_pair':
        from matrix_gfn2.relative_pair_roks import RelativePairROKSResult
        if not isinstance(joint,RelativePairROKSResult):raise ValueError('expected a relative active-pair checkpoint')
        return
    occ=joint.occupations
    expected=tuple(np.flatnonzero(occ[0]<occ[1] if backend.native_roks_composite_orbital_coupling=='shared_active_hole' else occ[0]!=occ[1]))
    if joint.shared_orbital_indices!=expected:
        raise ValueError('joint checkpoint has another active-orbital constraint')


def persist_shared(component,scf_state):
    Joint,save_joint,load_joint,schema=_coupled_io(component.backend)
    from matrix_gfn2.ground_referenced_roks import GroundReferencedROKSResult
    from matrix_gfn2.deltascf_derivatives import save_derivative_point
    from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
    from matrix_gfn2 import combine_roks_results
    total=scf_state[0];refs=_models(component.backend);contract=_contract(component.backend)
    if not isinstance(total,GroundReferencedROKSResult) or not isinstance(total.excited,Joint) or not total.spectrum_qualified:
        raise ValueError('accepted shared-pair surface needs a qualified joint solution and all S0 references')
    if not total.excited.converged or total.excited.orbital_residual>1e-8:raise ValueError('joint state is not stationary')
    _constraint(component.backend,total.excited)
    for parent,ref in zip(total.components,refs):
        if parent.spin_reference_fingerprint!=ref.fingerprint:raise ValueError('joint checkpoint and response differ')
    root=component.run_dir/'roks_50_50_accepted';folder=root/uuid.uuid4().hex;folder.mkdir(parents=True)
    data=dict(schema=schema,contract=contract)
    def row(path):return dict(path=str(path.relative_to(root)),sha256=_sha(path))
    joint=folder/'joint.npz';save_joint(joint,total.excited,request_hash=contract+':reference');data['reference']=row(joint)
    anchor=(component.anchor or scf_state)[0]
    if isinstance(anchor,GroundReferencedROKSResult):anchor=anchor.excited
    _relative_anchor(total.excited,anchor)
    if isinstance(anchor,Joint):
        path=folder/'anchor_joint.npz';save_joint(path,anchor,request_hash=contract+':anchor');data['anchor']=dict(kind='shared',checkpoint=row(path))
    else:
        checked=combine_roks_results(*anchor.components,component_tracking=component.backend.native_roks_composite_component_tracking or 'csf');rows=[]
        for i,parent in enumerate(checked.components):
            path=folder/f'anchor_{i}.npz';xyz=parent.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR
            save_derivative_point(path,parent,request_hash=contract+':anchor:'+str(i),coordinates_angstrom=xyz)
            rows.append(dict(**row(path),coordinates_angstrom=xyz.tolist()))
        data['anchor']=dict(kind='independent',checkpoints=rows)
    grounds=[]
    for i,ground in enumerate(total.grounds):
        path=folder/f'ground_{i}.npz';xyz=ground.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR
        save_derivative_point(path,ground.reference,request_hash=contract+':ground:'+str(i),coordinates_angstrom=xyz)
        grounds.append(dict(**row(path),coordinates_angstrom=xyz.tolist(),minimum_curvature=ground.minimum_orbital_curvature_hartree))
    data['grounds']=grounds
    temporary=root/(folder.name+'.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(root/'accepted.json')


def restore_shared(component):
    Joint,save_joint,load_joint,schema=_coupled_io(component.backend)
    from matrix_gfn2.ground_referenced_roks import GroundReferencedROKSResult
    from matrix_gfn2.restricted_ground import RestrictedGroundResult
    from matrix_gfn2.deltascf_derivatives import load_derivative_point
    from matrix_gfn2 import combine_roks_results
    root=component.run_dir/'roks_50_50_accepted';manifest=root/'accepted.json'
    if not manifest.exists():return
    data=json.loads(manifest.read_text());contract=_contract(component.backend);refs=_models(component.backend)
    if data.get('schema')!=schema or data.get('contract')!=contract:raise ValueError('shared-pair restart belongs to another backend contract')
    def checked(row):
        path=root/row['path']
        if _sha(path)!=row['sha256']:raise ValueError('shared-pair checkpoint checksum mismatch')
        return path
    joint=load_joint(checked(data['reference']),request_hash=contract+':reference')
    _constraint(component.backend,joint)
    grounds=[]
    if len(data['grounds'])!=3:raise ValueError('incomplete shared-pair S0 restart')
    for i,row in enumerate(data['grounds']):
        raw=load_derivative_point(checked(row),request_hash=contract+':ground:'+str(i),coordinates_angstrom=np.asarray(row['coordinates_angstrom']))
        grounds.append(RestrictedGroundResult(raw,row['minimum_curvature']))
    total=GroundReferencedROKSResult(joint,tuple(grounds))
    if not total.spectrum_qualified:raise ValueError('shared restart has a nonpositive gap')
    anchor=data['anchor']
    if anchor['kind']=='shared':reference=load_joint(checked(anchor['checkpoint']),request_hash=contract+':anchor',require_gradient=False)
    elif anchor['kind']=='independent':
        states=[load_derivative_point(checked(row),request_hash=contract+':anchor:'+str(i),coordinates_angstrom=np.asarray(row['coordinates_angstrom']),require_gradient=False) for i,row in enumerate(anchor['checkpoints'])]
        reference=combine_roks_results(*states,component_tracking=component.backend.native_roks_composite_component_tracking or 'csf')
    else:raise ValueError('unknown shared-pair anchor representation')
    _relative_anchor(joint,reference)
    for group in [joint.components,reference.components]:
        for parent,ref in zip(group,refs):
            if parent.spin_reference_fingerprint!=ref.fingerprint:raise ValueError('restored joint state has another response model')
    component.reference=(total,_selected(total,component.backend.electronic_state));component.anchor=(reference,_selected(reference,component.backend.electronic_state))


def evaluate_shared(component,coordinates,compute_hessian,diagnostics):
    Joint,save_joint,load_joint,schema=_coupled_io(component.backend)
    from matrix_gfn2.shared_pair_roks import evaluate_shared_pair_roks
    from matrix_gfn2.relative_pair_roks import evaluate_relative_pair_roks
    from matrix_gfn2.ground_referenced_roks import GroundReferencedROKSResult,reference_roks_to_native_ground,ground_referenced_roks_analytic_hessian
    from matrix_gfn2.roks_tracking import active_orbital_character
    from matrix_apoc import singlet_csf_overlap,StateContinuityError
    from .scan import PointEvaluationResult
    if component.reference is None:raise ValueError('shared ROKS requires retained qualified parent states')
    backend=component.backend;refs=_models(backend);retained=component.reference[0]
    previous=retained.excited if isinstance(retained,GroundReferencedROKSResult) else retained
    options=dict(reference_result=previous) if isinstance(previous,Joint) else dict(initial_components=previous.components)
    hole=backend.native_roks_composite_orbital_coupling=='shared_active_hole'
    relative=backend.native_roks_composite_orbital_coupling=='relative_active_pair'
    if relative:
        if not isinstance(previous,Joint):options['anchor_components']=(component.anchor or component.reference)[0].components
        evaluate=evaluate_relative_pair_roks
    else:
        options['shared_orbitals']='hole' if hole else 'active_pair'
        if hole and not isinstance(previous,Joint):options['freeze_and_release']=False
        evaluate=evaluate_shared_pair_roks
    joint=evaluate(component.calculator._calculator,coordinates,refs,gradient=True,
        minimum_state_overlap=backend.state_tracking_minimum_overlap,**options)
    result=reference_roks_to_native_ground(component.calculator._calculator,coordinates,refs,excited=joint,
        reference_result=retained if isinstance(retained,GroundReferencedROKSResult) else None)
    if not result.spectrum_qualified:raise StateContinuityError('shared-pair ROKS has a nonpositive parent gap')
    selected=_selected(result,backend.electronic_state);scores={};characters=[]
    for pair,label in [(component.reference,'reference'),(component.anchor,'anchor')]:
        if pair is None:continue
        values=[]
        for old,new,old_apoc,new_apoc in zip(pair[0].components,result.components,pair[1],selected):
            cross=component._cross_overlap(old.reference_state,new.reference_state);score=float(singlet_csf_overlap(old_apoc,new_apoc,cross)[0]);values.append(score)
            if label=='reference' and score<backend.state_tracking_minimum_overlap:raise StateContinuityError('joint parent CSF continuity failed')
        scores[label]=values
    if component.anchor is not None:
        for old,new,ref in zip(component.anchor[0].components,result.components,refs):
            character=active_orbital_character(old,new,ref.anchor_coordinates_bohr);characters.append(character)
            if character['transition']<backend.state_tracking_minimum_overlap:raise StateContinuityError('shared-pair original active character is below threshold')
    hessian=None
    if compute_hessian:hessian=ground_referenced_roks_analytic_hessian(component.calculator._calculator,coordinates,refs,reference_result=result).hessian_hartree_per_bohr2
    return PointEvaluationResult(0,0.,energy_hartree=result.energy_hartree,
        gradient_hartree_per_bohr=result.gradient_hartree_per_bohr.ravel(),hessian_hartree_per_bohr2=hessian,
        backend_coordinates_angstrom=np.asarray(coordinates),scf_state=(result,selected),source='native shared-active ROKS 50/50',
        execution=dict(response_model='roks_50_50_relative_active_pair' if relative else ('roks_50_50_shared_active_hole' if hole else 'roks_50_50_shared_active_pair'),weights=[.5,.5],stationary_reference=diagnostics,
            selected_electronic_state=backend.electronic_state,gradient_electronic_state=backend.electronic_state,
            state_assignment_overlap=min(scores['reference']),state_anchor_overlap=min(scores['anchor']) if 'anchor' in scores else None,
            component_state_overlaps=scores,component_csf_overlap=result.component_csf_overlap,component_active_pair_overlap=joint.component_active_pair_overlap,
            **(dict(relative_active_target_squared_overlaps=joint.target_squared_overlaps.tolist(),relative_active_multipliers=joint.multipliers.tolist()) if relative else dict(shared_orbital_indices=list(joint.shared_orbital_indices))),
            anchor_active_orbital_character=characters,joint_orbital_saddle_index=joint.orbital_saddle_index,
            orbital_residual=joint.orbital_residual,component_unconstrained_residuals=list(joint.component_unconstrained_residuals),
            component_excitation_gaps_hartree=list(result.component_gaps_hartree),spin_squared=0.,spin_pure_singlet=True,
            hessian_source=('analytic_relative_pair_kkt_response_plus_ground_replacement' if relative else ('analytic_shared_hole_coupled_response_plus_ground_replacement' if hole else 'analytic_shared_pair_coupled_response_plus_ground_replacement')) if compute_hessian else None))
