"""LINK adapter and atomic accepted-state restart for fixed raw/spectral ROKS."""
from pathlib import Path
import hashlib,json,uuid
import numpy as np


def _contract(backend):
    if backend.native_excited_method=='nosi_50_50':
        return hashlib.sha256(json.dumps(dict(method='nosi_50_50.v1',model_sha256=backend.native_nosi_model_sha256,state=backend.electronic_state,charge=backend.charge,multiplicity=backend.multiplicity,minimum_overlap=backend.state_tracking_minimum_overlap,ambiguity_margin=backend.state_tracking_ambiguity_margin),sort_keys=True).encode()).hexdigest()
    if backend.native_excited_method=='multicsf_50_50':
        return hashlib.sha256(json.dumps(dict(method='multicsf_50_50.v1',model_sha256=backend.native_multicsf_model_sha256,state=backend.electronic_state,charge=backend.charge,multiplicity=backend.multiplicity,minimum_overlap=backend.state_tracking_minimum_overlap),sort_keys=True).encode()).hexdigest()
    payload=dict(method='roks_50_50.v1',weights=[.5,.5],state=backend.electronic_state,
        promotion=backend.deltascf_promotion,indices=backend.native_roks_composite_saddle_indices,
        response_sha256=backend.native_roks_composite_response_sha256,
        charge=backend.charge,multiplicity=backend.multiplicity,
        minimum_overlap=backend.state_tracking_minimum_overlap)
    if backend.native_roks_composite_seed_paths is not None:
        payload['seed_sha256']=backend.native_roks_composite_seed_sha256
    if backend.native_roks_composite_following is not None:
        payload['following']=dict(policy=backend.native_roks_composite_following,metric='corotating_lowdin_singlet_active_pair.v1',
            maximum_index_change_from_seed=None if backend.native_roks_composite_following=='character_newton_index' else 1)
    if backend.native_roks_composite_ground_recovery is not None:
        payload['ground_reference_recovery']=backend.native_roks_composite_ground_recovery
    if backend.native_roks_composite_reference_policy is not None:
        payload['reference_policy']=backend.native_roks_composite_reference_policy
    if backend.native_roks_composite_component_tracking is not None:
        payload['component_tracking']=backend.native_roks_composite_component_tracking
    if backend.native_roks_composite_ground_surface is not None:
        payload['ground_surface']=backend.native_roks_composite_ground_surface
    if backend.native_roks_composite_orbital_coupling is not None:
        payload['orbital_coupling']=backend.native_roks_composite_orbital_coupling
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def _allowed_index(backend, index, component_index):
    initial=backend.native_roks_composite_saddle_indices[component_index]
    if backend.native_roks_composite_following=='character_newton_index':
        return index is not None and index>=0
    if backend.native_roks_composite_following in {'character_adaptive_index','local_csf_adaptive_index'}:
        return index is not None and abs(index-initial)<=1
    return index==initial


def _evaluate_components(component, coordinates, alpha, beta, refs, retained, ground):
    """One bounded adjacent-index retry, retaining the last accepted orbitals."""
    from matrix_gfn2 import combine_roks_results
    from matrix_gfn2.roks import evaluate_gfn2_roks_singlet,ROKSConvergenceError
    backend=component.backend;results=[];changes=[]
    adaptive=backend.native_roks_composite_following in {'character_adaptive_index','local_csf_adaptive_index'}
    for i,ref in enumerate(refs):
        old=None if retained is None else retained.components[i]
        index=backend.native_roks_composite_saddle_indices[i] if old is None else old.orbital_saddle_index
        options=dict(ground_state=ground,orbital_solver='gmf',saddle_index=index,maximum_iterations=150,
            orbital_tolerance=1e-10,minimum_state_overlap=backend.state_tracking_minimum_overlap)
        if backend.native_roks_composite_following=='character_newton_index':
            options.pop('saddle_index');options.pop('maximum_iterations')
            options.update(orbital_solver='newton',certify_index=True,maximum_evaluations=150,maximum_step=.2)
        if backend.native_roks_composite_reference_policy=='geometry_only':
            options.update(ground_state=None,reference_policy='geometry_only')
        def evaluate():
            return evaluate_gfn2_roks_singlet(component.calculator._calculator,coordinates,alpha,beta,
                spin_reference=ref,reference_result=old,gradient=True,**options)
        try:result=evaluate()
        except ROKSConvergenceError as exc:
            observed=exc.result.orbital_saddle_index
            if (not adaptive or old is None or observed is None or abs(observed-index)!=1
                    or not _allowed_index(backend,observed,i)):
                raise
            options['saddle_index']=observed
            result=evaluate()  # No recursive retry and no energy-based root selection.
            changes.append(dict(component=i,previous_index=index,new_index=observed,
                failed_residual=exc.result.orbital_residual))
        results.append(result)
        if backend.native_roks_composite_following=='character_newton_index':
            if not _allowed_index(backend,result.orbital_saddle_index,i):
                from matrix_apoc import StateContinuityError
                raise StateContinuityError('stationary ROKS branch lacks an analytic orbital-index certificate')
            if result.orbital_saddle_index!=index:
                changes.append(dict(component=i,previous_index=index,new_index=result.orbital_saddle_index,
                    reason='locally continued stationary root; analytic curvature certificate'))
    from matrix_gfn2.composite_roks import CompositeROKSStateMismatch
    from matrix_apoc import StateContinuityError
    try:
        combined=combine_roks_results(*results,minimum_component_overlap=backend.state_tracking_minimum_overlap,component_tracking=backend.native_roks_composite_component_tracking or 'csf')
    except CompositeROKSStateMismatch as exc:
        raise StateContinuityError(str(exc)) from exc
    return combined,changes


def _models(backend):
    from matrix_gfn2 import load_response_reference
    paths=backend.native_roks_composite_response_paths
    for path,expected in zip(paths,backend.native_roks_composite_response_sha256):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('ROKS 50/50 response changed during the calculation')
    return tuple(load_response_reference(p) for p in paths)


def _selected(result,state_id):return tuple(r.to_apoc(state_id) for r in result.components)


def restore_accepted(component):
    """Restore BOTH accepted and initial-anchor CSFs; never seed from a trial."""
    if component.backend.native_roks_composite_orbital_coupling is not None:
        from .shared_roks_component import restore_shared
        return restore_shared(component)
    from matrix_gfn2 import combine_roks_results
    from matrix_gfn2.deltascf_derivatives import load_derivative_point
    root=component.run_dir/'roks_50_50_accepted';manifest=root/'accepted.json'
    if not manifest.exists():return
    models=_models(component.backend)
    data=json.loads(manifest.read_text());contract=_contract(component.backend)
    if data.get('schema')!='mouse.link.roks_50_50.accepted.v1' or data.get('contract')!=contract:
        raise ValueError('ROKS 50/50 restart belongs to a different backend contract')
    pairs=[]
    for role in ('reference','anchor'):
        rows=data[role];results=[]
        if len(rows)!=2:raise ValueError('incomplete 50/50 accepted restart')
        for i,row in enumerate(rows):
            path=root/row['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('ROKS 50/50 accepted checkpoint checksum mismatch')
            result=load_derivative_point(path,request_hash=contract+':'+role+':'+str(i),coordinates_angstrom=np.array(row['coordinates_angstrom']),require_gradient=(role=='reference'))
            if result.spin_reference_fingerprint!=models[i].fingerprint:
                raise ValueError('ROKS 50/50 checkpoint and response model differ')
            if not _allowed_index(component.backend,result.orbital_saddle_index,i):
                raise ValueError('ROKS 50/50 checkpoint and saddle index differ')
            results.append(result)
        result=combine_roks_results(*results,minimum_component_overlap=component.backend.state_tracking_minimum_overlap,component_tracking=component.backend.native_roks_composite_component_tracking or 'csf')
        if role=='reference' and component.backend.native_roks_composite_ground_surface=='native_plus_gaps':
            from matrix_gfn2.restricted_ground import RestrictedGroundResult
            from matrix_gfn2.ground_referenced_roks import GroundReferencedROKSResult
            grounds=[]
            for i,row in enumerate(data.get('ground_references',[])):
                path=root/row['path']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                    raise ValueError('ground-reference checkpoint checksum mismatch')
                ground=load_derivative_point(path,request_hash=contract+':ground:'+str(i),
                    coordinates_angstrom=np.array(row['coordinates_angstrom']))
                grounds.append(RestrictedGroundResult(ground,row['minimum_curvature_hartree']))
            result=GroundReferencedROKSResult(result,tuple(grounds))
            if not result.spectrum_qualified:
                raise ValueError('ground-referenced restart has a nonpositive excitation gap')
        pairs.append((result,_selected(result,component.backend.electronic_state)))
    component.reference,component.anchor=pairs


def persist_accepted(component,scf_state):
    """Write a generation completely before atomically replacing the pointer."""
    if component.backend.native_roks_composite_orbital_coupling is not None:
        from .shared_roks_component import persist_shared
        return persist_shared(component,scf_state)
    from matrix_gfn2.deltascf_derivatives import save_derivative_point
    from matrix_gfn2 import CompositeROKSResult,combine_roks_results
    from matrix_gfn2.ground_referenced_roks import GroundReferencedROKSResult
    models=_models(component.backend);contract=_contract(component.backend)
    result=scf_state[0]
    if not isinstance(result,(CompositeROKSResult,GroundReferencedROKSResult)):raise TypeError('accepted 50/50 state must contain both ROKS solutions')
    replaced=component.backend.native_roks_composite_ground_surface=='native_plus_gaps'
    if replaced and (not isinstance(result,GroundReferencedROKSResult) or not result.spectrum_qualified):
        raise ValueError('cannot accept an unqualified ground-referenced spectrum')
    if not replaced and isinstance(result,GroundReferencedROKSResult):
        raise ValueError('ground-referenced energy requires its explicit backend contract')
    combine_roks_results(*result.components,minimum_component_overlap=component.backend.state_tracking_minimum_overlap,component_tracking=component.backend.native_roks_composite_component_tracking or 'csf')
    for i,r in enumerate(result.components):
        if r.spin_reference_fingerprint!=models[i].fingerprint:
            raise ValueError('accepted 50/50 state belongs to another response model')
        if not _allowed_index(component.backend,r.orbital_saddle_index,i):
            raise ValueError('accepted 50/50 state has another orbital saddle index')
    root=component.run_dir/'roks_50_50_accepted';generation=uuid.uuid4().hex
    directory=root/generation;directory.mkdir(parents=True)
    data=dict(schema='mouse.link.roks_50_50.accepted.v1',contract=contract)
    for role,pair in [('reference',scf_state),('anchor',component.anchor or scf_state)]:
        rows=[]
        for i,result in enumerate(pair[0].components):
            path=directory/f'{role}_{i}.npz'
            # Use the same canonical conversion as the native checkpoint consumer.
            from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
            xyz=result.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR
            save_derivative_point(path,result,request_hash=contract+':'+role+':'+str(i),coordinates_angstrom=xyz)
            rows.append(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),coordinates_angstrom=xyz.tolist()))
        data[role]=rows
    if replaced:
        rows=[]
        for i,ground in enumerate(scf_state[0].grounds):
            path=directory/f'ground_{i}.npz'
            xyz=ground.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR
            save_derivative_point(path,ground.reference,request_hash=contract+':ground:'+str(i),coordinates_angstrom=xyz)
            rows.append(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                coordinates_angstrom=xyz.tolist(),minimum_curvature_hartree=ground.minimum_orbital_curvature_hartree))
        data['ground_references']=rows
    tmp=root/f'accepted.{generation}.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(root/'accepted.json')


def _evaluate_composite_component(component,coordinates,compute_hessian,ground,diagnostics):
    if component.backend.native_roks_composite_orbital_coupling is not None:
        from .shared_roks_component import evaluate_shared
        return evaluate_shared(component,coordinates,compute_hessian,diagnostics)
    from matrix_gfn2 import evaluate_composite_roks_analytic_hessian
    from matrix_apoc import singlet_csf_overlap,StateContinuityError
    from .scan import PointEvaluationResult
    backend=component.backend;refs=_models(backend);state=None if ground is None else ground.orbital_state
    retained=None if component.reference is None else component.reference[0]
    if retained is None:
        if state is None:raise ValueError('geometry-only LINK requires an explicit retained excited-state checkpoint')
        alpha,beta=state.alpha_occupations.copy(),state.beta_occupations.copy();hole,particle=backend.deltascf_promotion
        if max(hole,particle)>=len(alpha) or alpha[hole]!=1 or alpha[particle]!=0:
            raise ValueError('requested 50/50 promotion is not occupied -> virtual')
        alpha[hole],alpha[particle]=0.,1.
    else:alpha,beta=retained.components[0].occupations
    # The two Hamiltonians remain independently stationary and independently tracked.
    calculators=(component.calculator._calculator,)*2
    result,index_changes=_evaluate_components(component,coordinates,alpha,beta,refs,retained,ground)
    if backend.native_roks_composite_ground_surface=='native_plus_gaps':
        from matrix_gfn2.ground_referenced_roks import (GroundReferencedROKSResult,
            reference_roks_to_native_ground)
        result=reference_roks_to_native_ground(component.calculator._calculator,coordinates,refs,
            excited=result,reference_result=retained if isinstance(retained,GroundReferencedROKSResult) else None)
        if not result.spectrum_qualified:
            raise StateContinuityError('ground-referenced ROKS has a nonpositive component excitation gap')
    selected=_selected(result,backend.electronic_state);scores={};characters=[]
    for pair,label in [(component.reference,'reference'),(component.anchor,'anchor')]:
        if pair is None:continue
        values=[]
        for old,new,old_apoc,new_apoc in zip(pair[0].components,result.components,pair[1],selected):
            cross=component._cross_overlap(old.reference_state,new.reference_state)
            score=float(singlet_csf_overlap(old_apoc,new_apoc,cross)[0]);values.append(score)
            if score<backend.state_tracking_minimum_overlap and (label=='reference' or backend.native_roks_composite_following is None):
                raise StateContinuityError(f'ROKS 50/50 {label} CSF overlap {score:.6f} below threshold')
        scores[label]=values
    if component.anchor is not None and backend.native_roks_composite_following is not None:
        from matrix_gfn2.roks_tracking import active_orbital_character
        for old,new,ref in zip(component.anchor[0].components,result.components,refs):
            character=active_orbital_character(old,new,ref.anchor_coordinates_bohr)
            characters.append(character)
            if character['transition']<backend.state_tracking_minimum_overlap and not backend.native_roks_composite_following.startswith('local_'):
                raise StateContinuityError(f"ROKS 50/50 anchor active-orbital character {character['transition']:.6f} below threshold")
    hessian=None
    if compute_hessian:
        if backend.native_roks_composite_ground_surface=='native_plus_gaps':
            from matrix_gfn2.ground_referenced_roks import ground_referenced_roks_analytic_hessian
            hessian=ground_referenced_roks_analytic_hessian(calculators[0],coordinates,refs,reference_result=result).hessian_hartree_per_bohr2
        else:
            hessian=evaluate_composite_roks_analytic_hessian(calculators,coordinates,refs,reference_result=result).hessian_hartree_per_bohr2
    return PointEvaluationResult(0,0.,energy_hartree=result.energy_hartree,
        gradient_hartree_per_bohr=result.gradient_hartree_per_bohr.reshape(-1),hessian_hartree_per_bohr2=hessian,
        backend_coordinates_angstrom=np.asarray(coordinates),scf_state=(result,selected),source='native ROKS 50/50',
        execution=dict(response_model='roks_50_50',weights=[.5,.5],stationary_reference=diagnostics,
            selected_electronic_state=backend.electronic_state,gradient_electronic_state=backend.electronic_state,
            state_assignment_overlap=min(scores['reference']) if 'reference' in scores else None,
            state_anchor_overlap=min(scores['anchor']) if 'anchor' in scores else None,
            component_state_overlaps=scores,component_csf_overlap=result.component_csf_overlap,
            component_active_pair_overlap=result.component_active_pair_overlap,component_tracking=result.component_tracking,
            ground_surface=backend.native_roks_composite_ground_surface or 'mean_absolute_corrected_surfaces',
            component_excitation_gaps_hartree=list(result.component_gaps_hartree) if backend.native_roks_composite_ground_surface else None,
            reference_policy=backend.native_roks_composite_reference_policy or 'stationary_ground',
            state_following_contract=backend.native_roks_composite_following or 'separate_apoc_cross_ao_csf',
            anchor_active_orbital_character=characters,orbital_index_transitions=index_changes,spin_squared=0.,spin_pure_singlet=True,
            orbital_residual=max(r.orbital_residual for r in result.components),
            component_orbital_saddle_indices=list(result.orbital_saddle_indices),
            no_offset_response_fingerprints=[r.fingerprint for r in refs],
            hessian_source=('analytic_native_ground_plus_mean_roks_gaps' if backend.native_roks_composite_ground_surface
                else 'mean_of_relaxed_analytic_nuclear_hessians') if compute_hessian else None))


def evaluate_composite_component(component,coordinates,compute_hessian,ground,diagnostics):
    """Preserve rejected trial diagnostics without advancing accepted states."""
    try:
        return _evaluate_composite_component(component,coordinates,compute_hessian,ground,diagnostics)
    except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
        root=component.run_dir/'rejected_trials'
        try:
            root.mkdir(parents=True,exist_ok=True)
            row=dict(error_type=type(exc).__name__,error=str(exc),coordinates_angstrom=np.asarray(coordinates).tolist())
            failed=getattr(exc,'result',None)
            if failed is not None:
                row.update(orbital_residual=failed.orbital_residual,observed_index=failed.orbital_saddle_index,target_index=failed.target_orbital_saddle_index)
            (root/(uuid.uuid4().hex+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        except OSError:
            pass  # A diagnostic failure must not replace the original error.
        raise


def initialize_from_seeds(component):
    """Read two certified parent CSFs only when no accepted generation exists."""
    backend=component.backend
    if component.reference is not None or backend.native_roks_composite_seed_paths is None:return
    from matrix_gfn2 import combine_roks_results
    from matrix_gfn2.roks import ROKSSingletResult
    from matrix_gfn2.deltascf_derivatives import load_derivative_point
    from matrix_gfn2.spin_reference import validate_spin_reference
    refs=_models(backend);parents=[]
    for path,checksum,ref,index in zip(backend.native_roks_composite_seed_paths,backend.native_roks_composite_seed_sha256,refs,backend.native_roks_composite_saddle_indices):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:raise ValueError('ROKS seed changed after backend construction')
        with np.load(path,allow_pickle=False) as data:
            request=str(data['request_hash']);xyz=np.array(data['coordinates_angstrom'],copy=True)
        parent=load_derivative_point(path,request_hash=request,coordinates_angstrom=xyz,require_gradient=False)
        if not isinstance(parent,ROKSSingletResult) or not parent.converged or parent.orbital_residual>1e-8 or parent.orbital_saddle_index!=index:
            raise ValueError('ROKS seed is not a certified parent of the configured saddle index')
        validate_spin_reference(parent,ref)
        hole,particle=backend.deltascf_promotion
        alpha=parent.reference_state.alpha_occupations.copy();beta=parent.reference_state.beta_occupations
        if max(hole,particle)>=len(alpha) or alpha[hole]!=1 or alpha[particle]!=0:raise ValueError('ROKS seed promotion differs from the backend')
        alpha[hole],alpha[particle]=0.,1.
        if not np.array_equal(parent.occupations,np.array([alpha,beta])):raise ValueError('ROKS seed occupations differ from the configured promotion')
        parents.append(parent)
    combined=combine_roks_results(*parents,minimum_component_overlap=backend.state_tracking_minimum_overlap,
        component_tracking=backend.native_roks_composite_component_tracking or 'csf')
    component.reference=(combined,_selected(combined,backend.electronic_state));component.anchor=component.reference
