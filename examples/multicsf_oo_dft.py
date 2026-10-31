"""Opt-in research OO-DFT from a prepared, checksummed multi-CSF model.

Run electronic work on the configured compute host. Model preparation currently
uses projected DFT response, bare ERIs and fixed reference transition weights.
The prepared model directory is portable; evaluation runs neither DFT nor TDA.
"""
from pathlib import Path
from matrix_gfn2 import MultiCSFModel
from matrix_link.scan import QMScanBackend
from matrix_link.qmqm_component import CappedQMComponent
from matrix_link.resident_calculators import create_resident_calculator


def prepared_component(model_manifest,run_directory,*,processors=2):
    model=MultiCSFModel(model_manifest)
    symbols=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in model.anchor.atomic_numbers]
    backend=QMScanBackend(name='matrix_gfn2',electronic_state=model.state_id,
        charge=model.anchor.charge,multiplicity=1,native_excited_method='multicsf_50_50',
        native_multicsf_model_path=Path(model_manifest))
    resident=create_resident_calculator('matrix_gfn2',symbols,model.xyz,
        charge=model.anchor.charge,multiplicity=1,processors=processors)
    return model,CappedQMComponent(backend,resident,Path(run_directory))


def anchor_energy_gradient_hessian(model_manifest,run_directory):
    model,component=prepared_component(model_manifest,run_directory)
    result=component.evaluate(model.xyz,compute_hessian=True)
    component.accept(result)  # Atomic publication; trial evaluations do not advance it.
    return result
