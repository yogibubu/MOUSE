"""Remote, checkpointed shared-pair vertical control on assigned parent states.

Reports every attempted state. Reference energies are read only after solving;
they never select a root. Ground and TDA checkpoints are reused unchanged.
"""
import os, socket, sys, json, time, hashlib, traceback
from pathlib import Path
from dataclasses import replace

BASE = Path('/home/enzo/MOUSE-roks-shared-hole-20261030')
OLD = Path('/home/enzo/MOUSE-roks-independent-20261030')
OUT = Path('/home/enzo/MOUSE-roks-shared-hole-20261030/vertical')
assert socket.getfqdn() == 'oracle.sns.it'
os.environ['MOUSE_VALIDATION_ROOT'] = str(BASE)
sys.path.insert(0, str(BASE/'source/qualification'))
import remote_bootstrap
for path in (BASE/'source/packages').glob('*/src'):
    sys.path.insert(0, str(path))
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator, FrozenNonlocalResponseReference
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2.shared_pair_roks import evaluate_shared_pair_roks, save_shared_pair_roks
from matrix_gfn2.roks_tracking import active_orbital_character


def run(name, state_id):
    root = OUT/name/state_id
    root.mkdir(parents=True, exist_ok=True)
    final = root/'result.json'
    if final.exists():
        return
    parents, responses, reports, tda = [], [], [], []
    for profile in ['full_response', 'qualified_spectral_response']:
        source = OLD/'results'/profile/name
        report = json.loads((source/state_id/'result.json').read_text())
        if not report['qualified']:
            raise ValueError('unqualified archived parent')
        with np.load(report['checkpoint']) as data:
            xyz = data['coordinates_angstrom'].copy()
        parent = load_derivative_point(report['checkpoint'], request_hash=report['request_hash'],
                                       coordinates_angstrom=xyz, require_gradient=False)
        state = parent.reference_state
        atoms = [{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in state.atomic_numbers]
        calc = GFN2Calculator(atoms, temperature_kelvin=0.)
        ctx = prepare_gfn2_deltascf(calc, xyz, orbital_reference=state)
        modelpath = source/'model.npz'
        with np.load(modelpath) as data:
            model = {k:data[k].copy() for k in ['operators','charge','spin','q0','linear']}
        response = FrozenNonlocalResponseReference.from_vertical_model(state, ctx.kernel.partition,
            model, hashlib.sha256(modelpath.read_bytes()).hexdigest())
        parents.append(replace(parent, spin_reference_fingerprint=response.fingerprint))
        responses.append(response)
        reports.append(report)
        tdafile = source/'tda.json'
        tda.append(next((r for r in json.loads(tdafile.read_text())['rows']
                         if r['state'] == state_id), None) if tdafile.exists() else None)
    calc.evaluate = lambda *args, **kw: (_ for _ in ()).throw(AssertionError('SCC invoked'))
    def checkpoint(data):
        path = root/'latest.tmp'
        with path.open('wb') as stream:
            np.savez_compressed(stream, **data)
        path.replace(root/'latest.npz')
    initial = parents
    prototype = Path('/home/enzo/MOUSE-roks-shared-pair-20261030/results/uracil/shared_hole/latest.npz')
    if name=='uracil' and state_id=='n_pi' and prototype.exists():
        with np.load(prototype) as data: coefficients=data['coefficients'].copy()
        initial=[replace(p,common_coefficients=c,converged=False,orbital_saddle_index=None,target_orbital_saddle_index=None) for p,c in zip(parents,coefficients)]
    start = time.perf_counter()
    result = evaluate_shared_pair_roks(calc, xyz, responses, initial_components=initial,shared_orbitals='hole',
        freeze_and_release=False, minimum_state_overlap=0., maximum_iterations=150,
        checkpoint=checkpoint)
    seconds = time.perf_counter()-start
    save_shared_pair_roks(root/'stationary.npz', result, request_hash='shared_hole_vertical_v1')
    characters = [active_orbital_character(old, new, ref.anchor_coordinates_bohr)
                  for old, new, ref in zip(parents, result.components, responses)]
    gap = (result.energy_hartree-.5*sum(p.ground_energy_hartree for p in parents))*27.211386245988
    report = dict(host=socket.getfqdn(), molecule=name, state=state_id,
        shared_orbitals='hole',excitation_ev=gap, source_characters=characters,
        qualified=all(c['transition']>=.7 for c in characters),
        joint_residual=result.orbital_residual, joint_index=result.orbital_saddle_index,
        minimum_absolute_curvature=result.minimum_absolute_orbital_curvature_hartree,
        seconds=seconds, weights=[.5,.5], lambda_exchange=.9, energy_offset=0.,
        independent_mean_ev=.5*sum(r['excitation_ev'] for r in reports),
        parent_tda_gfn2=tda, scope='exposed vertical regression; stationary roots selected by continuity')
    # Reporting only: reference data cannot affect solver or qualification.
    report['tbe_ev'] = reports[0].get('tbe_ev')
    final.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    cases = [(n,'n_pi') for n in ['nitroxyl','acetaldehyde','acetone','formamide',
                                 'glyoxal','pyrazine','tetrazine','acrolein','nitrosomethane','pyrimidine','cyclopentadienone','maleimide','pyranone']]
    cases += [('uracil','n_pi'),('uracil','pi_pi')]
    with threadpool_limits(limits=2):
        for name, state in cases:
            try:
                run(name,state)
            except Exception as exc:
                print('UNRESOLVED',name,state,repr(exc),flush=True)
                traceback.print_exc()
