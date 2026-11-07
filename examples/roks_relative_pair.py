"""Oracle example: automatic pure-singlet ROKS 50/50, gradient and exact Hessian.

Provide the two calibrated response files and their two assigned vertical CSF
checkpoints. Subsequent invocations resume the accepted generation and geometry.
"""
import argparse
import json
import socket
from pathlib import Path
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['raw-response','spectral-response','raw-seed','spectral-seed','run-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--coordinates-json',type=Path,help='Optional new N-by-3 geometry in angstrom')
    parser.add_argument('--hessian',action='store_true')
    args=parser.parse_args()
    host=socket.getfqdn()
    if host!='oracle.sns.it':
        raise RuntimeError(f'This electronic example is restricted to verified oracle.sns.it; current host: {host}')
    import numpy as np
    from threadpoolctl import threadpool_limits
    from matrix_gfn2 import GFN2Calculator
    from matrix_gfn2.calculator import ANGSTROM_TO_BOHR,ELEMENT_SYMBOLS
    from matrix_gfn2.deltascf_derivatives import load_derivative_point
    from matrix_link.scan import QMScanBackend
    from matrix_link.qmqm_component import CappedQMComponent
    parents=[]
    for path in [args.raw_seed,args.spectral_seed]:
        with np.load(path,allow_pickle=False) as data:
            request=str(data['request_hash']);xyz=data['coordinates_angstrom'].copy()
        parents.append(load_derivative_point(path,request_hash=request,coordinates_angstrom=xyz,require_gradient=False))
    occ=parents[0].occupations;hole=np.flatnonzero(occ[0]<occ[1]);particle=np.flatnonzero(occ[0]>occ[1])
    if len(hole)!=1 or len(particle)!=1:raise ValueError('provide a single assigned singlet excitation')
    state=parents[0].reference_state
    atoms=[ELEMENT_SYMBOLS[int(z)-1] for z in state.atomic_numbers]
    calc=GFN2Calculator(atoms,charge=state.charge,temperature_kelvin=0.)
    backend=QMScanBackend(name='matrix_gfn2',charge=state.charge,electronic_state=1,
        native_excited_method='roks_50_50',deltascf_promotion=(int(hole[0]),int(particle[0])),
        native_roks_composite_response_paths=(args.raw_response,args.spectral_response),
        native_roks_composite_seed_paths=(args.raw_seed,args.spectral_seed),
        native_roks_composite_saddle_indices=tuple(p.orbital_saddle_index for p in parents),
        native_roks_composite_reference_policy='geometry_only',
        native_roks_composite_ground_surface='native_plus_gaps',
        native_roks_composite_component_tracking='active_pair',
        native_roks_composite_following='character_newton_index',
        native_roks_composite_orbital_coupling='relative_active_pair')
    restarted=(args.run_dir/'roks_50_50_accepted/accepted.json').exists()
    with threadpool_limits(limits=2):
        component=CappedQMComponent(backend,SimpleNamespace(_calculator=calc),args.run_dir)
        xyz=component.reference[0].components[0].reference_state.coordinates_bohr/ANGSTROM_TO_BOHR
        if args.coordinates_json is not None:xyz=np.asarray(json.loads(args.coordinates_json.read_text()),dtype=float)
        point=component.evaluate(xyz,compute_hessian=args.hessian)
        component.accept(point)
    destination=args.run_dir/'point.npz'
    arrays=dict(coordinates_angstrom=xyz,energy_hartree=point.energy_hartree,
                gradient_hartree_per_bohr=point.gradient_hartree_per_bohr)
    if args.hessian:arrays['hessian_hartree_per_bohr2']=point.hessian_hartree_per_bohr2
    np.savez_compressed(destination,**arrays)
    print(json.dumps(dict(host=host,restarted=restarted,energy_hartree=point.energy_hartree,
        maximum_gradient_hartree_per_bohr=float(np.max(abs(point.gradient_hartree_per_bohr))),
        joint_residual=point.execution['orbital_residual'],hessian_source=point.execution['hessian_source'],
        result=str(destination.resolve())),indent=2))


if __name__=='__main__':main()
