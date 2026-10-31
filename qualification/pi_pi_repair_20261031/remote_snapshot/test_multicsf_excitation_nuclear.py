"""Full ground-restored multi-CSF excited PES: two analytic Hessian columns.
Reuses all completed parent and ground checkpoints. Oracle only, two threads.
"""
from parent_geometry_model import ParentGeometryModel,V
import socket,json,time,fcntl,hashlib
assert socket.getfqdn()=='oracle.sns.it'
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2.restricted_ground import RestrictedGroundResult,evaluate_restricted_ground,restricted_ground_analytic_hessian
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2 import GFN2Calculator
from multicsf_frame_response import RelativeActiveFrame
from multicsf_nuclear_response import parent_gradient

with threadpool_limits(limits=2):
    lock=(V.parent/'MOUSE-publication-validation-20261030/blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    models=[ParentGeometryModel('Furan','B022',i,str(i)) for i in range(2)];xyz=models[0].xyz;out=V/'pi_pi_repair/nuclear_validation';refs=[None]+[m.ref for m in models];weights=[1.,-.5,-.5];grounds=[];provenance=[];ground_h=[];ground_g=[];start=time.monotonic()
    calc=GFN2Calculator(models[0].symbols,temperature_kelvin=0.)
    def gradient(g,ref,x):
        ctx=prepare_gfn2_deltascf(calc,x,orbital_reference=models[0].anchor,spin_reference=ref);c=g.coefficients[0];f,_=ctx.fock_and_energy(g.spin_densities_ao)
        q=sum((c.T@ff@c)*o[None,:] for ff,o in zip(f,g.occupations));return ctx.analytic_gradient(g.spin_densities_ao,c@(.5*(q+q.T))@c.T)
    for i,ref in enumerate(refs):
        from pathlib import Path
        roots=[V,V.parent/'MOUSE-state-following-20261031',V.parent/'MOUSE-publication-validation-20261030/blind/mouse']
        candidates=[p for root in roots for p in root.glob(f'**/Furan/**/grounds/{i}/certificate.json')]
        raw=None
        for certificate in candidates:
            cp=certificate.parent/'stationary.npz'
            if not cp.exists():continue
            cert=json.loads(certificate.read_text())
            with np.load(cp) as a:req=str(a['request_hash']);gx=a['coordinates_angstrom'].copy()
            candidate=load_derivative_point(cp,request_hash=req,coordinates_angstrom=gx,require_gradient=False)
            if candidate.spin_reference_fingerprint==(None if ref is None else ref.fingerprint):raw=candidate;break
        if raw is None:raise RuntimeError(f'No matching completed ground checkpoint for profile {i}')
        g=RestrictedGroundResult(raw,cert['minimum_curvature'],cert['index'])
        assert np.max(abs(gx-xyz))<1e-10;grounds.append(g);provenance.append(dict(source=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
        cache=out/f'Furan_ground_{i}_hessian.npz'
        if cache.exists():
            with np.load(cache) as a:h=a['hessian'].copy()
        else:
            result=restricted_ground_analytic_hessian(calc,xyz,reference_result=g,spin_reference=ref);h=result.hessian_hartree_per_bohr2;np.savez_compressed(cache,hessian=h)
        ground_h.append(h);ground_g.append(gradient(g,ref,xyz))
    with np.load(out/'Furan_frame_parent_mean_analytic.npz') as a:hh=a['hessian'].copy();gg=a['gradient'].copy();energy=float(a['energy'])
    for w,h,g,r in zip(weights,ground_h,ground_g,grounds):hh+=w*h;gg+=w*g;energy+=w*r.energy_hartree
    np.savez_compressed(out/'Furan_excited_total_analytic.npz',energy=energy,gradient=gg,hessian=hh)
    report=dict(host=socket.getfqdn(),scope='full excited total energy: native S0 plus equally weighted multi-CSF parent gaps; fixed full active cross-overlap frame; not original single-pair PES',ground_checkpoint_provenance=provenance,analytic_seconds=time.monotonic()-start,checks=[])
    for coordinate in [0,8]:
        step=2e-4;values=[];gradients=[]
        for sign in [-1,1]:
            x=xyz.copy();x.ravel()[coordinate]+=sign*step/1.8897261254578281
            pairs=[m.objective(x) for m in models];oo=[p[0] for p in pairs];cc=RelativeActiveFrame(oo).polish([p[1] for p in pairs]);e=sum(.5*o.evaluate(c,False)[0] for o,c in zip(oo,cc));grad=sum(.5*parent_gradient(o,c,m.eri) for o,c,m in zip(oo,cc,models))
            for w,ref,seed in zip(weights,refs,grounds):
                g=evaluate_restricted_ground(calc,x,template=models[0].anchor,spin_reference=ref,reference_result=seed,gradient=True);e+=w*g.energy_hartree;grad+=w*g.gradient_hartree_per_bohr
            values.append(e);gradients.append(grad.ravel())
        row=dict(coordinate=coordinate,energy_gradient_error=abs((values[1]-values[0])/(2*step)-gg.ravel()[coordinate]),hessian_column_max_error=float(np.max(abs((gradients[1]-gradients[0])/(2*step)-hh[:,coordinate]))));report['checks'].append(row)
        (out/'Furan_excited_total_report.json').write_text(json.dumps(report,indent=2)+'\n');print(row,flush=True)
        assert row['energy_gradient_error']<2e-6 and row['hessian_column_max_error']<2e-5
    print(json.dumps(report,indent=2),flush=True)
