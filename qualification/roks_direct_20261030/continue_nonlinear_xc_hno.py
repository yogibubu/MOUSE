"""Repair the electronic identity through a fixed nonlinear-XC homotopy.
No energy- or TBE-based state selection. Adjacent index changes only.
"""
import importlib.util,json
from pathlib import Path
from dataclasses import replace
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030')
spec=importlib.util.spec_from_file_location('nonlinear',BASE/'source/qualification/probe_nonlinear_xc.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=m.p;np=m.np

def csf_overlap(a,b,occ,s):
 return abs(.5*sum(np.prod([np.linalg.det(a[:,oa>.5].T@s@b[:,ob>.5]) for oa,ob in zip(o1,o2)]) for o1 in [occ,occ[::-1]] for o2 in [occ,occ[::-1]]))

with p.threadpool_limits(limits=2):
 root=BASE/'results/nitroxyl/nonlinear_xc';out=root/'continuation';out.mkdir(exist_ok=True)
 if (out/'result.json').exists():raise SystemExit(0)
 cp=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030/results/nitroxyl/full_response/seed.npz')
 with np.load(cp) as d:xyz=d['coordinates_angstrom'].copy()
 old=m.load_derivative_point(cp,request_hash='retained_vertical',coordinates_angstrom=xyz,require_gradient=False);s=old.reference_state
 data=json.loads((BASE/'results/nitroxyl/prepared.json').read_text());ref=p.load_response_reference(data['response_paths'][0])
 folder=Path('/home/enzo/MATRIX-runs/tda-dft-independent-20261022-v1/docs/reports/independent_dft_tda_20261022/results/nitroxyl_pbe0')
 with np.load('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel/nitroxyl/kernel.npz') as d:lift=d['embedding'].copy()
 with np.load(folder/'reference.npz') as d:c=d['coefficients'];o=d['occupations']
 core=c[:,np.flatnonzero(o==2)[:int(np.sum(old.core_electrons)//2)]];mol=m.scf.chkfile.load_mol(str(folder/'backend.chk'))
 residual=m.XCResidual(mol,lift,core,s,ref,root/'grid_features.npz')
 calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.);ctx=p.prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=ref)
 previous=old.common_coefficients;index=old.orbital_saddle_index;rows=[]
 for gamma in np.linspace(.05,1.,20):
  residual.strength=float(gamma);attempts=[]
  for target in [index]+[j for j in [0,1] if j!=index]:
   checkpoint=out/f'gamma{gamma:.2f}_index{target}.npz'
   try:
    result,cache=m.solve(ctx,residual,previous,old.occupations,target,checkpoint)
    descriptor=replace(old,common_coefficients=result.point)
    pair=m.active_orbital_character(old,descriptor,s.coordinates_bohr)['transition']
    overlap=csf_overlap(previous,result.point,old.occupations,s.overlap)
    row=dict(gamma=float(gamma),index=target,original_pair=pair,previous_csf=overlap,derivative_checks=cache['derivative_checks'])
    if pair<.7 or overlap<.7:raise ValueError('identity rejected: '+str(row))
    previous=result.point;index=target;rows.append(row);print(json.dumps(row),flush=True);break
   except Exception as exc:attempts.append(dict(index=target,error=str(exc)))
  else:
   (out/'rejected.json').write_text(json.dumps(dict(gamma=float(gamma),attempts=attempts),indent=2)+'\n');raise RuntimeError('no root conserves identity at gamma '+str(gamma))
  (out/'trace.json').write_text(json.dumps(rows,indent=2)+'\n')
 with np.load(root/'ground_0_latest.npz') as d:ground=float(d['energy'])
 final=m.ROKSSingletResult(result.point,old.occupations,old.triplet_occupations,s.overlap,cache['values'][0][1],cache['values'][1][1],result.energy,float(np.max(abs(result.gradient))),result.iterations,True,ground_energy_hartree=ground,reference_state=s,core_electrons=old.core_electrons,orbital_saddle_index=result.observed_index,target_orbital_saddle_index=result.target_index,spin_reference_fingerprint='nonlinear_xc_vertical:'+ref.fingerprint)
 m.save_derivative_point(out/'excited_full.npz',final,request_hash='nonlinear_xc_hno_continuation_v1',coordinates_angstrom=xyz)
 original=json.loads((root/'result.json').read_text());spectral=original['rows'][1]['excitation_ev']
 summary=dict(molecule='nitroxyl',full_ev=final.excitation_energy_hartree*27.211386245988,spectral_ev=spectral,index=index,original_pair=rows[-1]['original_pair'],mean_excitation_ev=.5*(final.excitation_energy_hartree*27.211386245988+spectral),qualified=rows[-1]['original_pair']>=.7 and original['rows'][1]['initial_pair_overlap']>=.7,nuclear_derivatives=False)
 (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print('COMPLETE',json.dumps(summary),flush=True)
