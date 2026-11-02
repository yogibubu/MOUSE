"""Experimental energy-only zero-frequency subtraction on saved pigment pencils."""
from run_pigment_cid import *
from matrix_gfn2._nosi_static_subtraction import static_subtraction
from matrix_gfn2._nosi_selected_roots import solve_full
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--frontier',type=int,required=True);p.add_argument('--screening-frontier',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--eigensolver',choices=['arpack','lobpcg'],default='arpack');a=p.parse_args()
folder=HERE/f'pigment_cid_transition/{a.name}/screening_{a.screening_frontier}/frontier_{a.frontier}'
with threadpool_limits(4):
 start=time.monotonic();report=json.loads((folder/'result.json').read_text());nbase=1+2*len(report['retained_ROKS_ordinals'])
 with np.load(folder/'matrices.npz') as z:h=z['h'];s=z['s'];eg=float(z['ground_energy'])
 projection=s[0,1:]/s[0,0];sp=s[1:,1:]-np.outer(s[1:,0],projection)
 hp=h[1:,1:]-np.outer(h[1:,0],projection)-np.outer(projection,h[0,1:])+h[0,0]*np.outer(projection,projection)
 del h,s
 hp-=eg*sp
 print('PRIMARY',nbase-1,'COMPLEMENT',len(hp)-nbase+1,flush=True)
 r,delta,certificate=static_subtraction(hp,sp,np.arange(nbase-1))
 np.savez_compressed(HERE/a.output.replace('.json','_counterterm.npz'),r=r,delta=delta,certificate=json.dumps(certificate))
 print('SUBTRACTION',certificate,flush=True)
 hp+=(r@delta)@r.T
 e,c,eigen_certificate=solve_full(hp,sp,8,energy_shift=0.,algorithm=a.eigensolver)
 result=dict(molecule=a.name,frontier=a.frontier,screening_frontier=a.screening_frontier,states=[dict(root=i,excitation_ev=float(value*27.211386245988)) for i,value in enumerate(e,1)],subtraction_certificate=certificate,eigensolver_certificate=eigen_certificate,wall_seconds=time.monotonic()-start,scope='Experimental energy-only static-subtraction diagnostic; no oscillator-strength or nuclear-derivative qualification; not promoted to production')
 np.savez_compressed(HERE/a.output.replace('.json','.npz'),energies=e,ci=c,primary_counterterm=delta);write(HERE/a.output,result);print(json.dumps(result,indent=2),flush=True)
