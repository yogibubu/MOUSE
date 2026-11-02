"""Diagnostic: correlate ground and excited states in the identical saved pencil."""
from run_pigment_cid import *
from matrix_gfn2._nosi_selected_roots import solve_full
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--frontier',type=int,required=True);p.add_argument('--screening-frontier',type=int,required=True);p.add_argument('--screening-mode',default='coherent_transitions');p.add_argument('--output',required=True);a=p.parse_args()
folder=HERE/f'pigment_cid_transition/{a.name}/screening_{a.screening_frontier}/frontier_{a.frontier}' if a.screening_mode=='coherent_transitions' else HERE/f'pigment_cid_expansion/{a.name}/screening_{a.screening_frontier}/frontier_{a.frontier}'
with threadpool_limits(4):
 start=time.monotonic()
 with np.load(folder/'matrices.npz') as z:h=z['h'];s=z['s'];native=float(z['ground_energy'])
 e,c,certificate=solve_full(h,s,8);result=dict(molecule=a.name,frontier=a.frontier,screening_frontier=a.screening_frontier,screening_mode=a.screening_mode,states=[dict(root=i,excitation_ev=float((value-e[0])*27.211386245988)) for i,value in enumerate(e[1:],1)],computed_ground_correlation_ev=float((e[0]-native)*27.211386245988),native_ground_weight=float((s[0]@c[:,0])**2/s[0,0]),certificate=certificate,wall_seconds=time.monotonic()-start,scope='Balanced full-CI energy diagnostic only; transition moments and nuclear derivatives not yet qualified; no empirical energy offset')
 np.savez_compressed(HERE/a.output.replace('.json','.npz'),energies=e,ci=c);write(HERE/a.output,result);print(json.dumps(result,indent=2),flush=True)
