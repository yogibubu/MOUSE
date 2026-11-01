"""Certify unchanged valence span and inspect quadrature locality."""
from pigment_compact_reference import HERE,load_state
from gaussian_embedding_basis import source_molecule
from pigment_reference_embedding import write
from scipy.linalg import solve
from pyscf import scf,gto
from threadpoolctl import threadpool_limits
import numpy as np,json,argparse,socket
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();s=load_state(a.name);out=HERE/'pigment_full_response'/a.name/'localized_chart';out.mkdir(parents=True,exist_ok=True)
with threadpool_limits(2):
 with np.load(HERE/'pigment_embedding'/a.name/'embedding.npz') as z:lift=z['embedding'].copy()
 cert=json.loads((HERE/'pigment_embedding'/a.name/'certificate.json').read_text());nc=cert['frozen_core_orbitals']
 with np.load(HERE/'pigment_dft_reference'/a.name/'PBE0/reference.npz') as z:cd=z['coefficients'].copy();sd=z['overlap'].copy()
 mol=scf.chkfile.load_mol(str(HERE/'pigment_dft_reference'/a.name/'PBE0/backend.chk'));small,t=source_molecule(s,raw=True);cross=gto.intor_cross('int1e_ovlp',mol,small)@t;cv=cd[:,nc:];raw=cv@(cv.T@cross);R=solve(s.overlap,lift.T@sd@raw);inverse_R=np.linalg.inv(R);error=float(np.max(abs(lift@R-raw)));metric_error=float(np.max(abs((lift@R-raw).T@sd@(lift@R-raw))))**.5;print('Coefficient span error',error,'metric norm',metric_error,flush=True);assert metric_error<1e-7;np.savez(out/'local_chart.npz',R=R,inverse_R=inverse_R)
 files=sorted((HERE/'pigment_compact_reference'/a.name/'grid').glob('chunk*.npz'));rows=[]
 for path in files[::max(1,len(files)//6)]:
  with np.load(path) as z:before=z['phi'][:,:512];after=before@R
  rows.append(dict(chunk=path.name,native=[int(np.count_nonzero(np.max(abs(before),axis=(0,1))>cut)) for cut in [1e-12,1e-9,1e-7,1e-5]],localized=[int(np.count_nonzero(np.max(abs(after),axis=(0,1))>cut)) for cut in [1e-12,1e-9,1e-7,1e-5]]))
 report=dict(host=socket.getfqdn(),molecule=a.name,complete_span_coefficient_error=error,complete_span_metric_error=metric_error,condition=float(np.linalg.cond(R)),feature_AO_counts=rows,cutoffs=[1e-12,1e-9,1e-7,1e-5],orbitals_discarded=0);write(out/'local_chart_certificate.json',report);print(json.dumps(report,indent=2))
