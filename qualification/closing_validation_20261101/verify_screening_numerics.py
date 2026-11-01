"""Verify numerical screening algebra; W itself defines the active partition."""
from pathlib import Path
import socket,json
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
with threadpool_limits(2):
 rows=[]
 for w in [11,12,13,16]:
  folder=HERE/f'pigment_screening_expansion/Beta_carotene/frontier_{w}';cert=json.loads((folder/'certificate.json').read_text());d=np.load(folder/'dielectric_polarization.npy');bn=np.load(folder/'native_polarization.npy',mmap_mode='r');bb=np.load(folder/'bath_polarization.npy',mmap_mode='r');lp=np.load(folder/'lp.npy',mmap_mode='r');weighted=np.load(folder/'weighted_lp.npy',mmap_mode='r');screen=np.load(folder/'screened_pair.npy',mmap_mode='r');rng=np.random.default_rng(184+w);qe=0.;re=0.;we=0.
  for repeat in range(3):
   v=rng.normal(size=len(d));v/=np.linalg.norm(v);actual=float(v@d@v);expected=float(np.sum((v@bn)**2)+.6*np.sum((v@bb)**2));qe=max(qe,abs(actual-expected)/max(1.,abs(expected)))
   j=int(rng.integers(lp.shape[1]));residual=d@weighted[:,j]+weighted[:,j]-lp[:,j];re=max(re,float(np.linalg.norm(residual)/np.linalg.norm(lp[:,j])));we=max(we,float(np.max(abs(lp[:,j]@weighted-screen[j]))))
  assert max(qe,re,we)<1e-9,(w,qe,re,we)
  rows.append(dict(W=w,quadratic_form_relative_error=qe,dielectric_solve_relative_residual=re,factorized_screened_row_max_error=we,positive_dielectric_by_construction=cert['positive_dielectric_by_construction']))
 result=dict(host=socket.getfqdn(),rows=rows,threshold=1e-9,random_checks_each=3,SCF_calls=0,physical_parameters_retuned=False)
 (HERE/'screening_numerical_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
