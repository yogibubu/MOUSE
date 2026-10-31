"""Separate the ensemble-correlation change from changing the parent DFA.

Keep the validated PBE0-derived first/second derivatives and lambda=.9.
Only replace the open-shell singlet construction by E_T+2(1-xi)K, xi=.32
from the published dd-correlation model. No fitted energies or coefficients.
This is a research EDFT-inspired GFN2 adapter, not the published GX24 DFA.
"""
from remote_bootstrap import BASE,HOST
from pathlib import Path
import json,argparse,numpy as np
import edft_rsh_response as driver
from reks_probe import cases
import sys
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest,key

ROOT=BASE/'results/edft_pbe0';ROOT.mkdir(exist_ok=True)

def calibrate(case,s,ctx):
    source=BASE/'results/full_response'/case['name']/'model.npz'
    with np.load(source) as f:data={k:f[k].copy() for k in f.files if k!='request_hash'}
    request=key(dict(script=digest(__file__),source=digest(source),xi=.32,lambda_exchange=.9,profile='PBE0 parent plus ensemble Hartree/dd correlation'))
    return data,request

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--molecule',required=True);ap.add_argument('--state',default='n_pi');args=ap.parse_args()
    protocol=dict(host=HOST,source_sha256=digest(__file__),solver_source_sha256=digest(BASE/'edft_rsh_response.py'),
                  functional='same corrected PBE0/GFN2 triplet energy plus 2*(1-.32)*K_ab',xi=.32,lambda_exchange=.9,
                  first_and_second_derivatives='unchanged coherent PBE0-derived spectral functional',
                  scalar_energy_offset=0.,fit=False,scope='exposed development only; not GX24',
                  acceptance='all six qualified; spectral MAE below old spectral and old fixed 50/50, maxabs <=0.5, pi_pi guard <=0.3 before independent predictions',
                  existing_composite_weights=[.5,.5],nuclear_derivatives_supported=False)
    file=ROOT/'protocol.json'
    if file.exists():assert json.loads(file.read_text())==protocol
    else:write(file,protocol)
    driver.ROOT=ROOT;driver.calibrate=calibrate;driver.CONSISTENT_REFERENCE=False;driver.lib.num_threads(2)
    case=next(c for c in cases() if c['name']==args.molecule)
    with driver.threadpool_limits(limits=2,user_api='blas'):driver.run(case,'qualified_spectral_response',args.state)
    result=ROOT/'qualified_spectral_response'/args.molecule/args.state/'result.json'
    if result.exists():
        d=json.loads(result.read_text());d['label']='GFN2 EDFT-inspired singlet functional with PBE0 parent; not GX24';d['adapter_source_sha256']=digest(__file__)
        write(result,d);print('EDFT PBE0 RESULT',d,flush=True)
