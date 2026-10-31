"""Remote TD-DFTB2 and spin-purified DeltaDFTB2 on archived exact anchors."""
from pathlib import Path
import socket,os,subprocess,json,time,hashlib,signal,re,argparse
assert socket.getfqdn()=='oracle.sns.it'
P=Path(__file__).resolve().parent;B=P/'software/dftbplus24-env/bin/dftb+';SK=P/'software/mio-1-1/skfiles';OUT=P/'external/dftbplus24/attempt_03';OUT.mkdir(parents=True,exist_ok=True)
ENV=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');VAL={'H':1,'C':4,'N':5,'O':6};NAO={'H':1,'C':4,'N':4,'O':4}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def run(args,cwd):
 start=time.perf_counter();res=cwd/'resources.txt'
 with (cwd/'stdout.log').open('w') as log:
  proc=subprocess.Popen(['/usr/bin/time','-v','-o',str(res)]+list(map(str,args)),cwd=cwd,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=proc.wait(timeout=300);status='completed_unassigned' if rc==0 else 'execution_error'
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=None;status='timeout'
 txt=res.read_text() if res.exists() else '';m=re.search(r'Maximum resident set size \(kbytes\):\s+(\d+)',txt)
 return dict(returncode=rc,status=status,wall_seconds=time.perf_counter()-start,maximum_rss_bytes=int(m[1])*1024 if m else None)
def inp(xyz,method):
 atoms=[l.split()[0] for l in xyz.read_text().splitlines()[2:]];nocc=sum(VAL[x] for x in atoms)//2;nvir=sum(NAO[x] for x in atoms)-nocc;nroots=min(30,nocc*nvir-1)
 text='Geometry = xyzFormat {\n <<< "geometry.xyz"\n}\nHamiltonian = DFTB {\n SCC = Yes\n SCCTolerance = 1e-10\n MaxSCCIterations = 300\n Mixer = Broyden { MixingParameter = 0.1 }\n MaxAngularMomentum {\n H = "s"\n C = "p"\n N = "p"\n O = "p"\n }\n SlaterKosterFiles = Type2FileNames {\n Prefix = "'+str(SK)+'/"\n Separator = "-"\n Suffix = ".skf"\n }\n'
 if method=='delta':text+=' NonAufbau {\n GroundGuess = Yes\n SpinPurify = Yes\n }\n SpinConstants { ShellResolvedSpin = Yes\n H = { -0.0717 }\n C = { -0.0306 -0.0251 -0.0251 -0.0227 }\n N = { -0.0332 -0.0275 -0.0275 -0.0255 }\n O = { -0.0352 -0.0296 -0.0296 -0.0278 }\n }\n'
 text+='}\nOptions { WriteResultsTag = Yes }\nAnalysis {\n WriteEigenvectors = Yes\n EigenvectorsAsText = Yes\n }\n'
 if method=='td':text+='ExcitedState { Casida { NrOfExcitations = '+str(nroots)+'\n Symmetry = "singlet"\n Diagonaliser = Arpack {}\n WriteTransitions = Yes\n WriteXplusY = Yes\n } }\n'
 for element in set(VAL)-set(atoms):
  text=re.sub(r'\n '+element+r' = [^\n]+','',text)
 return text
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--all',action='store_true');a=ap.parse_args()
 protocol=dict(host=socket.getfqdn(),software_sha256=sha(B),parameters_commit=subprocess.check_output(['git','-C',str(SK.parent),'rev-parse','HEAD'],text=True).strip(),parameters='mio-1-1',threads=2,spin_constants_source='mio-1-1/extras/spinw.txt',assignment='pending; higher target states not applicable for DeltaDFTB')
 f=OUT/'protocol.json'
 if f.exists():assert json.loads(f.read_text())==protocol
 else:write(f,protocol)
 files=sorted((P/'inputs/anchors').glob('*_anchor.xyz'));files=files if a.all else [f for f in files if f.stem=='acetaldehyde_anchor']
 for xyz in files:
  for method in ['td','delta']:
   d=OUT/xyz.stem.removesuffix('_anchor')/method;d.mkdir(parents=True,exist_ok=True);done=d/'result.json';request=dict(geometry_sha256=sha(xyz),protocol=protocol,method=method)
   if done.exists():assert json.loads(done.read_text())['request']==request;continue
   (d/'geometry.xyz').write_bytes(xyz.read_bytes());text=inp(xyz,method);(d/'dftb_in.hsd').write_text(text);r=run([B],d);r.update(host=socket.getfqdn(),molecule=xyz.stem.removesuffix('_anchor'),method=method,request=request,input_sha256=sha(d/'dftb_in.hsd'))
   if (d/'detailed.out').exists():
    txt=(d/'detailed.out').read_text();m=re.search(r'S0\s*->\s*S1:\s+([-+0-9.Ee]+)\s+H\s+([-+0-9.Ee]+)\s+eV',txt)
    if m:r['lowest_singlet_gap_ev']=float(m[2])
   write(done,r);print(r['molecule'],method,r['status'],flush=True)
