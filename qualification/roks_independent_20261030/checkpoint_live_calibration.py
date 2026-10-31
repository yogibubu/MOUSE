"""Read-only snapshot of scientific NumPy arrays in our paused Oracle job.

No code is injected. Only the explicitly named calibration arrays are read.
The original job is resumed on any error and stopped only after a verified NPZ
checkpoint exists. This preserves completed quadrature blocks and ERI work.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import os,signal,subprocess,re,struct,json,time,numpy as np,sys
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest,key,plan,state_from
pid=3813366;proc=Path('/proc')/str(pid)
assert proc.joinpath('cwd').resolve()==BASE.resolve()
assert proc.stat().st_uid==os.getuid()
cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'full_response.py --molecule uracil' in cmd
out=BASE/'results/full_response/uracil';checkpoint=out/'partial_quadrature.npz';assert not checkpoint.exists()
stored=False
os.kill(pid,signal.SIGSTOP)
try:
 dump=subprocess.check_output([str(BASE/'profiling/bin/py-spy'),'dump','--pid',str(pid),'--locals','--nonblocking'],text=True,timeout=15)
 frame=dump[dump.index('calibrate (full_response.py:'):];frame=frame[:frame.index('\n    run (')]
 line=int(re.search(r'full_response.py:(\d+)',frame).group(1));assert 78<=line<=86,(line,frame)
 blocks=int(re.search(r'\bblocks: (\d+)',frame).group(1))
 pointers={name:int(re.search(r'\b'+name+r': <numpy.ndarray at (0x[0-9a-f]+)>',frame).group(1),16) for name in ['a','d','eri','charge','spin','linear','p0','coords','weight']}
 with proc.joinpath('mem').open('rb',buffering=0) as mem:
  def read(address,count):mem.seek(address);raw=mem.read(count);assert len(raw)==count;return raw
  def array(address):
   pointer=struct.unpack('<Q',read(address+16,8))[0];nd=struct.unpack('<i',read(address+24,4))[0];assert 1<=nd<=4
   dimensions=struct.unpack('<Q',read(address+32,8))[0];strideptr=struct.unpack('<Q',read(address+40,8))[0]
   shape=struct.unpack('<'+'q'*nd,read(dimensions,nd*8));strides=struct.unpack('<'+'q'*nd,read(strideptr,nd*8))
   assert all(d>0 for d in shape) and all(s>=0 for s in strides)
   size=sum((d-1)*s for d,s in zip(shape,strides))+8;assert size<=100_000_000
   result=np.ndarray(shape,dtype=np.float64,buffer=read(pointer,size),strides=strides).copy()
   assert np.all(np.isfinite(result));return result
  data={name:array(pointer) for name,pointer in pointers.items()}
 data['dual']=data.pop('d').reshape(666,36,36)
 assert data['a'].shape==(666,36,36) and data['dual'].shape==data['a'].shape
 assert data['eri'].shape==(36,36,36,36) and data['charge'].shape==(666,666) and data['spin'].shape==(666,666)
 assert data['linear'].shape==(666,) and data['p0'].shape==(36,36)
 case=next(c for c in plan() if c['name']=='uracil');r,xyz=state_from(case['targets'][0]);s=r.reference_state
 request=key(dict(reference=digest(Path(case['reference'])/'reference.npz'),script=digest(BASE/'full_response.py'),source_coordinates=s.coordinates_bohr.tolist()))
 temporary=checkpoint.with_suffix('.tmp.npz');np.savez_compressed(temporary,request_hash=request,completed_blocks=blocks,**data);temporary.replace(checkpoint)
 with np.load(checkpoint) as f:
  assert str(f['request_hash'])==request and int(f['completed_blocks'])==blocks
  assert np.array_equal(f['charge'],data['charge']) and np.array_equal(f['spin'],data['spin'])
 write(out/'partial_quadrature.json',dict(host=HOST,request_hash=request,completed_blocks=blocks,source_line=line,source_sha256=digest(BASE/'full_response.py'),
  process_command=cmd,checkpoint_sha256=digest(checkpoint),capture='scientific arrays read from our own paused process; no code injection',
  shape={k:v.shape for k,v in data.items()}))
 (out/'checkpoint_stack.txt').write_text(dump);stored=True
 print('PRESERVED',blocks,'quadrature blocks, transformed ERIs and first derivative',flush=True)
finally:
 if stored:os.kill(pid,signal.SIGTERM)
 os.kill(pid,signal.SIGCONT)
