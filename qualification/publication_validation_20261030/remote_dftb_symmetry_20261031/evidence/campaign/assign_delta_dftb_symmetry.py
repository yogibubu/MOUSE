"""Audit saved spin-purified Delta-DFTB S1 spatial symmetry, no new solves.

Only a globally first reference singlet is eligible. This does not establish
cross-model orbital character, a transition overlap, or exact spin projection.
"""
from pathlib import Path
import os,socket,re,json,hashlib
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
from pyscf import gto,scf,symm,lib
lib.num_threads(1)
P=Path('/home/enzo/MOUSE-publication-validation-20261030')
OUT=P/'external/dftbplus24/symmetry_audit_20261031'
td=json.loads((OUT/'report.json').read_text())
specs=json.loads((P/'inputs/target_references.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def audit(folder,ref):
    rm=scf.chkfile.load_mol(str(ref/'backend.chk'));rm.symmetry=True;rm.build(False,False)
    lines=(folder/'geometry.xyz').read_text().splitlines();atoms=[l.split() for l in lines[2:2+int(lines[0])]]
    basis={s:([[0,[1.,1.]]] if s=='H' else [[0,[1.,1.]],[1,[1.,1.]]]) for s in {a[0] for a in atoms}}
    mol=gto.M(atom=[(a[0],list(map(float,a[1:4]))) for a in atoms],unit='Angstrom',basis=basis,verbose=0)
    assert np.max(abs(mol.atom_coords()-rm.atom_coords()))<2e-6, 'reference geometry differs'
    qs,ids=symm.basis.symm_adapted_basis(mol,rm.groupname,rm._symm_orig,rm._symm_axes)
    assert rm.groupname!='C1', 'non-discriminating symmetry'
    permutation=[];offset=0
    for a in atoms:
        n=1 if a[0]=='H' else 4
        permutation.extend([offset] if n==1 else [offset,offset+3,offset+1,offset+2]);offset+=n
    blocks=re.split(r'Eigenvector:\s+(\d+)\s+\((up|down)\)',(folder/'eigenvec.out').read_text())[1:]
    vectors={'up':[],'down':[]}
    for i in range(0,len(blocks),3):
        values=[]
        for line in blocks[i+2].splitlines():
            m=re.match(r'\s*(?:\d+\s+[A-Za-z]+\s+)?(?:s|p_[xyz])\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$',line)
            if m:values.append(float(m[1]))
        assert len(values)==offset
        vectors[blocks[i+1]].append(np.array(values)[permutation])
    tag=(folder/'results.tag').read_text();m=re.search(r'^filling\s*:real:3:(\d+),1,2\s*\n(.*?)(?=^[A-Za-z_]+\s*:|\Z)',tag,re.M|re.S)
    assert m, 'final broken-symmetry occupations unavailable'
    occupations=np.array(list(map(float,m[2].split()))).reshape(2,int(m[1]))
    assert np.max(abs(occupations-np.rint(occupations)))<1e-8
    assert np.sum(occupations[0])==np.sum(occupations[1]), 'Ms differs from zero'
    labels=[];minimum=1.
    for spin,occ in zip(['up','down'],occupations):
        c=np.array(vectors[spin]).T;assert c.shape==(offset,offset)
        fractions=np.array([np.sum((q.T@c)**2,axis=0)/np.sum(c*c,axis=0) for q in qs])
        minimum=min(minimum,float(np.min(np.max(fractions[:,occ>0],axis=0))))
        assert minimum>.999, 'occupied determinant breaks spatial symmetry'
        labels.append(np.array(ids)[np.argmax(fractions,axis=0)][occ>0].tolist())
    determinant_irrep=0
    for spin in labels:
        for ir in spin:determinant_irrep^=int(ir)
    detailed=(folder/'detailed.out').read_text()
    assert 'Purified State Total Energy' in detailed
    gap=re.search(r'S0\s*->\s*S1:\s+([-+0-9.Ee]+)\s+H',detailed)
    return dict(symmetry=symm.irrep_id2name(rm.groupname,determinant_irrep),
                gap_ev=float(gap[1])*27.211386245988,minimum_occupied_irrep_weight=minimum,
                occupations=occupations.tolist(),occupied_irreps=labels,
                source_sha256={p.name:sha(p) for p in [folder/'geometry.xyz',folder/'eigenvec.out',folder/'results.tag',folder/'detailed.out']})

rows=[]
for t in td['rows']:
    name=t['molecule'];state=t.get('state');row=dict(molecule=name,state=state)
    if t['status']!='symmetry_ordinal_assigned':
        rows.append(dict(row,status='unassigned_reference_symmetry'));continue
    target=next(x for x in specs[name]['targets'] if x['id']==state)
    if target['reference_root']!=1:
        rows.append(dict(row,status='not_applicable_higher_reference_root'));continue
    folder=P/'external/dftbplus24/attempt_03'/name/'delta'
    alternate=P/'external/dftbplus24/attempt_04'/name/'delta'
    if (alternate/'result.json').exists() and json.loads((alternate/'result.json').read_text()).get('returncode')==0:folder=alternate
    try:
        assert json.loads((folder/'result.json').read_text())['returncode']==0, 'electronic calculation not completed'
        result=audit(folder,Path(specs[name]['reference']))
        assert result['symmetry']==t['symmetry'], 'final determinant symmetry differs from reference target'
        rows.append(dict(row,status='lowest_singlet_spatial_symmetry_assigned',assigned=result,source=str(folder),
                         cross_model_transition_overlap_qualified=False))
    except Exception as e:rows.append(dict(row,status='unassigned',reason=str(e),source=str(folder)))
out=dict(host=socket.getfqdn(),source_sha256=sha(Path(__file__)),rows=rows,
         scope='Spin-purified Delta-DFTB lowest-S1 energy with final broken-symmetry spatial irrep checked against first global reference singlet; no transition-overlap qualification or exact projected-DFT claim')
(OUT/'delta_report.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(rows,indent=2),flush=True)
