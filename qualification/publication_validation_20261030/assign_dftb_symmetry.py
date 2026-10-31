"""Read-only symmetry/ordinal audit of archived TD-DFTB spectra on Oracle.

The auxiliary Gaussian s/p shell has only a group-representation role; no
integral, energy, or radial-overlap calculation is used. State assignments
are explicitly weaker than cross-model transition-overlap qualification.
"""
from pathlib import Path
import socket,os,re,json,hashlib
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
from pyscf import gto,scf,symm,lib
lib.num_threads(1)
P=Path('/home/enzo/MOUSE-publication-validation-20261030')
OUT=P/'external/dftbplus24/symmetry_audit_20261031';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
targets=json.loads((P/'inputs/target_references.json').read_text())
protocol=dict(schema='mouse.dftb.symmetry.audit.v1',host=socket.getfqdn(),
              source_sha256=sha(Path(__file__)),numerical_reference_values_used_for_assignment=False,
              assignment='same point-group irrep and within-irrep energy ordinal as fixed DFT reference; require transition irrep weight >0.999',
              radial_overlap_available=False,cross_model_transition_overlap_available=False,
              scope='symmetry/ordinal assignment only; does not prove same electronic character for roots sharing an irrep',
              original_electronic_outputs_immutable=True,auxiliary_basis='one unit-exponent s and p shell for AO rotation representation only, no electronic calculation')
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')

def inspect(folder,refmol):
    lines=(folder/'geometry.xyz').read_text().splitlines();atoms=[r.split() for r in lines[2:2+int(lines[0])]]
    basis={s:([[0,[1.,1.]]] if s=='H' else [[0,[1.,1.]],[1,[1.,1.]]]) for s in {a[0] for a in atoms}}
    mol=gto.M(atom=[(a[0],list(map(float,a[1:4]))) for a in atoms],basis=basis,unit='Angstrom',symmetry=True,verbose=0)
    assert mol.groupname in ['Cs','C2v','C2h','D2h','C2','D2','Ci'], 'non-discriminating/unsupported group'
    mol.symm_orb,mol.irrep_id=symm.basis.symm_adapted_basis(mol,refmol.groupname,refmol._symm_orig,refmol._symm_axes)
    mol.groupname=refmol.groupname
    text=(folder/'eigenvec.out').read_text();blocks=re.split(r'Eigenvector:\s+(\d+)\s+\(up\)',text)[1:]
    vectors=[];ao_indices=[];offset=0
    for a in atoms:
        count=1 if a[0]=='H' else 4
        ao_indices.extend([offset] if count==1 else [offset,offset+3,offset+1,offset+2])
        offset+=count
    for i in range(0,len(blocks),2):
        values=[]
        for line in blocks[i+1].splitlines():
            m=re.match(r'\s*(?:\d+\s+[A-Za-z]+\s+)?(?:s|p_[xyz])\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$',line)
            if m:values.append(float(m[1]))
        assert len(values)==offset, 'eigenvector AO ordering incomplete'
        vectors.append(np.array(values)[ao_indices])
    c=np.array(vectors).T
    fractions=np.array([np.sum((q.T@c)**2,axis=0)/np.sum(c*c,axis=0) for q in mol.symm_orb])
    assert np.min(np.max(fractions,axis=0))>.999, 'mixed orbital irrep'
    orbital_ids=np.array(mol.irrep_id)[np.argmax(fractions,axis=0)]
    states=[]
    sections=re.split(r'Energy\s+(\d+)\s+([-+0-9.Ee]+)\s+eV\s+S',(folder/'TRA.DAT').read_text())[1:]
    for i in range(0,len(sections),3):
        weights={}
        for h,p,w in re.findall(r'^\s*(\d+)\s*->\s*(\d+)\s+([-+0-9.Ee]+)',sections[i+2],re.M):
            ir=int(orbital_ids[int(h)-1]^orbital_ids[int(p)-1]);weights[ir]=weights.get(ir,0.)+float(w)
        total=sum(weights.values());best=max(weights,key=weights.get)
        states.append(dict(root=int(sections[i]),energy_ev=float(sections[i+1]),
                           symmetry=symm.irrep_id2name(mol.groupname,best),symmetry_weight=weights[best]/total,
                           transition_weight_sum=total,orbital_symmetry_weights={symm.irrep_id2name(mol.groupname,k):v/total for k,v in weights.items()}))
    return mol,states

rows=[];all_states={}
for name,spec in targets.items():
    folder=P/'external/dftbplus24/attempt_03'/name/'td'
    alternate=P/'external/dftbplus24/attempt_04'/name/'td'
    if (alternate/'result.json').exists() and json.loads((alternate/'result.json').read_text()).get('returncode')==0:folder=alternate
    if not (folder/'TRA.DAT').exists():
        rows.append(dict(molecule=name,status='missing_spectrum'));continue
    try:
        ref=Path(spec['reference']);refmol=scf.chkfile.load_mol(str(ref/'backend.chk'))
        refmol.symmetry=True;refmol.build(False,False)
        mol,states=inspect(folder,refmol)
        assert mol.groupname==refmol.groupname, 'point groups differ'
        assert np.allclose((mol.atom_coords()[:,None]-mol.atom_coords()[None,:])**2,
                           (refmol.atom_coords()[:,None]-refmol.atom_coords()[None,:])**2,atol=1e-5), 'reference geometry differs'
        with np.load(ref/'reference.npz') as a:dc=a['coefficients'].copy();occ=a['occupations'].copy()
        no=int(sum(occ==2));q=np.column_stack(refmol.symm_orb)
        ids=np.concatenate([np.full(v.shape[1],ir) for v,ir in zip(refmol.symm_orb,refmol.irrep_id)])
        products=ids[:,None]^ids[None,:]
        vf=next(p for p in [ref/'selected_full.npz',ref/'backend_tda.npz',ref/'baseline.npz'] if p.exists())
        with np.load(vf) as a:rx=a['amplitudes'].reshape(-1,no,len(occ)-no).copy()
        labels=[]
        for x in rx[:max(t['reference_root'] for t in spec['targets'])]:
            ao=q.T@(dc[:,:no]@x@dc[:,no:].T)@q
            weights={int(ir):float(np.sum(ao[products==ir]**2)/np.sum(ao*ao)) for ir in np.unique(products)}
            best=max(weights,key=weights.get)
            assert weights[best]>.999, 'reference state irrep is mixed'
            labels.append(symm.irrep_id2name(refmol.groupname,best))
        labels=np.array(labels)
        all_states[name]=states
        for target in spec['targets']:
            index=target['reference_root']-1;wanted=str(labels[index]);ordinal=int(sum(labels[:index+1]==wanted))
            candidates=[s for s in states if s['symmetry']==wanted and s['symmetry_weight']>.999]
            assigned=candidates[ordinal-1] if len(candidates)>=ordinal else None
            rows.append(dict(molecule=name,state=target['id'],status='symmetry_ordinal_assigned' if assigned else 'no_qualified_irrep_root',
                             symmetry=wanted,ordinal=ordinal,assigned=assigned,
                             cross_model_transition_overlap_qualified=False,source=str(folder),
                             source_sha256={p.name:sha(p) for p in [folder/'geometry.xyz',folder/'eigenvec.out',folder/'TRA.DAT',vf,ref/'reference.npz']}))
    except Exception as e:rows.append(dict(molecule=name,status='unassigned',reason=str(e)))
out=dict(protocol=protocol,rows=rows,spectra=all_states)
(OUT/'report.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(rows,indent=2),flush=True)
