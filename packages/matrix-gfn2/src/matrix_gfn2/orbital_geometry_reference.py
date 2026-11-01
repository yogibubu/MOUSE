"""Geometry-bound GFN2 functional reference without a ground-state SCC solve.

The core eigensystem is only a complete canonical AO metadata/seed chart. Its
energy is the physical GFN2 energy of that density, including repulsion and
neutral ATM; it is neither an S0 solution nor an empirical energy anchor.
"""
from dataclasses import replace
import numpy as np


def geometry_orbital_reference(calculator, coordinates_angstrom, template):
    from .calculator import ANGSTROM_TO_BOHR,_covalent_radii
    from .scc import solve_generalized_symmetric_direct,monopole_energy
    from .multipole import anisotropic_energy_and_moment_potentials
    if (tuple(template.atomic_numbers)!=calculator.atomic_numbers
            or template.charge!=calculator.charge or template.multiplicity!=calculator.multiplicity):
        raise ValueError('geometry reference requires the same atoms, charge and multiplicity')
    model_id=('MATRIX-GFN2.v1' if calculator.onsite_profile is None or calculator.onsite_profile.is_identity
              else 'MATRIX-GFN2.v1.onsite.'+calculator.onsite_profile.content_sha256)
    if template.model_id!=model_id or calculator._gaussian_scc_active:
        raise ValueError('geometry reference requires the same supported GFN2 Hamiltonian')
    xyz=np.asarray(coordinates_angstrom,dtype=float)*ANGSTROM_TO_BOHR
    model=calculator._model(xyz,relocate_basis=True)
    basis,s,_,_,core,*_=model
    if (tuple(basis.spherical_atom)!=tuple(template.ao_atom)
            or tuple(basis.spherical_shell)!=tuple(template.ao_shell)
            or not np.array_equal(basis.cartesian_to_spherical,template.basis.spherical_to_cartesian)):
        raise ValueError('geometry reference changed the labelled AO basis')
    e,c=solve_generalized_symmetric_direct(core,s)
    # The metadata uses integer Aufbau occupations, only to define a reference
    # density for the nonlinear functional. Excited occupations are independent.
    electrons=int(sum(calculator._valence)-calculator.charge)
    if electrons%2 or calculator.multiplicity!=1:raise ValueError('ROKS geometry reference requires even-electron singlet metadata')
    occ=np.zeros(len(e));occ[:electrons//2]=1.
    state=replace(template,coordinates_bohr=xyz,overlap=s,fock=core,coefficients=c,
        orbital_energies_hartree=e,alpha_occupations=occ,beta_occupations=occ,
        total_energy_hartree=0.,electron_count=electrons)
    kernel=calculator.density_response_kernel(state)
    density=(c*(2*occ))@c.T;q=kernel.reference_shell_charges;m=kernel.reference_multipoles
    interaction=kernel.shell_interaction
    quadratic=(interaction.evaluate(q,compute_gradient=False).energy_hartree
        if hasattr(interaction,'evaluate') else .5*float(q@interaction@q))
    aes=anisotropic_energy_and_moment_potentials(m,xyz,kernel.anisotropic_parameters,geometry=kernel.anisotropic_geometry)[0]
    electronic=float(np.sum(density*core))+monopole_energy(q,interaction,kernel.partition,quadratic_energy_override=quadratic)+aes+kernel._d4(m.charges).energy_hartree
    native=calculator._active_native_context
    repulsion=float(native.repulsion_energy_gradient(xyz,calculator._atomic_number_array,
        calculator._repulsion_alpha,calculator._effective_nuclear_charges,24.)[0])
    atm=0.
    if len(xyz)>=3:
        neutral=kernel._d4(np.zeros(len(xyz)))
        par=kernel.d4_parameters
        from .dispersion import SparseD4Result, atm_three_body_energy_gradient_sparse
        if isinstance(neutral,SparseD4Result):
            atm=float(atm_three_body_energy_gradient_sparse(xyz,neutral,cutoff_bohr=25.,
                s9=par.s9,a1=par.a1,a2=par.a2).energy_hartree)
        else:
            atm=float(native.atm_energy_gradient(xyz,neutral.c6,neutral.c8,neutral.c6_coordination_derivative,
                _covalent_radii()[calculator._atomic_number_array-1],calculator._electronegativities,
                s9=par.s9,a1=par.a1,a2=par.a2)[0])
    return replace(state,total_energy_hartree=electronic+repulsion+atm)
