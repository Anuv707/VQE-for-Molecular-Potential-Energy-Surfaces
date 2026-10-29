"""
Ab initio molecular geometry construction and Hartree-Fock molecular integral generation
under the Born-Oppenheimer approximation (STO-3G minimal Gaussian basis set).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# Isotopic atomic masses (IUPAC, unified atomic mass units u / Da)
MASS_H1_AMU = 1.00782503223
MASS_LI7_AMU = 7.0160034366


@dataclass(frozen=True)
class MoleculeSpec:
    """Specification of a diatomic molecule and its active electronic space."""

    name: str
    symbols: List[str]
    bond_length_angstrom: float
    basis_name: str
    charge: int
    multiplicity: int
    total_electrons: int
    active_electrons: int
    core_orbitals: List[int]
    active_orbitals: List[int]
    n_qubits: int
    reduced_mass_amu: float


@dataclass
class MolecularIntegralData:
    """Container for Born-Oppenheimer molecular integrals and RHF reference data."""

    spec: MoleculeSpec
    pl_molecule: qml.qchem.Molecule
    core_constant: float
    one_body_integrals: np.ndarray
    two_body_integrals_phys: np.ndarray
    two_body_integrals_chem: np.ndarray
    orbital_energies: np.ndarray
    hf_energy: float


def get_default_bond_grid(molecule_name: str) -> np.ndarray:
    """
    Return a well-resolved interatomic distance grid (in Angstroms) covering
    the repulsive wall, equilibrium geometry (R_e), inflection point, and dissociation limit.
    """
    mol_upper = molecule_name.strip().upper()
    if mol_upper == "H2":
        # Fine spacing around R_e ~ 0.735 A, extending from 0.35 A to 2.80 A (25 points)
        return np.array(
            [
                0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                0.72, 0.735, 0.75, 0.80, 0.85, 0.90, 1.00, 1.10,
                1.20, 1.35, 1.50, 1.70, 1.90, 2.10, 2.30, 2.55, 2.80,
            ],
            dtype=float,
        )
    elif mol_upper == "LIH":
        # Fine spacing around R_e ~ 1.546 A (STO-3G), extending from 0.85 A to 3.40 A (22 points)
        return np.array(
            [
                0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.50,
                1.546, 1.60, 1.68, 1.78, 1.90, 2.05, 2.20, 2.35,
                2.50, 2.65, 2.80, 3.00, 3.20, 3.40,
            ],
            dtype=float,
        )
    else:
        raise ValueError(f"Unsupported molecule '{molecule_name}'. Supported: 'H2', 'LiH'.")


def build_molecule(
    molecule_name: str,
    bond_length: float,
    basis_name: str = "sto-3g",
    active_space: bool = True,
) -> Tuple[MoleculeSpec, qml.qchem.Molecule]:
    """
    Construct a PennyLane qchem.Molecule object and its active-space specification.

    Parameters
    ----------
    molecule_name : str
        Molecule identifier ('H2' or 'LiH').
    bond_length : float
        Interatomic nuclear separation R in Angstroms (must be > 0).
    basis_name : str
        Gaussian basis set name (default: 'sto-3g').
    active_space : bool
        If True, for LiH freezes the core Li 1s orbital and selects the three
        sigma-symmetry valence orbitals along the internuclear z-axis (6 spin-orbitals).

    Returns
    -------
    Tuple[MoleculeSpec, qml.qchem.Molecule]
    """
    if bond_length is None or not np.isfinite(bond_length) or float(bond_length) <= 0.0:
        raise ValueError(f"Bond length must be a positive finite number, got {bond_length}.")

    R = float(bond_length)
    mol_upper = molecule_name.strip().upper()
    basis_clean = basis_name.strip().lower()

    if mol_upper == "H2":
        symbols = ["H", "H"]
        coords = pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, R]], requires_grad=False)
        pl_mol = qml.qchem.Molecule(
            symbols, coords, unit="angstrom", basis_name=basis_clean, charge=0, mult=1
        )
        core_orbs: List[int] = []
        active_orbs = [0, 1]
        total_elec = 2
        active_elec = 2
        reduced_mass = (MASS_H1_AMU * MASS_H1_AMU) / (MASS_H1_AMU + MASS_H1_AMU)
        canonical_name = "H2"

    elif mol_upper == "LIH":
        symbols = ["Li", "H"]
        coords = pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, R]], requires_grad=False)
        pl_mol = qml.qchem.Molecule(
            symbols, coords, unit="angstrom", basis_name=basis_clean, charge=0, mult=1
        )
        total_elec = 4
        reduced_mass = (MASS_LI7_AMU * MASS_H1_AMU) / (MASS_LI7_AMU + MASS_H1_AMU)
        canonical_name = "LiH"

        if active_space:
            # Run RHF SCF to inspect molecular orbital symmetries along the z-axis.
            # In STO-3G for LiH, AO indices [0, 1, 4, 5] have sigma symmetry (1s, 2s, 2pz, H 1s)
            # while AO indices [2, 3] are pure pi_x, pi_y (Li 2px, 2py) with zero occupancy.
            qml.qchem.scf(pl_mol)()
            mo_coeffs = np.asarray(pl_mol.mo_coefficients, dtype=float)
            sigma_ao_indices = [0, 1, 4, 5]
            sigma_weights = np.sum(mo_coeffs[sigma_ao_indices, :] ** 2, axis=0)
            sigma_mo_indices = [int(i) for i in np.argsort(-sigma_weights)[:4]]
            sigma_mo_indices.sort()
            # Core is the lowest-energy orbital (Li 1s, index 0); active are the 3 valence sigma MOs
            core_orbs = [sigma_mo_indices[0]]
            active_orbs = sigma_mo_indices[1:]
            active_elec = 2
        else:
            core_orbs = [0]
            active_orbs = [1, 2, 3, 4, 5]
            active_elec = 2
    else:
        raise ValueError(f"Unsupported molecule '{molecule_name}'. Supported: 'H2', 'LiH'.")

    spec = MoleculeSpec(
        name=canonical_name,
        symbols=symbols,
        bond_length_angstrom=R,
        basis_name=basis_clean,
        charge=0,
        multiplicity=1,
        total_electrons=total_elec,
        active_electrons=active_elec,
        core_orbitals=core_orbs,
        active_orbitals=active_orbs,
        n_qubits=2 * len(active_orbs),
        reduced_mass_amu=float(reduced_mass),
    )
    return spec, pl_mol


def compute_molecular_integrals(
    molecule_name: str,
    bond_length: float,
    basis_name: str = "sto-3g",
    active_space: bool = True,
) -> MolecularIntegralData:
    """
    Compute Restricted Hartree-Fock (RHF) orbitals, core energy shift, and one-/two-body
    electronic integrals in the molecular orbital basis.
    """
    spec, pl_mol = build_molecule(
        molecule_name=molecule_name,
        bond_length=bond_length,
        basis_name=basis_name,
        active_space=active_space,
    )

    core_constant_arr, one_body, two_body_phys = qml.qchem.electron_integrals(
        pl_mol, core=spec.core_orbitals, active=spec.active_orbitals
    )()

    core_constant = float(np.asarray(core_constant_arr).reshape(-1)[0])
    one_body_np = np.asarray(one_body, dtype=float)
    two_body_phys_np = np.asarray(two_body_phys, dtype=float)
    # Convert PennyLane <pq|sr> to chemists' notation (pr|qs) for Qiskit Nature
    two_body_chem_np = np.einsum("pqrs->psqr", two_body_phys_np)

    # Compute exact Hartree-Fock energy and canonical Fock orbital energies from active-space integrals:
    # In PennyLane's electron_integrals convention (1/2 sum g_pqrs a+_p a+_q a_r a_s),
    # Coulomb J_pj = g_{p,j,j,p} and Exchange K_pj = g_{p,j,p,j}
    n_occ = spec.active_electrons // 2
    n_act = one_body_np.shape[0]
    orbital_energies = np.zeros(n_act, dtype=float)
    for p in range(n_act):
        fock_pp = one_body_np[p, p]
        for j in range(n_occ):
            fock_pp += 2.0 * two_body_phys_np[p, j, j, p] - two_body_phys_np[p, j, p, j]
        orbital_energies[p] = fock_pp

    hf_elec = 0.0
    for i in range(n_occ):
        hf_elec += one_body_np[i, i] + orbital_energies[i]
    hf_energy = core_constant + float(hf_elec)

    return MolecularIntegralData(
        spec=spec,
        pl_molecule=pl_mol,
        core_constant=core_constant,
        one_body_integrals=one_body_np,
        two_body_integrals_phys=two_body_phys_np,
        two_body_integrals_chem=two_body_chem_np,
        orbital_energies=orbital_energies,
        hf_energy=hf_energy,
    )
