"""
Exact Full Configuration Interaction (FCI) diagonalization and Hartree-Fock (HF) reference
evaluators for benchmarking VQE Potential Energy Surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.linalg

from src.chemistry.mappings import MappedHamiltonian


@dataclass
class ExactSolution:
    """Container for exact FCI and Hartree-Fock reference energies at a given geometry."""

    molecule_name: str
    bond_length: float
    mapping_name: str
    fci_energy: float
    hf_energy: float
    correlation_energy: float
    first_excited_energy: float
    energy_gap: float
    ground_state_vector: np.ndarray


def _filter_particle_sector_jw(
    ham_matrix: np.ndarray, n_qubits: int, n_electrons: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Restrict a Jordan-Wigner Hamiltonian matrix to the neutral singlet sector
    with n_alpha = n_electrons // 2 (even wires) and n_beta = n_electrons // 2 (odd wires).
    Returns (subspace_indices, subspace_matrix).
    """
    n_alpha_target = n_electrons // 2
    n_beta_target = n_electrons - n_alpha_target
    valid_indices = []

    for idx in range(1 << n_qubits):
        bits = [(idx >> (n_qubits - 1 - w)) & 1 for w in range(n_qubits)]
        n_alpha = sum(bits[0::2])
        n_beta = sum(bits[1::2])
        if n_alpha == n_alpha_target and n_beta == n_beta_target:
            valid_indices.append(idx)

    idx_arr = np.array(valid_indices, dtype=int)
    sub_mat = ham_matrix[np.ix_(idx_arr, idx_arr)]
    return idx_arr, sub_mat


def solve_exact_fci(
    mapped_ham: MappedHamiltonian,
    active_electrons: int = 2,
    restrict_particle_sector: bool = True,
) -> ExactSolution:
    """
    Compute the exact Full Configuration Interaction (FCI) ground-state energy and
    wavefunction via exact Hermitian matrix diagonalization.

    Parameters
    ----------
    mapped_ham : MappedHamiltonian
        The multi-qubit Pauli Hamiltonian.
    active_electrons : int
        Number of active electrons (default: 2 for both H2 and active-space LiH).
    restrict_particle_sector : bool
        If True and mapping is 'jordan_wigner', diagonalizes within the exact
        (N_alpha, N_beta) particle-conserving subspace to prevent unphysical
        ionized states from crossing at extreme geometries.
    """
    full_mat = mapped_ham.to_matrix()
    dim = full_mat.shape[0]

    if restrict_particle_sector and mapped_ham.mapping_name == "jordan_wigner":
        sub_indices, sub_mat = _filter_particle_sector_jw(
            full_mat, mapped_ham.n_qubits, active_electrons
        )
        eigvals, eigvecs_sub = scipy.linalg.eigh(sub_mat)
        fci_energy = float(np.real(eigvals[0]))
        first_excited = float(np.real(eigvals[1])) if len(eigvals) > 1 else fci_energy
        ground_vec = np.zeros(dim, dtype=complex)
        ground_vec[sub_indices] = eigvecs_sub[:, 0]
    else:
        eigvals, eigvecs = scipy.linalg.eigh(full_mat)
        fci_energy = float(np.real(eigvals[0]))
        first_excited = float(np.real(eigvals[1])) if len(eigvals) > 1 else fci_energy
        ground_vec = eigvecs[:, 0]

    hf_energy = float(mapped_ham.hf_energy)
    corr_energy = fci_energy - hf_energy

    return ExactSolution(
        molecule_name=mapped_ham.molecule_name,
        bond_length=mapped_ham.bond_length,
        mapping_name=mapped_ham.mapping_name,
        fci_energy=fci_energy,
        hf_energy=hf_energy,
        correlation_energy=corr_energy,
        first_excited_energy=first_excited,
        energy_gap=first_excited - fci_energy,
        ground_state_vector=ground_vec,
    )
