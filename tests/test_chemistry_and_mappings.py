"""
Unit and integration tests for molecular geometry construction, RHF integral evaluation,
and Fermion-to-Qubit transformations (Jordan-Wigner, Parity, Parity-2Q, Bravyi-Kitaev).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.chemistry import (
    build_molecule,
    build_qubit_hamiltonian,
    compare_mappings,
    compute_molecular_integrals,
    get_default_bond_grid,
)
from src.algorithms import solve_exact_fci


def test_h2_and_lih_geometry_and_integrals():
    """Verify RHF integrals, active-space dimensions, and HF energies for H2 and LiH."""
    h2_ints = compute_molecular_integrals("H2", 0.735)
    assert h2_ints.spec.name == "H2"
    assert h2_ints.spec.n_qubits == 4
    assert h2_ints.spec.active_electrons == 2
    assert h2_ints.one_body_integrals.shape == (2, 2)
    assert h2_ints.two_body_integrals_phys.shape == (2, 2, 2, 2)
    assert np.isclose(h2_ints.hf_energy, -1.11699899, atol=1e-5)
    assert np.isclose(h2_ints.orbital_energies[0], -0.5806, atol=1e-2)
    assert h2_ints.orbital_energies[0] < 0.0 < h2_ints.orbital_energies[1]

    lih_ints = compute_molecular_integrals("LiH", 1.546, active_space=True)
    assert lih_ints.spec.name == "LiH"
    assert lih_ints.spec.n_qubits == 6
    assert lih_ints.spec.active_electrons == 2
    assert lih_ints.spec.core_orbitals == [0]
    assert lih_ints.spec.active_orbitals == [1, 2, 5]
    assert lih_ints.one_body_integrals.shape == (3, 3)
    assert lih_ints.hf_energy < -7.85
    assert lih_ints.orbital_energies[0] < lih_ints.orbital_energies[1] < lih_ints.orbital_energies[2]


def test_molecule_edge_cases_and_invalid_inputs():
    """Verify that invalid bond lengths and unsupported molecule names raise ValueError."""
    with pytest.raises(ValueError, match="positive finite"):
        build_molecule("H2", 0.0)
    with pytest.raises(ValueError, match="positive finite"):
        build_molecule("H2", -1.2)
    with pytest.raises(ValueError, match="positive finite"):
        build_molecule("LiH", float("nan"))
    with pytest.raises(ValueError, match="positive finite"):
        build_molecule("LiH", float("inf"))
    with pytest.raises(ValueError, match="Unsupported molecule"):
        build_molecule("CO2", 1.1)
    with pytest.raises(ValueError, match="Unsupported molecule"):
        get_default_bond_grid("H2O")


def test_fermion_to_qubit_mappings_isospectrality():
    """
    Verify that Jordan-Wigner, Parity, Parity-2Q (two-qubit tapered), and Bravyi-Kitaev
    all produce the exact same ground-state FCI eigenvalue (isospectrality).
    """
    mapped = compare_mappings("H2", 0.735)
    assert set(mapped.keys()) == {"jordan_wigner", "parity", "parity_2q", "bravyi_kitaev"}

    # Check qubit reduction in parity_2q
    assert mapped["jordan_wigner"].n_qubits == 4
    assert mapped["parity"].n_qubits == 4
    assert mapped["bravyi_kitaev"].n_qubits == 4
    assert mapped["parity_2q"].n_qubits == 2
    assert mapped["parity_2q"].n_pauli_terms < mapped["jordan_wigner"].n_pauli_terms

    e_ref = -1.1373060359616476
    for m_name, ham in mapped.items():
        sol = solve_exact_fci(ham, restrict_particle_sector=(m_name == "jordan_wigner"))
        assert math.isclose(sol.fci_energy, e_ref, abs_tol=1e-9), (
            f"Mapping {m_name} FCI energy {sol.fci_energy} != {e_ref}"
        )


def test_invalid_mapping_raises():
    """Verify that an unknown fermion-to-qubit mapping name raises ValueError."""
    ints = compute_molecular_integrals("H2", 0.735)
    with pytest.raises(ValueError, match="Unsupported mapping"):
        build_qubit_hamiltonian(ints, mapping="bogoliubov_transform")
