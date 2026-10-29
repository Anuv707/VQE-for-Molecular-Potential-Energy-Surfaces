"""
Unit and integration tests for UCCSD and HEA quantum circuit ansatze,
classical optimizers (SLSQP, L-BFGS-B, COBYLA, SPSA), Rayleigh-Ritz variational
bound verification, and chemical accuracy on H2 and LiH.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.chemistry import build_qubit_hamiltonian, compare_mappings, compute_molecular_integrals
from src.algorithms import (
    CHEMICAL_ACCURACY_HA,
    build_ansatz,
    run_vqe,
    solve_exact_fci,
)


def test_ansatz_construction_and_edge_cases():
    """Verify UCCSD and HEA parameter counts and error handling."""
    ints = compute_molecular_integrals("H2", 0.735)
    ham = build_qubit_hamiltonian(ints, "jordan_wigner")

    ucc = build_ansatz(ham, ansatz_type="UCCSD")
    assert ucc.n_qubits == 4
    assert ucc.n_single_excitations == 2
    assert ucc.n_double_excitations == 1
    assert ucc.n_params == 3

    hea1 = build_ansatz(ham, ansatz_type="HEA", n_layers=1)
    hea2 = build_ansatz(ham, ansatz_type="HEA", n_layers=2)
    assert hea2.n_params == 2 * hea1.n_params
    assert hea2.n_two_qubit_gates == 2 * hea1.n_two_qubit_gates

    with pytest.raises(ValueError, match="Unsupported ansatz"):
        build_ansatz(ham, ansatz_type="QAOA")
    with pytest.raises(ValueError, match="n_layers must be >= 1"):
        build_ansatz(ham, ansatz_type="HEA", n_layers=0)


def test_vqe_chemical_accuracy_and_variational_bound_h2():
    """Verify VQE (UCCSD) achieves chemical accuracy and obeys Rayleigh-Ritz on H2."""
    for R in [0.50, 0.735, 1.60]:
        ints = compute_molecular_integrals("H2", R)
        ham = build_qubit_hamiltonian(ints, "jordan_wigner")
        exact = solve_exact_fci(ham)

        res = run_vqe(
            ham,
            ansatz_type="UCCSD",
            optimizer_name="SLSQP",
            maxiter=80,
            exact_solution=exact,
        )
        assert res.variational_bound_satisfied
        assert res.energy >= exact.fci_energy - 1e-9
        assert res.abs_error_ha < CHEMICAL_ACCURACY_HA
        assert res.chemical_accuracy_achieved


def test_direct_expval_matches_matrix_expectation():
    """Verify PennyLane QNode direct expval matches fast statevector matrix expectation for UCCSD and HEA."""
    ints = compute_molecular_integrals("H2", 0.735)
    ham = build_qubit_hamiltonian(ints, "jordan_wigner")
    exact = solve_exact_fci(ham)

    for ans in ["UCCSD", "HEA"]:
        res_fast = run_vqe(
            ham, ansatz_type=ans, optimizer_name="SLSQP", exact_solution=exact, use_direct_expval=False
        )
        res_direct = run_vqe(
            ham, ansatz_type=ans, optimizer_name="SLSQP", exact_solution=exact, use_direct_expval=True
        )
        assert np.isclose(res_fast.energy, res_direct.energy, atol=1e-6)


def test_all_optimizers_converge_and_respect_variational_bound():
    """Verify SLSQP, L-BFGS-B, COBYLA, and SPSA all reduce energy below HF and obey E >= E_FCI."""
    ints = compute_molecular_integrals("H2", 0.735)
    ham = build_qubit_hamiltonian(ints, "jordan_wigner")
    exact = solve_exact_fci(ham)

    for opt in ["SLSQP", "L-BFGS-B", "COBYLA", "SPSA"]:
        res = run_vqe(
            ham,
            ansatz_type="UCCSD",
            optimizer_name=opt,
            maxiter=100,
            exact_solution=exact,
            seed=42,
        )
        assert res.variational_bound_satisfied
        assert res.energy < exact.hf_energy
        assert res.abs_error_ha < CHEMICAL_ACCURACY_HA

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        run_vqe(ham, optimizer_name="ADAMW")


def test_lih_vqe_chemical_accuracy_all_mappings():
    """Verify VQE (UCCSD) achieves chemical accuracy on LiH across all 4 fermion-to-qubit mappings."""
    mapped = compare_mappings("LiH", 1.546)
    for m_name, ham in mapped.items():
        exact = solve_exact_fci(ham, restrict_particle_sector=(m_name == "jordan_wigner"))
        res = run_vqe(
            ham,
            ansatz_type="UCCSD",
            optimizer_name="SLSQP",
            maxiter=80,
            exact_solution=exact,
        )
        assert res.variational_bound_satisfied
        assert res.abs_error_ha < CHEMICAL_ACCURACY_HA, (
            f"LiH {m_name} failed chemical accuracy: err={res.abs_error_ha}"
        )
        assert res.correlation_energy_recovered_pct > 99.9


def test_hea_monotonic_layer_improvement_and_correlation_recovery():
    """Verify HEA escapes the HF saddle, recovers >90% correlation energy, and improves with L=1,2,3."""
    for mol, R in [("H2", 0.735), ("LiH", 1.546)]:
        ints = compute_molecular_integrals(mol, R, active_space=True)
        ham = build_qubit_hamiltonian(ints, "jordan_wigner")
        exact = solve_exact_fci(ham)
        errors = []
        for L in [1, 2, 3]:
            res = run_vqe(
                ham,
                ansatz_type="HEA",
                n_layers=L,
                optimizer_name="SLSQP",
                maxiter=100,
                exact_solution=exact,
            )
            assert res.variational_bound_satisfied
            assert res.correlation_energy_recovered_pct > 90.0
            assert res.abs_error_ha < CHEMICAL_ACCURACY_HA
            errors.append(res.abs_error_ha)
        assert errors[2] <= errors[1] <= errors[0]


def test_finite_shot_vqe_sampling():
    """Verify finite-shot NISQ VQE sampling works with SPSA and validates shot count."""
    ints = compute_molecular_integrals("H2", 0.735)
    ham = build_qubit_hamiltonian(ints, "parity_2q")
    exact = solve_exact_fci(ham, restrict_particle_sector=False)

    res_shots = run_vqe(
        ham,
        ansatz_type="UCCSD",
        optimizer_name="SPSA",
        maxiter=40,
        shots=2048,
        exact_solution=exact,
        seed=42,
    )
    assert res_shots.energy < exact.hf_energy
    assert res_shots.abs_error_ha < 0.01

    with pytest.raises(ValueError, match="shots must be a positive integer"):
        run_vqe(ham, shots=0)
