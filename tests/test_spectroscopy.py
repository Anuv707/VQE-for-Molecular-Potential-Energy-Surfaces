"""
Unit tests for Morse potential fitting, spectroscopic constant extraction
(R_e, D_e, omega_e, ZPE), and vibrational eigenstate turning points.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.spectroscopy import (
    compute_morse_vibrational_levels,
    extract_spectroscopic_constants,
    morse_potential,
)


def test_synthetic_morse_parameter_recovery():
    """Verify exact recovery of R_e, E_min, and D_e from a noiseless Morse curve."""
    R_grid = np.linspace(0.45, 2.60, 25)
    e_min_true = -1.1373
    d_e_true = 0.1680
    a_true = 1.95
    r_e_true = 0.735

    E_grid = morse_potential(R_grid, e_min_true, d_e_true, a_true, r_e_true)
    spec = extract_spectroscopic_constants("H2", "Synthetic Morse", R_grid, E_grid)

    assert np.isclose(spec.equilibrium_bond_length_A, r_e_true, atol=2e-3)
    assert np.isclose(spec.minimum_energy_Ha, e_min_true, atol=1e-3)
    assert np.isclose(spec.dissociation_energy_De_Ha, d_e_true, atol=2e-3)
    assert spec.r_squared > 0.999
    assert 4000.0 <= spec.harmonic_freq_omega_e_cm_inv <= 5500.0

    levels = compute_morse_vibrational_levels(spec, max_levels=4)
    assert len(levels) >= 3
    for nu, e_lvl, r_in, r_out in levels:
        assert r_in < spec.equilibrium_bond_length_A < r_out
        assert e_lvl > spec.minimum_energy_Ha


def test_spectroscopy_edge_cases():
    """Verify input validation on insufficient points or mismatched arrays."""
    with pytest.raises(ValueError, match="at least 5"):
        extract_spectroscopic_constants("H2", "Test", [0.5, 0.7, 1.0], [-1.0, -1.1, -0.9])
    with pytest.raises(ValueError, match="at least 5"):
        extract_spectroscopic_constants(
            "H2", "Test", [0.5, 0.7, 0.8, 1.0, 1.2], [-1.0, -1.1, -1.05, -0.9]
        )
    with pytest.raises(ValueError, match="Unsupported molecule"):
        extract_spectroscopic_constants(
            "N2", "Test", [0.8, 1.0, 1.1, 1.3, 1.6, 2.0], [-109.0, -109.4, -109.5, -109.3, -109.1, -109.0]
        )
