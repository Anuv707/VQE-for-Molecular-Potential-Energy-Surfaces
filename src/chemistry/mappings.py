"""
Fermion-to-Qubit transformations (Jordan-Wigner, Parity, Parity with 2-Qubit Z2 Tapering,
and Bravyi-Kitaev) bridging PennyLane qchem and Qiskit Nature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pennylane as qml
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    ParityMapper,
)

from .molecules import MolecularIntegralData, compute_molecular_integrals


SUPPORTED_MAPPINGS = ("jordan_wigner", "parity", "parity_2q", "bravyi_kitaev")


@dataclass
class MappedHamiltonian:
    """Multi-qubit Pauli Hamiltonian produced by a fermion-to-qubit transformation."""

    molecule_name: str
    bond_length: float
    mapping_name: str
    n_qubits: int
    n_pauli_terms: int
    pl_hamiltonian: qml.operation.Operator
    qiskit_pauli_op: SparsePauliOp
    core_constant: float
    hf_energy: float
    hf_bitstring: np.ndarray
    mean_pauli_weight: float
    max_pauli_weight: int
    pauli_weight_histogram: Dict[int, int]
    _cached_matrix: Optional[np.ndarray] = None

    def to_matrix(self) -> np.ndarray:
        """Return the exact dense (2^n x 2^n) Hermitian matrix of the total Hamiltonian."""
        if self._cached_matrix is None:
            self._cached_matrix = np.asarray(
                qml.matrix(self.pl_hamiltonian, wire_order=list(range(self.n_qubits))),
                dtype=complex,
            )
        return self._cached_matrix


def _sparse_pauli_to_pennylane(
    pauli_op: SparsePauliOp, core_constant: float
) -> qml.operation.Operator:
    """
    Convert a Qiskit SparsePauliOp + scalar core_constant into a PennyLane Hamiltonian
    with consistent big-endian wire ordering (wire 0..n-1).
    """
    n_qubits = pauli_op.num_qubits
    coeffs: List[float] = [float(core_constant)]
    ops: List[qml.operation.Operator] = [qml.Identity(0)]

    for pauli_str, coeff in pauli_op.simplify().to_list():
        c_real = float(np.real(coeff))
        if abs(c_real) < 1e-14:
            continue
        # Qiskit string is little-endian (q_{n-1} ... q_0); reverse to map onto wires 0..n-1
        factors = []
        for wire_idx, char in enumerate(reversed(pauli_str)):
            if char == "X":
                factors.append(qml.PauliX(wire_idx))
            elif char == "Y":
                factors.append(qml.PauliY(wire_idx))
            elif char == "Z":
                factors.append(qml.PauliZ(wire_idx))
        if len(factors) == 0:
            coeffs[0] += c_real
        elif len(factors) == 1:
            coeffs.append(c_real)
            ops.append(factors[0])
        else:
            coeffs.append(c_real)
            ops.append(qml.prod(*factors))

    return qml.Hamiltonian(coeffs, ops)


def _compute_pauli_weight_stats(
    pl_hamiltonian: qml.operation.Operator, n_qubits: int
) -> Tuple[int, float, int, Dict[int, int]]:
    """Compute Pauli term count, mean Pauli weight (locality), max weight, and weight histogram."""
    coeffs, ops = pl_hamiltonian.terms()
    hist: Dict[int, int] = {}
    valid_terms = 0
    total_weight = 0
    max_weight = 0

    for c, op in zip(coeffs, ops):
        if abs(float(np.real(c))) < 1e-14:
            continue
        valid_terms += 1
        # Determine how many non-identity Pauli operators are in this term
        if isinstance(op, qml.Identity):
            w = 0
        else:
            w = len(op.wires)
        hist[w] = hist.get(w, 0) + 1
        total_weight += w
        if w > max_weight:
            max_weight = w

    mean_weight = float(total_weight) / max(valid_terms, 1)
    return valid_terms, mean_weight, max_weight, dict(sorted(hist.items()))


def build_qubit_hamiltonian(
    integral_data: MolecularIntegralData,
    mapping: str = "jordan_wigner",
) -> MappedHamiltonian:
    """
    Transform second-quantized molecular integrals into a multi-qubit Pauli Hamiltonian
    using both PennyLane qchem and Qiskit Nature mappers.

    Parameters
    ----------
    integral_data : MolecularIntegralData
        Precomputed RHF molecular integrals and active-space specification.
    mapping : str
        One of 'jordan_wigner', 'parity', 'parity_2q' (two-qubit Z2 symmetry reduced),
        or 'bravyi_kitaev'.
    """
    m_clean = mapping.strip().lower().replace("-", "_")
    if m_clean not in SUPPORTED_MAPPINGS:
        raise ValueError(
            f"Unsupported mapping '{mapping}'. Choose from {SUPPORTED_MAPPINGS}."
        )

    spec = integral_data.spec
    core_arr = np.array([integral_data.core_constant], dtype=float)
    one_body = integral_data.one_body_integrals
    two_body_phys = integral_data.two_body_integrals_phys
    two_body_chem = integral_data.two_body_integrals_chem

    # Build Qiskit Nature second-quantized ElectronicEnergy operator
    elec_energy = ElectronicEnergy.from_raw_integrals(one_body, two_body_chem)
    fermionic_op_qiskit = elec_energy.second_q_op()

    n_alpha = spec.active_electrons // 2
    n_beta = spec.active_electrons - n_alpha

    if m_clean == "jordan_wigner":
        qiskit_mapper = JordanWignerMapper()
        qiskit_op = qiskit_mapper.map(fermionic_op_qiskit).simplify()
        ferm_pl = qml.qchem.fermionic_observable(
            core_arr, one_body, two_body_phys, cutoff=1e-12
        )
        pl_ham = qml.qchem.qubit_observable(ferm_pl, mapping="jordan_wigner")
        n_qubits = spec.n_qubits
        hf_bits = np.asarray(
            qml.qchem.hf_state(spec.active_electrons, n_qubits, basis="occupation"),
            dtype=int,
        )

    elif m_clean == "parity":
        qiskit_mapper = ParityMapper()
        qiskit_op = qiskit_mapper.map(fermionic_op_qiskit).simplify()
        ferm_pl = qml.qchem.fermionic_observable(
            core_arr, one_body, two_body_phys, cutoff=1e-12
        )
        pl_ham = qml.qchem.qubit_observable(ferm_pl, mapping="parity")
        n_qubits = spec.n_qubits
        hf_bits = np.asarray(
            qml.qchem.hf_state(spec.active_electrons, n_qubits, basis="parity"),
            dtype=int,
        )

    elif m_clean == "parity_2q":
        # Qiskit Nature ParityMapper with two-qubit Z2 symmetry reduction (tapering 2 qubits)
        qiskit_mapper = ParityMapper(num_particles=(n_alpha, n_beta))
        qiskit_op = qiskit_mapper.map(fermionic_op_qiskit).simplify()
        n_qubits = qiskit_op.num_qubits
        pl_ham = _sparse_pauli_to_pennylane(qiskit_op, integral_data.core_constant)
        # Identify the exact computational basis bitstring corresponding to the RHF energy
        mat_2q = np.asarray(
            qml.matrix(pl_ham, wire_order=list(range(n_qubits))), dtype=complex
        )
        diag_real = np.real(np.diag(mat_2q))
        hf_idx = int(np.argmin(np.abs(diag_real - integral_data.hf_energy)))
        hf_bits = np.array(
            [(hf_idx >> (n_qubits - 1 - w)) & 1 for w in range(n_qubits)], dtype=int
        )

    elif m_clean == "bravyi_kitaev":
        qiskit_mapper = BravyiKitaevMapper()
        qiskit_op = qiskit_mapper.map(fermionic_op_qiskit).simplify()
        ferm_pl = qml.qchem.fermionic_observable(
            core_arr, one_body, two_body_phys, cutoff=1e-12
        )
        pl_ham = qml.qchem.qubit_observable(ferm_pl, mapping="bravyi_kitaev")
        n_qubits = spec.n_qubits
        hf_bits = np.asarray(
            qml.qchem.hf_state(spec.active_electrons, n_qubits, basis="bravyi_kitaev"),
            dtype=int,
        )

    n_terms, mean_w, max_w, hist = _compute_pauli_weight_stats(pl_ham, n_qubits)

    return MappedHamiltonian(
        molecule_name=spec.name,
        bond_length=spec.bond_length_angstrom,
        mapping_name=m_clean,
        n_qubits=n_qubits,
        n_pauli_terms=n_terms,
        pl_hamiltonian=pl_ham,
        qiskit_pauli_op=qiskit_op,
        core_constant=integral_data.core_constant,
        hf_energy=integral_data.hf_energy,
        hf_bitstring=hf_bits,
        mean_pauli_weight=mean_w,
        max_pauli_weight=max_w,
        pauli_weight_histogram=hist,
    )


def compare_mappings(
    molecule_name: str,
    bond_length: float,
    basis_name: str = "sto-3g",
    mappings: Tuple[str, ...] = SUPPORTED_MAPPINGS,
) -> Dict[str, MappedHamiltonian]:
    """
    Build and compare all requested fermion-to-qubit mappings for a given molecule and bond length.
    """
    integrals = compute_molecular_integrals(
        molecule_name=molecule_name,
        bond_length=bond_length,
        basis_name=basis_name,
        active_space=True,
    )
    return {m: build_qubit_hamiltonian(integrals, mapping=m) for m in mappings}
