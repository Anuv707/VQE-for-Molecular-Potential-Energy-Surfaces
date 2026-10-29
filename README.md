# Molecular Potential Energy Surface Mapping via the Variational Quantum Eigensolver (VQE)

**A Hybrid Quantum-Classical Framework for Simulating Chemical Bonding & Dissociation Dynamics**

---

## 1. Overview & Scientific Objectives

Accurate characterization of molecular electronic structures and atomic bonding is a foundational pursuit in computational chemistry, materials science, and drug discovery. Classical electronic structure methods face a fundamental trade-off:
- **Hartree-Fock (HF)** mean-field theory scales polynomially with system size ($\mathcal{O}(M^4)$) but neglects instantaneous electron-electron correlation, failing dramatically during covalent bond stretching and dissociation.
- **Full Configuration Interaction (FCI)** solves the electronic Schrödinger equation exactly within a chosen basis set, but exhibits exponential Hilbert-space scaling ($\mathcal{O}\!\left(\binom{2M}{N}\right)$) that rapidly becomes intractable on classical supercomputers.

This repository implements a complete hybrid quantum-classical **Variational Quantum Eigensolver (VQE)** framework (using **PennyLane** and **Qiskit Nature**) to map the **Potential Energy Surfaces (PES)** of benchmark diatomic molecules ($\text{H}_2$ and $\text{LiH}$) across continuous grids of interatomic distances $R$.

### Key Features Implemented
1. **Ab Initio Second-Quantized Electronic Hamiltonians (`src/chemistry/molecules.py`)**:
   - Constructs Born-Oppenheimer molecular Hamiltonians in the minimal Gaussian **STO-3G** basis set:
     $$\hat{H}_{\text{elec}}(R) = E_{\text{core}}(R) + \sum_{p,q} h_{pq}(R)\,\hat{a}_p^\dagger \hat{a}_q + \frac{1}{2}\sum_{p,q,r,s} g_{pqrs}(R)\,\hat{a}_p^\dagger \hat{a}_q^\dagger \hat{a}_s \hat{a}_r$$
   - Full-space $\text{H}_2$ ($2\text{ electrons in } 2\text{ spatial orbitals} \rightarrow 4\text{ spin-orbitals / qubits}$).
   - Symmetry-consistent active-space $\text{LiH}$: freezes the core $\text{Li } 1s$ orbital and automatically selects the three $\sigma$-symmetry valence molecular orbitals along the internuclear $z$-axis ($2\text{ active electrons in } 3\text{ }\sigma\text{ orbitals} \rightarrow 6\text{ spin-orbitals / qubits}$), recovering $>95\%$ of the full valence correlation energy while eliminating unphysical $\pi$-orbital active-space discontinuities.
2. **Fermion-to-Qubit Transformations (`src/chemistry/mappings.py`)**:
   - Bridges **PennyLane `qchem`** and **Qiskit Nature (`second_q.mappers`)** to support four canonical encodings:
     - **Jordan-Wigner (`jordan_wigner`)**: Occupation-basis encoding ($4\text{ qubits}, 15\text{ Pauli terms}$ for $\text{H}_2$; $6\text{ qubits}, 118\text{ Pauli terms}$ for $\text{LiH}$).
     - **Parity (`parity`)**: Cumulative parity encoding.
     - **Parity with Two-Qubit $\mathbb{Z}_2$ Symmetry Reduction (`parity_2q`)**: Exploits $\alpha/\beta$ electron-number parity conservation via `ParityMapper(num_particles=(1, 1))` to taper **2 qubits** automatically ($4 \rightarrow 2\text{ qubits}$ and $15 \rightarrow 5\text{ Pauli terms}$ for $\text{H}_2$; $6 \rightarrow 4\text{ qubits}$ and $118 \rightarrow 100\text{ Pauli terms}$ for $\text{LiH}$) with zero loss in ground-state accuracy.
     - **Bravyi-Kitaev (`bravyi_kitaev`)**: Logarithmic locality binary-tree encoding.
3. **Parameterized Quantum Circuit Ansätze (`src/algorithms/ansatze.py`)**:
   - **Chemically Motivated UCCSD (Unitary Coupled-Cluster Singles and Doubles)**:
     $$|\psi_{\text{UCCSD}}(\vec{\theta})\rangle = e^{\hat{T}(\vec{\theta}) - \hat{T}^\dagger(\vec{\theta})} |\Phi_{\text{HF}}\rangle, \qquad \hat{T} = \hat{T}_1 + \hat{T}_2$$
   - **Hardware-Efficient Ansatz (HEA)**: Configurable multi-layer ($L \in \{1, 2, 3\}$) native single-qubit $R_Y$ and two-qubit entangling circuits initialized at the Hartree-Fock reference state $|\Phi_{\text{HF}}\rangle$.
4. **Hybrid VQE Optimization & Rayleigh-Ritz Verification (`src/algorithms/vqe_runner.py`)**:
   - Minimizes the quantum expectation value in accordance with the Rayleigh-Ritz variational principle:
     $$E_0(R) \le E_{\text{VQE}}(\vec{\theta}; R) = \langle \psi(\vec{\theta}) | \hat{H}(R) | \psi(\vec{\theta}) \rangle$$
   - Benchmarks gradient-based (**SLSQP**, **L-BFGS-B**), gradient-free simplex (**COBYLA**), and stochastic NISQ (**SPSA**) classical optimizers with warm-starting across interatomic geometries.
5. **Physicochemical & Vibrational Spectroscopy (`src/analysis/spectroscopy.py`)**:
   - Fits computed PES curves to the analytical **Morse Oscillator Potential**:
     $$V_{\text{Morse}}(R) = E_{\min} + D_e \left(1 - e^{-a(R - R_e)}\right)^2$$
   - Extracts equilibrium bond length ($R_e$), ground-state minimum energy ($E_{\min}$), spectroscopic dissociation energy ($D_e$ and quantum ZPE-corrected $D_0$), harmonic force constant ($k_e$), harmonic vibrational frequency ($\omega_e$ in $\text{cm}^{-1}$), anharmonicity ($\omega_e x_e$), and quantized vibrational levels ($\nu = 0, 1, 2, 3, 4$).

---

## 2. Repository Structure

```text
Project/
├── README.md                              # Technical documentation & results summary
├── PROJECT_GUIDE.md                       # Beginner-friendly intuitive guide to the physics & code
├── requirements.txt                       # Python dependencies (PennyLane, Qiskit Nature, SciPy, etc.)
├── src/
│   ├── chemistry/
│   │   ├── molecules.py                   # H2 & LiH geometry builder, RHF SCF & molecular integrals
│   │   └── mappings.py                    # Jordan-Wigner, Parity, Parity-2Q, & Bravyi-Kitaev mappers
│   ├── algorithms/
│   │   ├── exact_solvers.py               # Exact FCI diagonalization & HF reference evaluators
│   │   ├── ansatze.py                     # UCCSD and Hardware-Efficient (HEA) quantum circuits
│   │   └── vqe_runner.py                  # Hybrid VQE solver (SLSQP, L-BFGS-B, COBYLA, SPSA) & PES scan
│   └── analysis/
│       ├── spectroscopy.py                # Morse potential fitting & vibrational spectroscopy (R_e, omega_e, D_e)
│       └── plotting.py                    # 300 DPI publication figure generator
├── experiments/
│   ├── run_h2_pes.py                      # Full 25-point H2 PES scan -> data/h2_pes_results.csv
│   ├── run_lih_pes.py                     # Full 22-point LiH PES scan -> data/lih_pes_results.csv
│   ├── run_benchmarks.py                  # Optimizer, mapping, ansatz, & spectroscopy benchmarks + plots
│   └── run_all.py                         # Master end-to-end execution pipeline
├── tests/
│   ├── test_chemistry_and_mappings.py     # Unit tests for integrals & mapping isospectrality
│   ├── test_vqe_and_ansatze.py            # Unit tests for VQE accuracy, optimizers, & Rayleigh-Ritz bound
│   └── test_spectroscopy.py               # Unit tests for Morse fitting & spectroscopic constants
├── data/
│   ├── h2_pes_results.csv                 # Computed H2 PES dataset (25 geometries)
│   ├── lih_pes_results.csv                # Computed LiH PES dataset (22 geometries)
│   ├── optimizer_convergence.csv          # Iteration-by-iteration optimizer error trajectories
│   ├── mapping_ansatz_benchmarks.csv      # Qubit, Pauli term, CNOT, and error comparison table
│   └── spectroscopic_constants.csv        # Extracted R_e, E_min, D_e, D_0, k_e, omega_e, and ZPE
└── outputs/
    └── figures/
        ├── h2_pes_comparison.png          # H2 PES curves + log-scale error vs chemical accuracy
        ├── lih_pes_comparison.png         # LiH PES curves + log-scale error vs chemical accuracy
        ├── optimizer_convergence.png      # SLSQP, L-BFGS-B, COBYLA, & SPSA convergence trajectories
        ├── mapping_and_ansatz_benchmark.png # Mapping overhead & ansatz expressibility comparison
        └── morse_spectroscopy_fit.png     # Morse oscillator fits & quantized vibrational levels
```

---

## 3. Summary of Key Quantitative Results

### 3.1 Potential Energy Surface Accuracy & Spectroscopic Constants (STO-3G)

| Molecule | Method | Equilibrium Bond Length $R_e$ (Å) | Minimum Energy $E(R_e)$ (Ha) | Dissociation Energy $D_e$ (eV) | Harmonic Frequency $\omega_e$ ($\text{cm}^{-1}$) | Max PES Error $|E - E_{\text{FCI}}|$ (Ha) | Chemical Accuracy ($\le 1.59\text{ mHa}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\text{H}_2$** | Hartree-Fock (RHF) | $0.7122$ | $-1.117506$ | $12.064$ *(overbound)* | $5501.6$ | $2.62 \times 10^{-1}$ | No ($0/25$ pts) |
| **$\text{H}_2$** | VQE (HEA, $L=2$) | $0.7348$ | $-1.137299$ | $5.990$ | $5094.5$ | $3.08 \times 10^{-5}$ | Yes ($25/25$ pts) |
| **$\text{H}_2$** | **VQE (UCCSD)** | **$0.7348$** | **$-1.137306$** | **$5.990$** | **$5093.7$** | **$4.36 \times 10^{-9}$** | **Yes ($25/25$ pts)** |
| **$\text{H}_2$** | Exact FCI (STO-3G) | $0.7348$ | $-1.137306$ | $5.990$ | $5093.7$ | $0.00$ (Reference) | Reference |
| **$\text{LiH}$** | Hartree-Fock (RHF) | $1.5107$ | $-7.863381$ | $7.330$ *(overbound)* | $1899.9$ | $1.19 \times 10^{-1}$ | No ($0/22$ pts) |
| **$\text{LiH}$** | VQE (HEA, $L=2$) | $1.5489$ | $-7.881316$ | $3.883$ | $1706.3$ | $9.69 \times 10^{-3}$ | Near-$R_e$ ($15/22$ pts) |
| **$\text{LiH}$** | **VQE (UCCSD)** | **$1.5502$** | **$-7.881463$** | **$3.442$** | **$1688.2$** | **$7.89 \times 10^{-9}$** | **Yes ($22/22$ pts)** |
| **$\text{LiH}$** | Exact FCI (STO-3G) | $1.5502$ | $-7.881463$ | $3.442$ | $1688.2$ | $0.00$ (Reference) | Reference |

- **Why Hartree-Fock Fails at Dissociation**: Because Restricted Hartree-Fock forces both valence electrons into the same spatial bonding orbital $\sigma_g$, stretching the bond to $R \to \infty$ forces an unphysical $50\%$ ionic ($\text{H}^+\text{H}^-$) + $50\%$ covalent ($\text{H}^\bullet\text{H}^\bullet$) superposition, roughly doubling the apparent dissociation energy $D_e$ ($12.06\text{ eV}$ vs $5.99\text{ eV}$ for $\text{H}_2$; $7.33\text{ eV}$ vs $3.44\text{ eV}$ for $\text{LiH}$).
- **How VQE (UCCSD) Solves This**: By applying the double-excitation operator $\hat{a}_{\sigma_u \uparrow}^\dagger \hat{a}_{\sigma_u \downarrow}^\dagger \hat{a}_{\sigma_g \downarrow} \hat{a}_{\sigma_g \uparrow}$, VQE smoothly entangles the $|\sigma_g^2\rangle$ and $|\sigma_u^2\rangle$ configurations, recovering **$100.00\%$ of the FCI correlation energy** across the entire bond-breaking curve with maximum errors below $10^{-8}\text{ Ha}$—more than **five orders of magnitude better than the chemical accuracy threshold ($1.59 \times 10^{-3}\text{ Ha}$)**.

### 3.2 Fermion-to-Qubit Mapping Comparison

| Molecule | Mapping | Qubits | Pauli Terms in $\hat{H}$ | Mean Pauli Weight | Exact FCI Energy (Ha) | VQE (UCCSD) Error $|E - E_{\text{FCI}}|$ (Ha) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$\text{H}_2$** ($0.735$ Å) | Jordan-Wigner | $4$ | $15$ | $2.13$ | $-1.137306$ | $2.28 \times 10^{-13}$ |
| **$\text{H}_2$** ($0.735$ Å) | Parity | $4$ | $15$ | $2.27$ | $-1.137306$ | $2.25 \times 10^{-13}$ |
| **$\text{H}_2$** ($0.735$ Å) | **Parity (2Q Tapered)** | **$2$** | **$5$** | **$1.20$** | **$-1.137306$** | **$1.19 \times 10^{-13}$** |
| **$\text{H}_2$** ($0.735$ Å) | Bravyi-Kitaev | $4$ | $15$ | $2.40$ | $-1.137306$ | $2.23 \times 10^{-13}$ |
| **$\text{LiH}$** ($1.546$ Å) | Jordan-Wigner | $6$ | $118$ | $3.66$ | $-7.881460$ | $4.21 \times 10^{-10}$ |
| **$\text{LiH}$** ($1.546$ Å) | Parity | $6$ | $118$ | $3.75$ | $-7.881460$ | $1.74 \times 10^{-9}$ |
| **$\text{LiH}$** ($1.546$ Å) | **Parity (2Q Tapered)** | **$4$** | **$100$** | **$2.80$** | **$-7.881460$** | **$1.74 \times 10^{-9}$** |
| **$\text{LiH}$** ($1.546$ Å) | Bravyi-Kitaev | $6$ | $118$ | $3.93$ | $-7.881460$ | $1.75 \times 10^{-9}$ |

### 3.3 Ansatz Expressibility vs. Two-Qubit Gate Depth (Jordan-Wigner)

| Molecule | Ansatz | Variational Params | CNOT Gates | Circuit Depth | VQE Error $|E - E_{\text{FCI}}|$ (Ha) | Correlation Energy Recovered |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$\text{H}_2$** ($0.735$ Å) | HEA ($L=1$) | $3$ | $10$ | $7$ | $7.67 \times 10^{-4}$ | $96.22\%$ |
| **$\text{H}_2$** ($0.735$ Å) | HEA ($L=2$) | $6$ | $20$ | $13$ | $6.95 \times 10^{-6}$ | $99.97\%$ |
| **$\text{H}_2$** ($0.735$ Å) | HEA ($L=3$) | $9$ | $30$ | $19$ | $9.80 \times 10^{-11}$ | $100.00\%$ |
| **$\text{H}_2$** ($0.735$ Å) | **UCCSD** | **$3$** | **$21$** | **$5$** | **$2.28 \times 10^{-13}$** | **$100.00\%$** |
| **$\text{LiH}$** ($1.546$ Å) | HEA ($L=1$) | $5$ | $14$ | $7$ | $1.54 \times 10^{-3}$ | $91.61\%$ |
| **$\text{LiH}$** ($1.546$ Å) | HEA ($L=2$) | $10$ | $28$ | $13$ | $1.45 \times 10^{-4}$ | $99.21\%$ |
| **$\text{LiH}$** ($1.546$ Å) | HEA ($L=3$) | $15$ | $42$ | $19$ | $1.72 \times 10^{-5}$ | $99.91\%$ |
| **$\text{LiH}$** ($1.546$ Å) | **UCCSD** | **$8$** | **$68$** | **$10$** | **$4.21 \times 10^{-10}$** | **$100.00\%$** |

---

## 4. Installation & Usage

### 4.1 Environment Setup
Requires Python 3.10+ (tested on Windows with Python 3.12):
```powershell
pip install -r requirements.txt
```

### 4.2 Running the Experiments & Generating Figures
Run individual experiments or the complete end-to-end pipeline:
```powershell
# Run H2 Potential Energy Surface scan (saves data/h2_pes_results.csv)
python experiments/run_h2_pes.py

# Run LiH Potential Energy Surface scan (saves data/lih_pes_results.csv)
python experiments/run_lih_pes.py

# Run optimizer, mapping, ansatz, & spectroscopy benchmarks and generate all 5 PNG plots
python experiments/run_benchmarks.py

# Or run the entire pipeline end-to-end:
python experiments/run_all.py
```

### 4.3 Running the Automated Test Suite
```powershell
python -m pytest tests/ -v
```
