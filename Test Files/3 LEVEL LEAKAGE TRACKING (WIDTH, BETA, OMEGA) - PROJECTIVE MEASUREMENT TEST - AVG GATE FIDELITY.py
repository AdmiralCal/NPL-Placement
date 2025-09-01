# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 14:28:31 2025

@author: ctm1g20
"""

"""
Updated for 3-level Transmon Simulation
Includes: 3x3 Hamiltonian, leakage tracking, DRAG-style correction,
and updated fidelity calculation with leakage penalty
Now includes optimization over width, beta, and omega_d.
Converted from ℓ = 1 natural units to real physical units:
- Time in nanoseconds (ns)
- Frequencies in MHz
- ℓ = 6.582119569e-7 MHz·ns
"""
from measurement_integration import (
    measure_probs_from_statevec,
    simulate_shots,
    empirical_probs_and_ci,
    measurement_reward,
    optimize_with_measurements,
    plot_counts_vs_probs
)

from single_qubit_tom import (
    hadamard,
    sqrt_z_dag,
    projective_meas,
    single_qubit_tomography,
    bloch_vec_and_density_mat,
    reduced_rho_from_psi
)

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, sqrtm
from scipy.optimize import minimize
import math


# -----------------------------
# SYSTEM HAMILTONIAN (3-LEVEL)
# -----------------------------
def single_qubit_hamiltonian_3level(t, d_t, delta, alpha, omega_d):
    b = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    b_dag = b.conj().T
    n_op = b_dag @ b

    H_sys = delta * n_op + (alpha / 2.0) * (b_dag @ b_dag @ b @ b)
    H_ctrl = (omega_d / 2.0) * (d_t * b_dag + np.conj(d_t) * b)

    return H_sys + H_ctrl

# -----------------------------
# UPDATED DRIVE PULSE FUNCTION
# -----------------------------
def normalized_drive_pulse(t, shape, center, width, I, Q, beta=0.0):
    if shape == "gaussian":
        envelope = np.exp(-((t - center) ** 2) / (2 * width ** 2))
        d_envelope_dt = -((t - center) / (width ** 2)) * envelope
    else:
        raise ValueError(f"Unknown shape '{shape}'")

    max_component = max(abs(I), abs(Q))
    I_norm = I / max_component
    Q_norm = Q / max_component
    I_t = envelope * I_norm
    Q_t = -beta * d_envelope_dt * I_norm

    return I_t, Q_t

# -----------------------------
# HAMILTONIAN FUNCTION WRAPPER
# -----------------------------
def H_func_3level(t, delta, alpha, omega_d, shape, center, width, I, Q, beta=0.0):
    I_t, Q_t = normalized_drive_pulse(t, shape, center, width, I, Q, beta)
    d_t = I_t + 1j * Q_t
    return single_qubit_hamiltonian_3level(t, d_t, delta, alpha, omega_d)

# -----------------------------
# UNITARY EVOLUTION FOR 3-LEVEL
# -----------------------------
def unitary_evolution_3level(H_func, times, psi0,
                              delta, alpha, omega_d, shape, center, width, I, Q, beta=0.0):
    dt = times[1] - times[0]
    psi_t = [psi0]
    psi = psi0.copy()
    for t in times[:-1]:
        H = H_func(t, delta, alpha, omega_d, shape, center, width, I, Q, beta)
        U = expm(-1j * H * dt)
        psi = U @ psi
        psi_t.append(psi)
    return np.array(psi_t)

# -----------------------------
# PROJECTIVE MEASUREMENT FUNCTION
# -----------------------------
def measure_probabilities(psi):
    """Return probability of being in |0>, |1>, |2> states from final wavefunction."""
    probs = np.abs(psi.flatten()) ** 2
    return probs

# -----------------------------
# PARAMETERS AND INITIAL STATE
# -----------------------------
t_total = 10  # ns
times = np.linspace(0, t_total, 1000)
delta = 0.0
alpha = -3  # rad/ns
omega_d = 1  # rad/ns
shape = "gaussian"
width = 0.5  # ns
center = 3 * width
beta = 0
I, Q = 1.0, 0.0
n_shots = 1000
psi0 = np.array([[1], [0], [0]], dtype=complex)

# -----------------------------
# OBJECTIVE FUNCTION FOR OPTIMIZATION
# -----------------------------

def matrix_fidelity(rho_actual, rho_measured):
    """
    Compute Uhlmann fidelity F(ρ,σ) = [Tr(√(√ρ σ √ρ))]^2
    Ensures Hermiticity to remove small numerical imaginary parts.
    """
    rho_actual = (rho_actual + rho_actual.conj().T) / 2
    rho_measured = (rho_measured + rho_measured.conj().T) / 2

    sqrt_rho = sqrtm(rho_actual)
    inner = sqrt_rho @ rho_measured @ sqrt_rho
    sqrt_inner = sqrtm(inner)

    return np.real((np.trace(sqrt_inner))**2)


def tomography_state_fidelity(psi_init, U_target, width, beta, omega_d, times):
    """
    Fidelity for a *single input state*:
    - Evolve psi_init under pulse
    - Perform simulated tomography (with n_shots projective measurements)
    - Reconstruct measured density matrix rho_m
    - Compare with ideal target output under U_target
    """
    center = 3 * width

    # Simulate pulse evolution
    psi_t = unitary_evolution_3level(
        H_func_3level, times, psi_init,
        delta, alpha, omega_d, shape, center, width, I, Q, beta
    )
    psi_final = psi_t[-1].flatten()[:2]  # restrict to {|0>,|1>} subspace

    # Simulate tomography from projective measurements
    tomography_results = single_qubit_tomography(psi_final, n_shots)
    r_vec, rho_m = bloch_vec_and_density_mat(tomography_results, n_shots)

    # Ideal target output (density matrix after X gate)
    rho_in = psi_init[:2] @ psi_init[:2].conj().T
    rho_target = U_target @ rho_in @ U_target.conj().T

    return matrix_fidelity(rho_target, rho_m)


def tomography_gate_fidelity(width, beta, omega_d, times):
    """
    Compute average gate fidelity of implemented operation vs. target X gate,
    using tomography with projective measurement simulation.
    """
    U_target = np.array([[0, 1],
                         [1, 0]], dtype=complex)  # Pauli-X

    # Tomography input states (span Bloch sphere)
    psi0 = np.array([[1], [0], [0]], dtype=complex)  # |0>
    #psi1 = np.array([[0], [1], [0]], dtype=complex)  # |1>
    psip = (np.array([[1], [1], [0]], dtype=complex)) / np.sqrt(2)   # |+>
    #psim = (np.array([[1], [-1], [0]], dtype=complex)) / np.sqrt(2)  # |->
    psii = (np.array([[1], [1j], [0]], dtype=complex)) / np.sqrt(2)  # |i>
    #psiminusi = (np.array([[1], [-1j], [0]], dtype=complex)) / np.sqrt(2)  # |-i>

    #states = [psi0, psi1, psip, psim, psii, psiminusi]
    states = [psi0, psip, psii]

    fidelities = []
    for psi_in in states:
        F = tomography_state_fidelity(psi_in, U_target, width, beta, omega_d, times)
        fidelities.append(F)

    return np.mean(fidelities)


# ===============================================
# OBJECTIVE FUNCTION FOR OPTIMISER
# ===============================================
def objective_tomography(params, times):
    """
    Objective for optimizer: minimize -F, where
    F = tomography-based average gate fidelity.
    """
    width, beta, omega_d = params
    F = tomography_gate_fidelity(width, beta, omega_d, times)
    return -F


best_fid = [0]
iter_count = [0]
best_solution = [0, 0, 0]

def stop_if_high_enough(current_guess):
    iter_count[0] += 1
    width, beta, omega_d = current_guess
    best_solution = [width, beta, omega_d]
    #fid = tomography_fidelity(width, beta, omega_d, times)
    fid = tomography_gate_fidelity(width, beta, omega_d, times)
    best_fid[0] = fid
    print(f"\n[Iteration {iter_count[0]}] Fidelity: {fid:.7f}")
    if fid >= 0.999995:
        raise StopIteration
        


# -----------------------------
# OPTIMIZATION BLOCK
# -----------------------------
optim = "Y"
if optim == "Y":
    print("Optimisation ON")

    initial_guess = [width, beta, omega_d]
    bounds = [(0.1, t_total / 2), (-3.0, 3.0), (0, 5.0)]
    try:
        result = minimize(objective_tomography, initial_guess, args=(times), method='Nelder-Mead',
                          bounds=bounds, callback=stop_if_high_enough,
                          options={'maxiter': 200})
    except StopIteration:
        print("Early stopping: fidelity threshold reached.")
        
   # result = minimize(objective, initial_guess, method='Nelder-Mead', bounds=bounds)

    opt_width, opt_beta, opt_omega_d = result.x
    opt_center = 3 * opt_width
    max_fidelity = best_fid[0]
    
    print("------------")
    print(f"Optimal width:     {opt_width:.5f}")
    print(f"Optimal beta:      {opt_beta:.5f}")
    print(f"Optimal omega_d:   {opt_omega_d:.5f}")
    print(f"Max Fidelity:      {max_fidelity:.8f}")    
    
    # Final evolution
    psi_t = unitary_evolution_3level(H_func_3level, times, psi0,
                                     delta, alpha, opt_omega_d, shape, opt_center, opt_width, I, Q, opt_beta)

    final_psi = psi_t[-1]
    probs = measure_probs_from_statevec(final_psi)
    counts = simulate_shots(probs, 10000)
    plot_counts_vs_probs(counts, probs)

    empirical_probs_and_ci(counts, alpha=0.05)
    # -----------------------------
    # PROJECTIVE MEASUREMENT
    # -----------------------------
    #final_psi = psi_t[-1]
    #probs = measure_probs_from_statevec(final_psi)
    #counts = simulate_shots(probs, 10000)
    #plot_counts_vs_probs(counts, probs)
    #empirical_probs_and_ci(counts, alpha=0.05)
    #probabilities = measure_probabilities(final_psi)
    #print("------------")
    #print(f"P(|0⟩): {probabilities[0]:.4f}")
    #print(f"P(|1⟩): {probabilities[1]:.4f}")
    #print(f"P(|2⟩): {probabilities[2]:.4f} (Leakage)")

    # -----------------------------
    # PLOTTING STATE POPULATIONS
    # -----------------------------
    populations = np.abs(psi_t) ** 2
    plt.figure(figsize=(8, 4))
    plt.plot(times, populations[:, 0], label="|0⟩")
    plt.plot(times, populations[:, 1], label="|1⟩")
    plt.plot(times, populations[:, 2], label="|2⟩ (leakage)", linestyle="--")
    plt.xlabel("Time (ns)")
    plt.ylabel("Population")
    plt.title("3-Level Qubit Evolution (Optimized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Get ideal probabilities
    probs = np.abs(final_psi.flatten())**2
    
    # Simulate projective measurements
    n_shots = 100000
    outcomes = np.random.choice([0, 1, 2], size=n_shots, p=probs)
    counts = np.bincount(outcomes, minlength=3)
    empirical_probs = counts / n_shots
    
    print("------------")
    print("Theoretical probabilities:", probs)
    print("Empirical probabilities:", empirical_probs)
    print("Counts:", counts)
    
    rho_qubit = reduced_rho_from_psi(psi_t)
    
    psi_final = psi_t[-1].flatten()[:2]
    tomography_results = single_qubit_tomography(psi_final, n_shots)
    r_vec, rho_m = bloch_vec_and_density_mat(tomography_results, n_shots)
    
    
    # MEASURED DENSITY MATRIX (2X2): |0>, |1> SUBSPACE
    # COMPONENTS: c0 = <0|psi>, c1 = <1|psi>, c2 = <2|psi>

    # Keep p_leak around to report leakage separately
    
   # tom_fid = tomography_fidelity(opt_width, opt_beta, opt_omega_d, times)

    print("------------")
    print(f"Measured Bloch vector: {r_vec}")
    #np.set_printoptions(precision=5)
    print(f"Reconstructed Density Matrix:\n{rho_m}\n")
    print(f"Measured Density Matrix:\n{rho_qubit}\n")
    print(f"Target Density Matrix:\n{np.array([[0, 0], [0, 1]], dtype=complex)}\n")
    print(f"Fidelity between Target and Target Density Matrices: {max_fidelity:.10f}\n")
    
