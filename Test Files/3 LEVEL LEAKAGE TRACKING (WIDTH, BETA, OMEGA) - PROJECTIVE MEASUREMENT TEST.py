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
    optimize_with_measurements
)
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize
import math

# -----------------------------
# CONSTANTS AND UNIT CONVERSION
# -----------------------------

P0 = np.array([[1, 0, 0],
               [0, 0, 0],
               [0, 0, 0]])
P1 = np.array([[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]])
P2 = np.array([[0, 0, 0],
               [0, 0, 0],
               [0, 0, 1]])

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
    Q_t = -beta * d_envelope_dt

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
omega_d = 1.0  # rad/ns
shape = "gaussian"
width = 1  # ns
center = 3 * width
beta = 0.0
I, Q = 1.0, 0.0
psi0 = np.array([[1], [0], [0]], dtype=complex)

# -----------------------------
# OBJECTIVE FUNCTION FOR OPTIMIZATION
# -----------------------------
def gate_fidelity(width, beta, omega_d, times):
    center = 3 * width
    psi0 = np.array([[1], [0], [0]], dtype=complex)
    psi1 = np.array([[0], [1], [0]], dtype=complex)

    psi0_t = unitary_evolution_3level(H_func_3level, times, psi0,
                                      delta, alpha, omega_d, shape, center, width, I, Q, beta)
    psi1_t = unitary_evolution_3level(H_func_3level, times, psi1,
                                      delta, alpha, omega_d, shape, center, width, I, Q, beta)

    U_sim = np.hstack([psi0_t[-1][:2], psi1_t[-1][:2]])
    U_target = np.array([[0, 1], [1, 0]], dtype=complex)

    M = U_sim @ U_target.conj().T
    n = 2
    F_avg = (np.trace(M @ M.conj().T) + abs(np.trace(M)) ** 2) / (n * (n + 1))
    return np.real(F_avg)

# -----------------------------
# OPTIMIZATION BLOCK
# -----------------------------
optim = "Y"
if optim == "Y":
    print("Optimisation ON")
    def objective(params):
        width, beta, omega_d = params
        return -gate_fidelity(width, beta, omega_d, times)

    initial_guess = [width, beta, omega_d]
    bounds = [(0.1, t_total / 2), (-3.0, 3.0), (0, 5.0)]

    result = minimize(objective, initial_guess, method='Nelder-Mead', bounds=bounds)

    opt_width, opt_beta, opt_omega_d = result.x
    opt_center = 3 * opt_width
    print("------------")
    print(f"Optimal width:     {opt_width:.5f}")
    print(f"Optimal beta:      {opt_beta:.5f}")
    print(f"Optimal omega_d:   {opt_omega_d:.5f}")
    print(f"Max Fidelity:      {-result.fun:.8f}")

    # Final evolution
    psi_t = unitary_evolution_3level(H_func_3level, times, psi0,
                                     delta, alpha, opt_omega_d, shape, opt_center, opt_width, I, Q, opt_beta)

    # -----------------------------
    # PROJECTIVE MEASUREMENT
    # -----------------------------
    final_psi = psi_t[-1]
    probabilities = measure_probabilities(final_psi)
    print("------------")
    print(f"P(|0⟩): {probabilities[0]:.4f}")
    print(f"P(|1⟩): {probabilities[1]:.4f}")
    print(f"P(|2⟩): {probabilities[2]:.4f} (Leakage)")

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