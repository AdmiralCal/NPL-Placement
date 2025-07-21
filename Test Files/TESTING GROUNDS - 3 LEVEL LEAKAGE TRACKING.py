# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 13:29:43 2025

@author: ctm1g20
"""

"""
Updated for 3-level Transmon Simulation
Includes: 3x3 Hamiltonian, leakage tracking, DRAG-style correction, and updated fidelity calculation with leakage penalty
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize
import math

# -----------------------------
# UPDATED SYSTEM HAMILTONIAN
# -----------------------------
def single_qubit_hamiltonian_3level(t, d_t, delta, alpha, omega_d):
    b = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    b_dag = b.conj().T
    n_op = b_dag @ b

    # Drift Hamiltonian
    H_sys = delta * n_op + (alpha / 2.0) * n_op @ (n_op - np.eye(3))

    # Control Hamiltonian
    H_ctrl = (omega_d / 2.0) * (d_t * b_dag + np.conj(d_t) * b)

    return H_sys + H_ctrl

# -----------------------------
# DRIVE PULSE WITH DRAG
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
    Q_t = Q_norm * envelope + beta * d_envelope_dt  # DRAG-style correction

    return I_t, Q_t

# -----------------------------
# HAMILTONIAN WRAPPER
# -----------------------------
def H_func_3level(t, delta, alpha, omega_d, shape, center, width, I, Q, beta=0.0):
    I_t, Q_t = normalized_drive_pulse(t, shape, center, width, I, Q, beta)
    d_t = I_t + 1j * Q_t
    return single_qubit_hamiltonian_3level(t, d_t, delta, alpha, omega_d)

# -----------------------------
# UNITARY EVOLUTION (3-LEVEL)
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
# PARAMETERS AND INITIAL STATE
# -----------------------------
t_total = 15
n_steps = 1000
times = np.linspace(0, t_total, n_steps)
delta = 0.0
alpha = -0.3
omega_d = 1.0
shape = "gaussian"
width = 1.0
center = 3 * width
I = 1.0
Q = 0.0
beta = 0.3  # DRAG coefficient

psi0 = np.array([[1], [0], [0]], dtype=complex)
target = np.array([[0], [1], [0]], dtype=complex)

# -----------------------------
# FIDELITY WITH LEAKAGE TRACKING
# -----------------------------
def gate_fidelity(center, width, times, beta):
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
    F_avg = (np.trace(M @ M.conj().T) + abs(np.trace(M))**2) / (n * (n + 1))

    # Leakage defined as population outside qubit subspace
    leakage0 = abs(psi0_t[-1][2, 0])**2
    leakage1 = abs(psi1_t[-1][2, 0])**2
    avg_leakage = (leakage0 + leakage1) / 2.0

    return np.real(F_avg), avg_leakage

# -----------------------------
# OBJECTIVE FUNCTION
# -----------------------------
def objective2(params):
    center, width = params
    F_avg, leakage = gate_fidelity(center, width, times, beta)
    fidelity_loss = -F_avg

    I_t0, _ = normalized_drive_pulse(times[0], shape, center, width, I, Q, beta)
    start_penalty = abs(I_t0)**2
    lambda_penalty = 50.0
    leakage_penalty = 5.0

    return fidelity_loss + lambda_penalty * start_penalty + leakage_penalty * leakage

# -----------------------------
# EARLY STOPPING CALLBACK
# -----------------------------
best_fid = [0]
iter_count = [0]

def stop_if_high_enough(current_guess):
    iter_count[0] += 1
    center, width = current_guess
    fid, leakage = gate_fidelity(center, width, times, beta)
    best_fid[0] = fid
    print(f"\n[Iteration {iter_count[0]}] Fidelity: {fid:.7f}, Leakage: {leakage:.4e}")
    if fid >= 0.999995:
        raise StopIteration

# -----------------------------
# OPTIMIZATION
# -----------------------------
initial_guess = [center, width]
bounds = [(0.0, t_total), (0.1, t_total / 2)]
try:
    result = minimize(objective2, initial_guess, method='Nelder-Mead',
                      bounds=bounds, callback=stop_if_high_enough,
                      options={'maxiter': 200})
except StopIteration:
    print("Early stopping: fidelity threshold reached.")

opt_center, opt_width = result.x
print("------------")
print(f"Optimal center: {opt_center:.5f}")
print(f"Optimal width:  {opt_width:.5f}")
print(f"Max fidelity:   {best_fid[0]:.10f}")

# -----------------------------
# FINAL SIMULATION
# -----------------------------
psi_t = unitary_evolution_3level(H_func_3level, times, psi0,
                                 delta, alpha, omega_d, shape, opt_center, opt_width, I, Q, beta)

populations = np.abs(psi_t)**2
plt.figure(figsize=(8, 4))
plt.plot(times, populations[:, 0], label="|0⟩")
plt.plot(times, populations[:, 1], label="|1⟩")
plt.plot(times, populations[:, 2], label="|2⟩ (leakage)", linestyle='--')
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("3-Level Qubit Evolution under Optimized DRAG Pulse")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
