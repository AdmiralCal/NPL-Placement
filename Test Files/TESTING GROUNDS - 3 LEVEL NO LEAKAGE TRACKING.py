# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 10:32:36 2025

@author: ctm1g20
"""

# -*- coding: utf-8 -*-
"""
Updated for 3-level Transmon Simulation
Includes: 3x3 Hamiltonian, leakage tracking, and updated fidelity calculation
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
    # Ladder operators in 3-level basis
    b = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    b_dag = b.conj().T
    n_op = b_dag @ b

    # Drift Hamiltonian: δn + (α/2) n(n-1)
    H_sys = delta * n_op + (alpha / 2.0) * (b_dag @ b_dag @ b @ b)

    # Drive Hamiltonian
    H_ctrl = (omega_d / 2.0) * (d_t * b_dag + np.conj(d_t) * b)

    return H_sys + H_ctrl

# -----------------------------
# UPDATED DRIVE PULSE FUNCTION
# -----------------------------
def normalized_drive_pulse(t, shape, center, width, I, Q):
    if shape == "gaussian":
        envelope = np.exp(-((t - center) ** 2) / (2 * width ** 2))
        d_envelope_dt = -((t - center) / (width ** 2)) * envelope
    else:
        raise ValueError(f"Unknown shape '{shape}'")

    max_component = max(abs(I), abs(Q))
    I_norm = I / max_component
    Q_norm = Q / max_component
    I_t = envelope * I_norm
    Q_t = -d_envelope_dt * Q_norm  # DRAG-style Q pulse (optional)

    return I_t, Q_t

# -----------------------------
# HAMILTONIAN FUNCTION WRAPPER
# -----------------------------
def H_func_3level(t, delta, alpha, omega_d, shape, center, width, I, Q):
    I_t, Q_t = normalized_drive_pulse(t, shape, center, width, I, Q)
    d_t = I_t + 1j * Q_t
    return single_qubit_hamiltonian_3level(t, d_t, delta, alpha, omega_d)

# -----------------------------
# UNITARY EVOLUTION FOR 3-LEVEL
# -----------------------------
def unitary_evolution_3level(H_func, times, psi0,
                              delta, alpha, omega_d, shape, center, width, I, Q):
    dt = times[1] - times[0]
    psi_t = [psi0]
    psi = psi0.copy()
    for t in times[:-1]:
        H = H_func(t, delta, alpha, omega_d, shape, center, width, I, Q)
        U = expm(-1j * H * dt)
        psi = U @ psi
        psi_t.append(psi)
    return np.array(psi_t)

# -----------------------------
# PARAMETERS AND INITIAL STATE
# -----------------------------
t_total = 10
n_steps = 1000
times = np.linspace(0, t_total, n_steps)
delta = 0.0
alpha = 0.0  #-0.3
omega_d = 1.0
shape = "gaussian"
width = 1
center = 3 * width
I = 1.0
Q = 0

psi0 = np.array([[1], [0], [0]], dtype=complex)
target = np.array([[0], [1], [0]], dtype=complex)  # Target: |0> -> |1>

# -----------------------------
# UPDATED FIDELITY FUNCTION (Eq.2)
# -----------------------------
def gate_fidelity(width, times):
    center = 3 * width  # FIXED center
    psi0 = np.array([[1], [0], [0]], dtype=complex)
    psi1 = np.array([[0], [1], [0]], dtype=complex)

    psi0_t = unitary_evolution_3level(H_func_3level, times, psi0,
                                      delta, alpha, omega_d, shape, center, width, I, Q)
    psi1_t = unitary_evolution_3level(H_func_3level, times, psi1,
                                      delta, alpha, omega_d, shape, center, width, I, Q)

    U_sim = np.hstack([psi0_t[-1][:2], psi1_t[-1][:2]])
    U_target = np.array([[0, 1], [1, 0]], dtype=complex)

    M = U_sim @ U_target.conj().T
    n = 2
    F_avg = (np.trace(M @ M.conj().T) + abs(np.trace(M))**2) / (n * (n + 1))
    return np.real(F_avg)

# -----------------------------
# OBJECTIVE FUNCTION WITH PENALTY
# -----------------------------
def objective2(params):
    width = params
    center = 3 * width  # FIXED center
    F_avg = gate_fidelity(width, times)
    fidelity_loss = -F_avg

    I_t0, _ = normalized_drive_pulse(times[0], shape, center, width, I, Q)
    start_penalty = abs(I_t0)**2
    lambda_penalty = 5.0

    return fidelity_loss
    #return fidelity_loss + lambda_penalty * start_penalty

# -----------------------------
# EARLY STOPPING CALLBACK
# -----------------------------
best_fid = [0]
iter_count = [0]

def stop_if_high_enough(current_guess):
    iter_count[0] += 1
    width = current_guess
    fid = gate_fidelity(width, times)
    best_fid[0] = fid
    print(f"\n[Iteration {iter_count[0]}] Fidelity: {fid:.7f}")
    if fid >= 0.999995:
        raise StopIteration

# -----------------------------
# OPTIMIZATION CALL
# -----------------------------
optim = "Y"
initial_guess = width
bounds = [(0.1, t_total / 2)]
if optim == "Y":
    print("Optimisation ON")
    try:
        result = minimize(objective2, initial_guess, method='Nelder-Mead',
                          bounds=bounds, callback=stop_if_high_enough,
                          options={'maxiter': 200})
    except StopIteration:
        print("Early stopping: fidelity threshold reached.")
    
    # Extract best parameters
    opt_width = result.x
    opt_center = 3 * opt_width
    max_fidelity = best_fid[0]
    print("------------")
    print(f"Optimal center:    {opt_center.item():.5f}")
    print(f"Optimal width:     {opt_width.item():.5f}")
    print(f"Maximum fidelity:  {max_fidelity:.10f}")
    
    # -----------------------------
    # FINAL SIMULATION AND PLOTS
    # -----------------------------
    psi_t = unitary_evolution_3level(H_func_3level, times, psi0,
                                     delta, alpha, omega_d, shape, opt_center, opt_width, I, Q)
    I_vals, Q_vals = normalized_drive_pulse(times, shape, opt_center, opt_width, I, Q)
    
    plt.figure(figsize=(8, 4))
    plt.plot(times, I_vals, label="I(t): Real Component (X)", color='blue')
    plt.plot(times, Q_vals, label="Q(t): Imaginary Component (Y)", color='orange')
    plt.axhline(1, linestyle='--', color='gray', alpha=0.4)
    plt.axhline(0, linestyle='--', color='gray', alpha=0.3)
    plt.axhline(-1, linestyle='--', color='gray', alpha=0.4)
    #plt.axhline(-1, linestyle='--', color='gray', alpha=0.4)
    plt.xlabel("Time (Unitless)")
    plt.ylabel("Amplitude")
    plt.title(f"Control Pulse Shape: I(t) and Q(t), Centre: {opt_center.item():.5f}, Width: {opt_width.item():.5f}")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    populations = np.abs(psi_t)**2
    plt.figure(figsize=(8, 4))
    plt.plot(times, populations[:, 0], label="|0⟩")
    plt.plot(times, populations[:, 1], label="|1⟩")
    plt.plot(times, populations[:, 2], label="|2⟩ (leakage)", linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("3-Level Qubit Evolution under Optimized Gaussian Pulse")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Optimisation OFF")
    psi_t = unitary_evolution_3level(H_func_3level, times, psi0,
                                     delta, alpha, omega_d, shape, center, width, I, Q)
    I_vals, Q_vals = normalized_drive_pulse(times, shape, center, width, I, Q)
    
    plt.figure(figsize=(8, 4))
    plt.plot(times, I_vals, label="I(t): Real Component (X)", color='blue')
    plt.plot(times, Q_vals, label="Q(t): Imaginary Component (Y)", color='orange')
    plt.axhline(1, linestyle='--', color='gray', alpha=0.4)
    plt.axhline(0, linestyle='--', color='gray', alpha=0.3)
    plt.axhline(-1, linestyle='--', color='gray', alpha=0.4)
    #plt.axhline(-1, linestyle='--', color='gray', alpha=0.4)
    plt.xlabel("Time (Unitless)")
    plt.ylabel("Amplitude")
    plt.title(f"Control Pulse Shape: I(t) and Q(t), Centre: {center:.5f}, Width: {width:.5f}")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    populations = np.abs(psi_t)**2
    plt.figure(figsize=(8, 4))
    plt.plot(times, populations[:, 0], label="|0⟩")
    plt.plot(times, populations[:, 1], label="|1⟩")
    plt.plot(times, populations[:, 2], label="|2⟩ (leakage)", linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("3-Level Qubit Evolution under Optimized Gaussian Pulse")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()