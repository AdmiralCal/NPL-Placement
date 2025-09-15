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
# PARAMETERS AND INITIAL STATE
# -----------------------------
t_total = 10
n_steps = 1000
times = np.linspace(0, t_total, n_steps)
delta = -0.54412 # rad/ns 0
alpha = -1.95909  # rad/ns -3
omega_d = 1.28617  # rad/ns 1
shape = "gaussian"
width = 1
center = 3 * width
beta = 0
I = 1.0
Q = 0
optim = "Y"

psi0 = np.array([[1], [0], [0]], dtype=complex)
target = np.array([[0], [1], [0]], dtype=complex)

# -----------------------------
# UPDATED FIDELITY FUNCTION (Eq.2)
# -----------------------------
def gate_fidelity(width, beta, times):
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
    F_avg = (np.trace(M @ M.conj().T) + abs(np.trace(M))**2) / (n * (n + 1))
    return np.real(F_avg)

# -----------------------------
# OBJECTIVE FUNCTION WITH PENALTY
# -----------------------------
def objective2(params):
    width, beta = params
    center = 3 * width
    F_avg = gate_fidelity(width, beta, times)
    fidelity_loss = -F_avg
    return fidelity_loss

# -----------------------------
# EARLY STOPPING CALLBACK
# -----------------------------
best_fid = [0]
iter_count = [0]

def stop_if_high_enough(current_guess):
    iter_count[0] += 1
    width, beta = current_guess
    fid = gate_fidelity(width, beta, times)
    best_fid[0] = fid
    print(f"\n[Iteration {iter_count[0]}] Fidelity: {fid:.7f}")
    if fid >= 0.999995:
        raise StopIteration

# -----------------------------
# OPTIMIZATION CALL
# -----------------------------

initial_guess = [width, beta]  # beta initial guess = 0.0
bounds = [(0.1, t_total / 2), (-2.0, 2.0)]
if optim == "Y":
    print("Optimisation ON")
    print(f"INITIAL PULSE WIDTH:  {width}")
    print(f"INITIAL DRAG BETA:    {beta}")
    
    
    try:
        result = minimize(objective2, initial_guess, method='Nelder-Mead',
                          bounds=bounds, callback=stop_if_high_enough,
                          options={'maxiter': 200})
    except StopIteration:
        print("Early stopping: fidelity threshold reached.")

    opt_width, opt_beta = result.x
    opt_center = 3 * opt_width
    max_fidelity = best_fid[0]
    print("------------")
    print(f"Optimal center:    {opt_center:.5f}")
    print(f"Optimal width:     {opt_width:.5f}")
    print(f"Optimal beta:      {opt_beta:.5f}")
    print(f"Maximum fidelity:  {max_fidelity:.10f}")

    psi_t = unitary_evolution_3level(H_func_3level, times, psi0,
                                     delta, alpha, omega_d, shape, opt_center, opt_width, I, Q, opt_beta)
    I_vals, Q_vals = normalized_drive_pulse(times, shape, opt_center, opt_width, I, Q, opt_beta)

    plt.figure(figsize=(8, 4))
    plt.plot(times, I_vals, label="I(t): Real Component (X)", color='blue')
    plt.plot(times, Q_vals, label="Q(t): Imaginary Component (Y)", color='orange')
    plt.axhline(1, label="Optimisation ON", linestyle='--', color='gray', alpha=0.4)
    plt.axhline(0, linestyle='--', color='gray', alpha=0.3)
    plt.axhline(-1, linestyle='--', color='gray', alpha=0.4)
    plt.text(8, -0.95, 'Optimisation: ON', fontsize = 12)
    plt.xlabel("Time (Unitless)")
    plt.ylabel("Amplitude")
    plt.title(f"Control Pulse Shape: Centre:{opt_center:.5f}, Width: {opt_width:.5f}, $\\beta$: {opt_beta:.5f}")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    populations = np.abs(psi_t)**2
    plt.figure(figsize=(8, 4))
    plt.plot(times, populations[:, 0], label="|0⟩")
    plt.plot(times, populations[:, 1], label="|1⟩")
    plt.plot(times, populations[:, 2], label="|2⟩ (leakage)", linestyle='--')
    plt.xlabel("Time (Unitless)")
    plt.ylabel("Population")
    plt.title("3-Level Qubit Evolution under Optimized Gaussian + DRAG Pulse")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    durations = np.linspace(0, t_total, 100)  # Vary total gate time
    fidelities = []
    for t_total in durations:
        time = np.linspace(0, t_total, 500)
        F = gate_fidelity(opt_width, opt_beta, time)
        fidelities.append(F)
        
    # Plot fidelity vs duration
    plt.figure(figsize=(8, 4))
    plt.plot(durations, fidelities, label="Optimisation ON")
    plt.xlabel("Gate Duration, T (Unitless)")
    plt.ylabel("Average Gate Fidelity")
    plt.title(f"Fidelity vs Gate Duration - INPUT: Width: {width:.5f}, $\\beta$: {beta:.5f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

else:
    print("Optimisation OFF")
    psi_t = unitary_evolution_3level(H_func_3level, times, psi0,
                                     delta, alpha, omega_d, shape, center, width, I, Q, beta)
    I_vals, Q_vals = normalized_drive_pulse(times, shape, center, width, I, Q, beta)

    fid = gate_fidelity(width, beta, times)
    
    print("------------")
    print(f"Pulse Center:    {center:.5f}")
    print(f"Pulse Width:     {width:.5f}")
    print(f"DRAG Beta:      {beta:.5f}")
    print(f"Avg. Gate Fidelity idelity:  {fid:.10f}")    

    plt.figure(figsize=(8, 4))
    plt.plot(times, I_vals, label="I(t): Real Component (X)", color='blue')
    plt.plot(times, Q_vals, label="Q(t): Imaginary Component (Y)", color='orange')
    plt.axhline(1, label="Optimisation OFF", linestyle='--', color='gray', alpha=0.4)
    plt.axhline(0, linestyle='--', color='gray', alpha=0.3)
    plt.axhline(-1, linestyle='--', color='gray', alpha=0.4)
    plt.xlabel("Time (Unitless)")
    plt.ylabel("Amplitude")
    plt.title(f"Control Pulse Shape: Centre: {center:.5f}, Width: {width:.5f}, $\\beta$: {beta:.5f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    populations = np.abs(psi_t)**2
    plt.figure(figsize=(8, 4))
    plt.plot(times, populations[:, 0], label="|0⟩")
    plt.plot(times, populations[:, 1], label="|1⟩")
    plt.plot(times, populations[:, 2], label="|2⟩ (leakage)", linestyle='--')
    plt.xlabel("Time (Unitless)")
    plt.ylabel("Population")
    plt.title("3-Level Qubit Evolution under Unoptimized Gaussian Pulse + DRAG Pulse")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    durations = np.linspace(0, t_total, 100)  # Vary total gate time
    fidelities = []
    for t_total in durations:
        time = np.linspace(0, t_total, 500)
        F = gate_fidelity(width, beta, time)
        fidelities.append(F)
        
    # Plot fidelity vs duration
    plt.figure(figsize=(8, 4))
    plt.plot(durations, fidelities, label="Optimisation OFF")
    plt.xlabel("Gate Duration, T (Unitless)")
    plt.ylabel("Average Gate Fidelity")
    plt.title(f"Fidelity vs Gate Duration - INPUT: Width: {width:.5f}, $\\beta$: {beta:.5f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
