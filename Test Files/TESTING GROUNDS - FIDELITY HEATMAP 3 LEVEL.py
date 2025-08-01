# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 15:00:22 2025

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
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
t_total = 20
n_steps = 1000
times = np.linspace(0, t_total, n_steps)
delta = 0.0
alpha = 0 #-0.3
shape = "gaussian"
width = 1.25625#1.30354
center = 3 * width
I = 1.0
Q = 0
optim = "Y"

psi0 = np.array([[1], [0], [0]], dtype=complex)
target = np.array([[0], [1], [0]], dtype=complex)

# -----------------------------
# UPDATED FIDELITY FUNCTION (Eq.2)
# -----------------------------
def gate_fidelity(width, beta, times, omega=1.0):
    center = 3 * width
    psi0 = np.array([[1], [0], [0]], dtype=complex)
    psi1 = np.array([[0], [1], [0]], dtype=complex)

    psi0_t = unitary_evolution_3level(H_func_3level, times, psi0,
                                      delta, alpha, omega, shape, center, width, I, Q, beta)
    psi1_t = unitary_evolution_3level(H_func_3level, times, psi1,
                                      delta, alpha, omega, shape, center, width, I, Q, beta)

    U_sim = np.hstack([psi0_t[-1][:2], psi1_t[-1][:2]])
    U_target = np.array([[0, 1], [1, 0]], dtype=complex)

    M = U_sim @ U_target.conj().T
    n = 2
    F_avg = (np.trace(M @ M.conj().T) + abs(np.trace(M))**2) / (n * (n + 1))
    return np.real(F_avg)   

values = 61
beta_values = np.linspace(3.0, -3.0, values)  
omega_vals = np.linspace(0.0, 10.0, values)
fidelities = []
combined_f = np.zeros((values,values))

while alpha >= -1.0 :
        
    for i, beta in enumerate(beta_values):
        F = gate_fidelity(width, beta, times)
        fidelities.append(F)
        
        for j, omega in enumerate(omega_vals):
            f = gate_fidelity(width, beta, times, omega)
            combined_f[i][j] = f
    
    
    plt.figure(figsize=(8, 4))
    plt.plot(beta_values, fidelities, label="Avg. Gate Fidelity")
    plt.xlabel("DRAG Coefficient β")
    plt.ylabel("Fidelity")
    plt.title(f"Fidelity vs DRAG Beta for 3-Level Transmon w/ $\\Omega$ = 1.0, $\\alpha$ = {alpha:.1f}")
    plt.grid(True)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=400)
    im = ax.imshow(combined_f, cmap='Reds')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(im, cax=cax)
    # Show all ticks and label them with the respective list entries
    
    
    stepx = 3
    stepy = 3
    ax.set_xticks(range(0, len(omega_vals), stepx))
    ax.set_xticklabels([f"{o:.2f}" for o in omega_vals[::stepx]], rotation=45)
    ax.set_yticks(range(0, len(beta_values), stepy))
    ax.set_yticklabels([f"{b:.2f}" for b in beta_values[::stepy]])
    #ax.set_xticks(range(len(beta_values)), labels=[f"{b:.2f}" for b in beta_values])
    #ax.set_yticks(range(len(omega_vals)), labels=[f"{o:.2f}" for o in omega_vals])
    
    ax.set_xlabel('Omega, $\\Omega$', fontweight ='bold')
    ax.set_ylabel('Beta, $\\beta$', fontweight ='bold')
    # Loop over data dimensions and create text annotations.
    #for i in range(len(omega_vals)):
    #    for j in range(len(beta_values)):
    #        text = ax.text(j, i, "%.2f" % combined_f[i, j],
    #                       ha="center", va="center", fontsize=6.5)
    
    ax.set_title(f"Fidelity Heatmap, Width: {width:.5f}, $\\alpha$ = {alpha:.1f}")
    fig.tight_layout()
    plt.show()
    
    fidelities = []
    combined_f = np.zeros((values,values))
    alpha -= 0.05