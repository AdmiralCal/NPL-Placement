# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 00:43:08 2025

@author: ctm1g20
"""

#TEST

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# -- Reconstruct minimal functions from previous code to simulate fidelity vs duration --

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

# Ideal target: X gate
U_target = X

# Drive pulse (Gaussian)
def normalized_drive_pulse(t, center, width, I, Q):
    envelope = np.exp(-((t - center) ** 2) / (2 * width ** 2))
    max_component = max(abs(I), abs(Q))
    I_norm = I / max_component
    Q_norm = Q / max_component
    I_t = envelope * I_norm
    Q_t = envelope * Q_norm
    return I_t + 1j * Q_t

# Control Hamiltonian
def H_func_2level(t, d_t, delta, omega_d):
    H_sys = delta * np.array([[4, 0], [0, 0]], dtype=complex)
    H_ctrl = (omega_d / 2) * (d_t.real * X + d_t.imag * Y)
    return H_sys + H_ctrl

# Simulate unitary evolution for two basis states
def unitary_evolution_gate(times, delta, omega_d, center, width, I, Q):
    dt = times[1] - times[0]
    U_sim = np.zeros((2, 2), dtype=complex)
    for col, psi0 in enumerate([np.array([[1], [0]], dtype=complex),
                                 np.array([[0], [1]], dtype=complex)]):
        psi = psi0.copy()
        for t in times[:-1]:
            d_t = normalized_drive_pulse(t, center, width, I, Q)
            H = H_func_2level(t, d_t, delta, omega_d)
            U = expm(-1j * H * dt)
            psi = U @ psi
        U_sim[:, col] = psi[:, 0]
    return U_sim

# Fidelity formula from Eq. 2
def gate_fidelity(U_sim, U_target):
    M = U_sim @ U_target.conj().T
    n = U_sim.shape[0]
    return np.real((np.trace(M @ M.conj().T) + abs(np.trace(M))**2) / (n * (n + 1)))

# Sweep gate durations
durations = np.linspace(0, 20, 100)  # Vary total gate time
fidelities = []

for t_total in durations:
    times = np.linspace(0, t_total, 500)
    center = 3
    width = 1
    I = 1.0
    Q = 0.0
    delta = 0.0
    omega_d = 1.0
    U_sim = unitary_evolution_gate(times, delta, omega_d, center, width, I, Q)
    F = gate_fidelity(U_sim, U_target)
    fidelities.append(F)

# Plot fidelity vs duration
plt.figure(figsize=(8, 4))
plt.plot(durations, fidelities)
plt.xlabel("Gate Duration")
plt.ylabel("Average Gate Fidelity")
plt.title("Fidelity vs Gate Duration for Gaussian X Gate")
plt.grid(True)
plt.tight_layout()
plt.show()
