# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 17:23:53 2025

@author: ctm1g20

BASIC TEST OF A THREE LEVEL TRANSMON SYSTEM, TO BE RESTRICTED TO TWO LEVELS
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Constructs Single-qubit rotating-frame Hamiltonian for 3-level transmon - ASK IVAN/YANNICK ABOUT THIS
def single_qubit_hamiltonian(t, d_t, delta, alpha, omega_d):
    b = np.array([[0, 1, 0],               # Lowering Operator
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    b_dag = b.conj().T                     # Raising Operator
    n = b_dag @ b                          # Number Operator                             
    H_sys = delta * n + (alpha / 2) * (b_dag @ b_dag @ b @ b)    # System (Drift) Hamiltonian
    H_ctrl = (omega_d / 2) * (d_t * b + np.conj(d_t) * b_dag)    # Control Hamiltonian
    return H_sys + H_ctrl

# Generates Normalized complex drive envelope d(t)
def normalized_drive_pulse(t, shape="gaussian", center=25e-9, width=5e-9, I=1.0, Q=0.5):
    if shape == "gaussian":
        envelope = np.exp(-((t - center) ** 2) / (2 * width ** 2))
#    elif shape == "square":
#        envelope = np.where(np.abs(t - center) < width, 1.0, 0.0)
#    elif shape == "flat-top":
#        ramp = np.exp(-((t - center) ** 2) / (2 * (0.25 * width) ** 2))
#        envelope = np.maximum(ramp, 0.8)
    else:
        raise ValueError(f"Unknown shape '{shape}'")

    max_component = max(abs(I), abs(Q))   # Normalises Control Signals to [-1, 1]
    I_norm = I / max_component
    Q_norm = Q / max_component
    return envelope * (I_norm + 1j * Q_norm)

# Constructs Time-dependent Hamiltonian function using d(t)
def H_func_normalized(t, delta, alpha, omega_d, **kwargs):
    d_t = normalized_drive_pulse(t, **kwargs)
    return single_qubit_hamiltonian(t, d_t, delta, alpha, omega_d)

# Unitary evolution via time-ordered exponentials
def unitary_evolution(H_func, times, psi0, **kwargs):
    dt = times[1] - times[0]
    psi_t = [psi0]
    psi = psi0.copy()
    for t in times[:-1]:
        H = H_func(t, **kwargs)
        U = expm(-1j * H * dt)
        psi = U @ psi
        psi_t.append(psi)
    return np.array(psi_t)

# Simulation setup - THESE ARE BOGUS VALUES JUST MEANT TO TEST THAT I DONT GET ERRORS
t_total = 50e-9      # total time: 50 ns
n_steps = 500
times = np.linspace(0, t_total, n_steps)
delta = 0.0          # on-resonance (to be replaced with transition freq. and driving freq. for specific adjustments)
alpha = -200e6       # anharmonicity in Hz
omega_d = 1e6        # drive strength in Hz 
psi0 = np.array([[1], [0], [0]], dtype=complex)  # initial |0⟩ state

# Simulate evolution
psi_t = unitary_evolution(H_func_normalized, times, psi0,
                          delta=delta, alpha=alpha, omega_d=omega_d,
                          shape="gaussian", center=25e-9, width=5e-9,
                          I=1.0, Q=0.5)

# Population Plots
populations = np.abs(psi_t)**2
plt.plot(times * 1e9, populations[:, 0], label="|0⟩")
plt.plot(times * 1e9, populations[:, 1], label="|1⟩")
plt.plot(times * 1e9, populations[:, 2], label="|2⟩")
plt.xlabel("Time (ns)")
plt.ylabel("Population")
plt.title("Transmon Qubit Evolution with Normalized Complex Pulse")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
