# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 13:31:55 2025

@author: ctm1g20
"""

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


def grape_step(drift, controls, waveform, psi_init, psi_targ, dt):
    """
    Perform one GRAPE optimization step for piecewise-constant pulse.

    Parameters
    ----------
    drift : ndarray
        Drift Hamiltonian (3x3 for transmon).
    controls : list of ndarray
        List of control Hamiltonians [H_I, H_Q].
    waveform : ndarray
        Shape (n_ctrls, n_steps). Control amplitudes for each time step.
        waveform[0,:] = I(t), waveform[1,:] = Q(t).
    psi_init : ndarray
        Initial statevector (3,).
    psi_targ : ndarray
        Target statevector (3,).
    dt : float
        Time step duration (ns).

    Returns
    -------
    fidelity : float
        Fidelity between evolved final state and target state.
    grad : ndarray
        Gradient of fidelity wrt waveform amplitudes (same shape as waveform).
    """
    n_ctrls, n_steps = waveform.shape

    # Forward trajectories
    fwd = np.zeros((len(psi_init), n_steps+1), dtype=complex)
    fwd[:,0] = psi_init.flatten()

    # Backward trajectories
    bwd = np.zeros((len(psi_targ), n_steps+1), dtype=complex)
    bwd[:,0] = psi_targ.flatten()

    # Hamiltonians at each step
    H_fwd = []
    H_bwd = []

    # Forward propagation
    for n in range(n_steps):
        Hn = drift.copy()
        for k in range(n_ctrls):
            Hn += waveform[k,n] * controls[k]
        H_fwd.append(Hn)
        U = expm(-1j * Hn * dt)
        fwd[:,n+1] = U @ fwd[:,n]

    # Backward propagation (time reversed)
    for n in range(n_steps):
        Hn = drift.copy()
        for k in range(n_ctrls):
            Hn += waveform[k, n_steps-1-n] * controls[k]
        H_bwd.append(Hn)
        U = expm(+1j * Hn * dt)
        bwd[:,n+1] = U @ bwd[:,n]

    # Fidelity
    fidelity = np.real(np.vdot(psi_targ.flatten(), fwd[:,-1]))

    # Gradient calculation
    grad = np.zeros_like(waveform, dtype=float)
    bwd = np.fliplr(bwd)  # align with forward time

    for n in range(n_steps):
        for k in range(n_ctrls):
            # Auxiliary block matrix
            Hn = H_fwd[n]
            zero_drift = np.zeros_like(Hn)
            # AUXMAT METHOD FOR EXACT GRADIENT DERIVATIVE - see Ilya's Spin book for more info
            aux_matrix = np.block([[ Hn,            controls[k] ],
                                   [ zero_drift,   Hn          ]])
            aux_vec = np.concatenate([np.zeros_like(fwd[:,n]), fwd[:,n]])
            aux_vec = expm(1j * aux_matrix * dt) @ aux_vec

            grad[k,n] = np.real(np.vdot(bwd[:,n+1], aux_vec[:len(fwd[:,n])]))

    return fidelity, grad

def grape_objective(waveform_flat, drift, controls, psi_init, psi_targ, dt, n_steps):
    """
    Objective + gradient function for SciPy minimize.
    waveform_flat: flattened vector of shape (n_controls * n_steps,)
    Returns:
      f: scalar (negative fidelity)
      g: flattened gradient vector
    """
    # Reshape waveform to (n_controls, n_steps)
    waveform = waveform_flat.reshape(len(controls), n_steps)

    # Compute fidelity and gradient from GRAPE
    fidelity, grad = grape_step(drift, controls, waveform, psi_init, psi_targ, dt)

    # SciPy minimizes, so we take -fidelity
    f = -fidelity
    g = grad.flatten()   # flatten gradient to match waveform_flat
    return f, g

def run_grape_optimization(drift, controls, psi_init, psi_targ, dt, n_steps, n_controls, max_iter=200):
    """
    Run GRAPE optimization using L-BFGS-B.
    """
    # Initial waveform guess (e.g., zeros or small random values)
    #x0 = np.random.uniform(-0.1, 0.1, size=(n_controls, n_steps)).flatten()
    x0 = np.random.uniform(-0.01, 0.01, size=(n_controls, n_steps)).flatten()

    # Define wrapper so SciPy sees (f, g)
    def objective_with_grad(x_flat):
        return grape_objective(x_flat, drift, controls, psi_init, psi_targ, dt, n_steps)

    # Run L-BFGS-B
    result = minimize(
        objective_with_grad,
        x0,
        method="L-BFGS-B",
        jac=True,                   # use analytic gradient
        options={"maxiter": max_iter, "disp": True, "ftol": 1e-10, "gtol": 1e-8, "maxls": 50}
    )

    # Reshape optimized waveform
    opt_waveform = result.x.reshape(n_controls, n_steps)
    return opt_waveform, -result.fun, result

# -----------------------------
# DRIFT + CONTROL HAMILTONIANS FOR GRAPE
# -----------------------------
t_total = 10
delta = -0.54412 # rad/ns 0 or -0.54412
alpha = -1.95093  # rad/ns -3 or -1.95093
omega_d = 1.28617  # rad/ns 1 or 1.28617
b = np.array([[0, 1, 0],
              [0, 0, np.sqrt(2)],
              [0, 0, 0]], dtype=complex)
b_dag = b.conj().T
n_op = b_dag @ b

# Drift term (anharmonicity + detuning)
H_drift = delta * n_op + (alpha/2) * (b_dag @ b_dag @ b @ b)

# Control Hamiltonians for piecewise amplitudes
H_I = (omega_d/2) * (b_dag + b)               # "in-phase" quadrature
H_Q = (omega_d/2) * (1j*b_dag - 1j*b)        # "quadrature"
controls = [H_I, H_Q]
n_controls = 2
# -----------------------------
# GRAPE OPTIMIZATION LOOP
# -----------------------------

n_steps = 200              # number of piecewise segments
dt = t_total / n_steps    # duration of each segment
#waveform = np.random.uniform(-0.1, 0.1, (2, n_steps))  # initial I/Q pulse

psi_init = np.array([1,0,0], dtype=complex)   # start in |0>
psi_targ = np.array([0,1,0], dtype=complex)   # target is |1>



opt_waveform, fidelity, result = run_grape_optimization(
    drift=H_drift,
    controls=[H_I, H_Q],
    psi_init=psi_init,
    psi_targ=psi_targ,
    dt=dt,
    n_steps=n_steps,
    n_controls=n_controls,
    max_iter=400
)

plt.figure(figsize=(8,4))
plt.step(range(n_steps), opt_waveform[0], where="mid", label="I control")
plt.step(range(n_steps), opt_waveform[1], where="mid", label="Q control")
plt.xlabel("Time step")
plt.ylabel("Amplitude")
plt.title("Optimized Piecewise-Constant Control Pulse (GRAPE)")
plt.legend()
plt.grid(True)
plt.show()

"""
fidelity_history = []
lr = 0.05  # learning rate
n_iters = 1000
for it in range(n_iters):
    fid, grad = grape_step(H_drift, controls, waveform, psi_init, psi_targ, dt)
    waveform += lr * grad   # gradient ascent
    fidelity_history.append(fid)
    print(f"[Iter {it}] Fidelity = {fid:.6f}")

# Plot fidelity convergence
plt.figure(figsize=(6,4))
plt.plot(fidelity_history, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Fidelity")
plt.title("GRAPE Optimization Convergence")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.step(range(n_steps), waveform[0,:], where='mid', label="I(t)", color="blue")
plt.step(range(n_steps), waveform[1,:], where='mid', label="Q(t)", color="orange")
plt.xlabel("Time step")
plt.ylabel("Amplitude")
plt.title("Optimized Piecewise-Constant Pulse (GRAPE)")
plt.legend()
plt.grid(True)
plt.show()
"""