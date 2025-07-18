# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 08:44:23 2025

@author: ctm1g20
"""

#UPDATED 2-LEVEL SYSTEM

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize
import math



# Rotating-frame Hamiltonian for a 2-level system
def single_qubit_hamiltonian_2level(t, d_t, delta, omega_d):
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    #Z = np.array([[1, 0], [0, -1]], dtype=complex)  # NOT NEEDED, HERE FOR COMPLETENESS SAKE

    # Control Hamiltonian: drive = (Re[d(t)] X + Im[d(t)] Y)
    H_sys = delta * np.array([[4, 0], [0, 0]])  # System (Drift) Hamiltonian - CONFUSED ABOUT THIS, ASK YANNICK
    H_ctrl = (omega_d / 2) * (d_t.real * X + d_t.imag * Y) # Control Hamiltonian 
    
    return H_sys + H_ctrl

# Normalized drive envelope d(t)
def normalized_drive_pulse(t, shape, center, width, I, Q):
    if shape == "gaussian":
        envelope = np.exp(-((t - center) ** 2) / (2 * width ** 2))
#    elif shape == "square":
#        envelope = np.where(np.abs(t - center) < width, 1.0, 0.0)
#    elif shape == "flat-top":
#        ramp = np.exp(-((t - center) ** 2) / (2 * (0.25 * width) ** 2))
#        envelope = np.maximum(ramp, 0.8)
    else:
        raise ValueError(f"Unknown shape '{shape}'")

    max_component = max(abs(I), abs(Q))
    I_norm = I / max_component
    Q_norm = Q / max_component
    I_t = envelope * I_norm
    Q_t = envelope * Q_norm

    return I_t, Q_t  # Return real and imaginary parts of the pulse separately



# Time-dependent Hamiltonian wrapper (2-level)
def H_func_2level(t, delta, omega_d, shape, center, width, I, Q):
    I_t, Q_t = normalized_drive_pulse(t, shape, center, width, I, Q)
    d_t = I_t + 1j*Q_t  
    return single_qubit_hamiltonian_2level(t, d_t, delta, omega_d)

# Unitary evolution for 2-level system
def unitary_evolution_2level(H_func, times, psi0,
                             delta, omega_d, shape, center, width, I, Q):
    dt = times[1] - times[0]
    psi_t = [psi0]
    psi = psi0.copy()
    for t in times[:-1]:
        H = H_func(t, delta, omega_d, shape, center, width, I, Q)
        U = expm(-1j * H * dt)
        psi = U @ psi
        psi_t.append(psi)
    return np.array(psi_t)

# Simulation parameters - THESE ARE BOGUS VALUES JUST MEANT TO TEST THAT I DONT GET ERRORS
t_total = 15     # 50 ns total
n_steps = 1000
times = np.linspace(0, t_total, n_steps)
psi0 = np.array([[1], [0]], dtype=complex)  # start in |0⟩
target = np.array([[0], [1]], dtype=complex)
# Drive and detuning
delta = 0.0            # on-resonance
omega_d = 1        # drive strength (Hz)

# Pulse shape parameters
shape = "gaussian"

width = 1    # width of pulse
center = 3*width        # center of pulse
I = 1                # real (X) component
Q = 0              # imaginary (Y) component

# -----------------------
# OPTIMIZATION BLOCK START
# -----------------------

def gate_fidelity(center, width, times):

    # Simulate evolution from |0⟩ and |1⟩ to build simulated unitary U_sim
    psi0 = np.array([[1], [0]], dtype=complex)  # |0⟩
    psi1 = np.array([[0], [1]], dtype=complex)  # |1⟩

    # Evolve both basis states under the control pulse
    psi0_t = unitary_evolution_2level(H_func_2level, times, psi0,
                                      delta, omega_d, shape, center, width, I, Q)
    psi1_t = unitary_evolution_2level(H_func_2level, times, psi1,
                                      delta, omega_d, shape, center, width, I, Q)

    # Construct simulated 2x2 unitary from evolved states
    # Each column is the final state from evolving one of the basis states
    U_qubit = np.hstack([psi0_t[-1], psi1_t[-1]])  # Simulated final unitary

    # Define ideal target gate: X gate
    U_target = np.array([[0, 1], [1, 0]], dtype=complex)

    M = U_qubit @ U_target.conj().T
    n = 2  # Dimension of qubit subspace

    # EQUATION 2
    F_avg = (np.trace(M @ M.conj().T) + abs(np.trace(M))**2) / (n * (n + 1))
    F_avg = np.real(F_avg)
    return F_avg

# Objective function to minimize (negative fidelity with target state |1⟩)
def objective1(params):
    center, width = params
    psi_t = unitary_evolution_2level(H_func_2level, times, psi0,
                                     delta, omega_d, shape, center, width,
                                     I, Q)  # FIXED I, Q
    final_state = psi_t[-1]
    fidelity = np.abs(np.vdot(target, final_state))**2
    
    # PENALTY FOCUSED OPTIMISATION
    fidelity_loss = -fidelity  # we minimize negative fidelity

    # --- Penalty: encourage I(t=0) to be ~0 ---
    I_t0, _ = normalized_drive_pulse(times[0], shape, center, width, I, Q)
    start_penalty = abs(I_t0)**2

    #PENALTY WEIGHT!
    # 0.1-1 = Fidelity dominates (fast, easy)
    # 1-5 = Balanced between fidelity and pulse
    # 5-50 = Pulse dominates (Fidelity still good somehow)
    lambda_penalty = 50.0

    # Total loss = fidelity loss + penalty
    return fidelity_loss + lambda_penalty * start_penalty
    #return fidelity_loss  # minimize negative fidelity

# Objective function using averaged gate fidelity (Eq. 2 of the paper) for a 2-level system
def objective2(params):
    center, width = params
    F_avg = gate_fidelity(center, width, times)
    fidelity_loss = -np.real(F_avg)  # We minimize negative fidelity

    # --- Penalty: encourage I(t=0) to be ~0 ---
    I_t0, _ = normalized_drive_pulse(times[0], shape, center, width, I, Q)
    start_penalty = abs(I_t0)**2

    #PENALTY WEIGHT!
    # 0.1-1 = Fidelity dominates (fast, easy)
    # 1-5 = Balanced between fidelity and pulse
    # 5-50 = Pulse dominates (Fidelity still good somehow)
    lambda_penalty = 50.0

    # Total loss = fidelity loss + penalty
    return fidelity_loss + lambda_penalty * start_penalty


# Early stopping: raise StopIteration if fidelity ≥ threshold
best_fid = [0]
iter_count = [0]

def stop_if_high_enough(current_guess):
    iter_count[0] += 1
    center, width = current_guess
    fid = gate_fidelity(center, width, times)
    best_fid[0] = fid
    I_t0, _ = normalized_drive_pulse(times[0], shape, center, width, I, Q)
    I_tfinal, _ = normalized_drive_pulse(times[-1], shape, center, width, I, Q)
    print()
    print(f"[Iteration {iter_count[0]}]")
    print(f"Current fidelity: {fid:.7f}")
    print(f"I(t=0):           {I_t0:.6f}")
    print(f"I(t={times[-1]}):        {I_tfinal:.6f}")
    if fid >= 0.999995:
        raise StopIteration

# Initial guess for pulse center, width
initial_guess = [center, width]
bounds = [(0.0, t_total), (0.1, t_total / 2)] 
# Sets bounds for optimised pulse center and width
# Pulse center must be within the time window
# Pulse has reasonable width, AT LEAST 0.1 to avoid infinitesimally short pulses, AT MOST TOTAL TIME/2 to avoid flat pulses
solving_method = 'Nelder-Mead' # Other solving methods can also be used, but N-M is very simple.

# Run optimization with early stopping
try:
    print("Chosen Solving Method: " + solving_method)
    result = minimize(objective2, initial_guess, method=solving_method,
                      bounds=bounds, callback=stop_if_high_enough,
                      options={'maxiter': 200})
except StopIteration:
    print("Early stopping: fidelity threshold reached.")

# Extract best parameters
opt_center, opt_width = result.x
max_fidelity = best_fid[0]
print("------------")
print(f"Optimal center:    {opt_center:.5f}")
print(f"Optimal width:     {opt_width:.5f}")
print(f"Maximum fidelity:  {max_fidelity:.10f}")

# -------------------------
# OPTIMISATION BLOCK END
# -------------------------


# Run simulation
psi_t = unitary_evolution_2level(H_func_2level, times, psi0,
                                 delta, omega_d, shape, opt_center, opt_width, I, Q)

I_vals, Q_vals = normalized_drive_pulse(times, shape, opt_center, opt_width, I, Q)
I_array = np.array(I_vals)
area = (np.trapz(I_array, times))/np.pi
expected_area = (np.pi/omega_d)/np.pi
error = (np.abs(area - expected_area))
print("------------")
print(f"Gaussian Pulse Area:    {area:.6f} Pi")
print(f"Expected Pulse Area:    {expected_area:.6f} Pi")
print(f"Absolute error:         {error:.6e} Pi")

# CHECKS PHASE ANGLE OF FINAL STATE COMPARED TO TARGET STATE
actual = psi_t[-1]
print("------------")
print("Final State:")
print(actual)
overlap = np.vdot(target, actual)
phase_angle = np.angle(overlap)
phase_angle_pi = phase_angle/math.pi
print(f"Relative Phase Angle (radians):  {phase_angle_pi:.4f} Pi")

# EXPERIMENTAL PHASE CORRECTION STUFF - NOT IMPORTANT

final_corrected = actual * np.exp(-1j * phase_angle)
print("------------")
print("Corrected Final State:")
print(abs(final_corrected))
cor_overlap = np.vdot(target, final_corrected)
cor_phase_angle = np.angle(cor_overlap)
cor_phase_angle_pi = cor_phase_angle/math.pi
print(f"Corrected Relative Phase Angle (radians): {cor_phase_angle_pi:.4f} Pi")



plt.figure(figsize=(8, 4))
plt.plot(times, I_vals, label="I(t): Real Component (X)", color='blue')
plt.plot(times, Q_vals, label="Q(t): Imaginary Component (Y)", color='orange')
plt.axhline(1, linestyle='--', color='gray', alpha=0.4)
plt.axhline(0, linestyle='--', color='gray', alpha=0.3)
#plt.axhline(-1, linestyle='--', color='gray', alpha=0.4)
plt.xlabel("Time (Unitless)")
plt.ylabel("Amplitude")
plt.title(f"Control Pulse Shape: I(t) and Q(t), Centre: {opt_center:.5f}, Width: {opt_width:.5f}")
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# Population Plots
populations = np.abs(psi_t)**2 # Gives probabilities of finding the system in states |0> and |1>
plt.figure(figsize=(8, 4))
plt.plot(times, populations[:, 0], label="|0⟩")
plt.plot(times, populations[:, 1], label="|1⟩")
plt.xlabel("Time (Unitless)")
plt.ylabel("Population (Probability)")
plt.title("2-Level Qubit Evolution under Normalized Drive")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

durations = np.linspace(0, 15, 100)  # Vary total gate time
fidelities = []

for t_total in durations:
    time = np.linspace(0, t_total, 500)
    F = gate_fidelity(opt_center, opt_width, time)
    fidelities.append(F)



# Plot fidelity vs duration
plt.figure(figsize=(8, 4))
plt.plot(durations, fidelities)
plt.xlabel("Gate Duration (Unitless)")
plt.ylabel("Average Gate Fidelity")
plt.title("Fidelity vs Gate Duration for Gaussian X Gate")
plt.grid(True)
plt.tight_layout()
plt.show()
