# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:34:14 2025

@author: ctm1g20
"""

import numpy as np

# HADAMARD GATE
def hadamard():
    return (1/np.sqrt(2))*np.array([[1, 1],
                                    [1, -1]], dtype=complex)

# √Z† GATE
def sqrt_z_dag():
    return np.array([[1, 0],
                     [0, np.exp(-1j*np.pi/2)]], dtype=complex)

# TEMPORARY PROJECTIVE MEASUREMENT
def projective_meas(state):
    probs = np.abs(state)**2
    outcome = np.random.choice(len(probs), p=probs)
    return outcome


# STEP 1 OF SINGLE-QUBIT TOMOGRAPHY SEQUENCE (as seen on Wikipedia)

def single_qubit_tomography(psi_final, n_shots=1024):
    # Z GATE: MEASURE DIRECTLY
    # X GATE: APPLY H, THEN MEASURE
    # Y GATE: APPLY √Z† AND H, THEN MEASURE
    
    results = {'Z':[], 'X':[], 'Y':[]}
    H_gate = hadamard()
    SZD_gate = sqrt_z_dag()
    
    # Z BASIS MEASUREMENT
    for i in range(n_shots):
        state = psi_final[:2]
        outcome = projective_meas(state)
        results['Z'].append(outcome)
        
    #X BASIS MEASUREMENT
    for i in range(n_shots):
        state = psi_final[:2]
        state = H_gate @ state
        outcome = projective_meas(state)
        results['X'].append(outcome)
        
    # Y BASIS MEASUREMENT
    for i in range(n_shots):
        state = psi_final[:2]
        state = H_gate @ (SZD_gate @ state)
        outcome = projective_meas(state)
        results['Y'].append(outcome)
        
    return results

# STEP 2 & 3 OF SINGLE-QUBIT TOMOGRAPHY SEQUENCE

def bloch_vec_and_density_mat(tomography_results, n_shots):
    
    # PREAMBLE: PAULI MATRICES
    I2 = np.eye(2, dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # FIRST, COUNT NO. OF MEASUREMENTS IN EACH BASIS
    counts_z = np.bincount(tomography_results['Z'], minlength=2)
    counts_x = np.bincount(tomography_results['X'], minlength=2)
    counts_y = np.bincount(tomography_results['Y'], minlength=2)

    # GET EXPECTATION VALUES: (n_+ - n_-) / (n_+ + n_-)
    z_bar = (counts_z[0]-counts_z[1])/n_shots
    x_bar = (counts_x[0]-counts_x[1])/n_shots
    y_bar = (counts_y[0]-counts_y[1])/n_shots
    
    # CONSTRUCT BLOCH VECTOR AND CHECK IF IT IS NORMALISED
    r_vec = np.array([x_bar, y_bar, z_bar], dtype=float)
    r_norm = np.linalg.norm(r_vec)
    if r_norm > 1.0:
        r_vec = r_vec/r_norm 
        
    # CONSTRUCT DENSITY MATRIX: ρ_m = 1/2 (I + r⃗ ⋅ σ⃗)
    rho_m = (1/2)*(I2 + (r_vec[0] * sigma_x) + (r_vec[1] * sigma_y) + (r_vec[2] * sigma_z))
    
    return r_vec, rho_m
    
    































