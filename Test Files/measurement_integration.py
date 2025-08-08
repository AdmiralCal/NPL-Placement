# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 11:57:46 2025

@author: ctm1g20
"""

"""
measurement_integration.py

Utilities to integrate projective measurements for a 3-level transmon
into an existing simulation / optimization routine.

Assumptions:
- Your simulator returns psi_t: a 1D or 2D numpy array containing the
  time evolution of the state vectors. The final state is psi_t[-1].
  Each statevector is shape (3,) or (3,1) complex numpy array.
- For optimization, you pass candidate pulses to a function
  `simulate_pulse_and_return_psi_t(pulse, sim_params)` that returns psi_t.

What to import into your main code:
from measurement_integration import (
    measure_probs_from_statevec,
    simulate_shots,
    empirical_probs_and_ci,
    measurement_reward,
    optimize_with_measurements
)
"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Basic measurement utilities
# ---------------------------------------------------------------------------

def measure_probs_from_statevec(psi_final):
    """
    Compute exact probabilities p0,p1,p2 from final statevector psi_final.
    - psi_final: complex array shape (3,) or (3,1)
    Returns:
    - probs: np.array([p0,p1,p2]) that sums to 1 (normalized numerically).
    """
    psi = np.asarray(psi_final).ravel()
    if psi.shape[0] != 3:
        raise ValueError("measure_probs_from_statevec expects a length-3 statevector.")
    probs = np.abs(psi)**2
    total = probs.sum()
    if total <= 0:
        # numerical safety: return uniform if zero norm (shouldn't happen)
        return np.array([1/3, 1/3, 1/3])
    return probs / total


def simulate_shots(probs, n_shots=1024, rng=None):
    """
    Simulate projective measurement 'shots' from three-level probabilities.
    - probs: array-like of length 3 summing to ~1
    - n_shots: number of repeated measurements to simulate
    - rng: optional np.random.Generator for reproducibility
    Returns:
    - counts: np.array([n0, n1, n2]) integer counts
    - outcomes: if you need per-shot outcomes, returns the array of outcomes (optional)
    Notes:
    - Uses np.random.multinomial for efficient sampling of the multinomial distribution.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.shape[0] != 3:
        raise ValueError("probs must be length 3")
    # enforce numerical normalization
    probs = probs / probs.sum()
    if rng is None:
        rng = np.random.default_rng()
    counts = rng.multinomial(n_shots, probs)
    return counts


def empirical_probs_and_ci(counts, alpha=0.05):
    """
    Compute empirical probabilities and approximate 1-alpha (default 95%) CI
    using normal approximation for multinomial entries (works well when n_shots large).
    Returns dict with keys:
      - p_emp: empirical probs
      - stderr: standard errors (sqrt(p(1-p)/n))
      - ci_low, ci_high arrays for each state
    For small n_shots or extreme p, use Wilson score / exact binomial to be safer.
    """
    counts = np.asarray(counts, dtype=int)
    n = counts.sum()
    if n == 0:
        raise ValueError("counts sum to zero")
    p_emp = counts / n
    z = 1.96  # ~97.5 percentile for two-sided 95% CI; for alpha generic, use scipy.stats.norm.ppf
    stderr = np.sqrt(p_emp * (1 - p_emp) / n)
    ci_low = p_emp - z * stderr
    ci_high = p_emp + z * stderr
    # clamp to [0,1]
    ci_low = np.clip(ci_low, 0.0, 1.0)
    ci_high = np.clip(ci_high, 0.0, 1.0)
    return {
        "p_emp": p_emp,
        "stderr": stderr,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_shots": n
    }

# ---------------------------------------------------------------------------
# Helpers to use measurement outcomes in optimization / rewards
# ---------------------------------------------------------------------------

def measurement_reward(probs, target_state=1, metric="negative_log_prob", counts=None, n_shots=1024):
    """
    Example: turn measurement result into a scalar reward for optimization.
    Arguments:
      - probs: exact probabilities (3-vector) from final statevector (theoretical).
      - target_state: integer 0/1/2 that we want to maximize (e.g., |1> population)
      - metric: choose how to compute reward:
          * "negative_log_prob": -log10(1 - p_target)  (similar to -log10(infidelity) style)
          * "prob": p_target (direct probability)
          * "shot_based_prob": simulate shots and use empirical p_target
      - counts: optional precomputed shot counts (if you already simulated them)
      - n_shots: used if we need to simulate shots here
    Returns:
      - reward: float (higher = better)
      - extra: dict with diagnostic values (p_target, counts, empirical_p)
    Notes:
      - Negative log style is useful because it stretches out near-unity probabilities.
      - If you use this reward in RL, keep it bounded/stable (clamp extreme values).
    """
    probs = np.asarray(probs, dtype=float)
    p_target = probs[target_state]
    extra = {"p_target_exact": float(p_target)}
    if metric == "prob":
        reward = float(p_target)
        return reward, extra

    if metric == "negative_log_prob":
        # small epsilon to avoid -inf when p_target=1
        eps = 1e-12
        reward = -np.log10(max(eps, 1.0 - p_target))
        extra["reward_scale"] = "log"
        return float(reward), extra

    if metric == "shot_based_prob":
        if counts is None:
            counts = simulate_shots(probs, n_shots=n_shots)
        emp = counts / counts.sum()
        p_emp_target = emp[target_state]
        extra["counts"] = counts
        extra["p_emp_target"] = float(p_emp_target)
        # reward can be empirical probability, or -log10(1-p_emp)
        reward = float(p_emp_target)
        return reward, extra

    raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Integration wrapper: place into your optimization/agent loop
# ---------------------------------------------------------------------------

def optimize_with_measurements(
    simulate_fn,
    pulse_generator,
    n_iterations=100,
    shots_per_eval=500,
    use_shot_noise=False,
    target_state=1,
    metric="negative_log_prob",
    rng=None,
    verbose=True
):
    """
    Example (black-box) optimization wrapper that demonstrates how to
    integrate projective measurement into evaluation.

    Arguments:
      - simulate_fn(pulse) -> psi_t
         a callable that accepts a candidate pulse and returns psi_t (time-ordered states)
      - pulse_generator(i) -> pulse
         returns a candidate pulse for evaluation (dummy example: random pulses).
         In your code, replace with RL agent action or optimizer-driven candidate.
      - n_iterations: how many candidates to evaluate (episodes)
      - shots_per_eval: if shot-based evaluation enabled, number of shots to simulate
      - use_shot_noise: if True, uses shot-based empirical reward; else uses exact probs
      - target_state: which level to target (0,1,2)
      - metric: measurement_reward metric
      - rng: np.random.Generator optionally for reproducibility
    Returns:
      - results: list of dicts with diagnostics per iteration
    Notes:
      - This is intentionally generic: your optimizer will instead generate pulses.
      - You can adapt the returned 'results' to store in replay buffer, log files, etc.
    """
    if rng is None:
        rng = np.random.default_rng()

    results = []
    for i in range(n_iterations):
        # --- 1) get a candidate pulse from your generator / agent / optimizer
        pulse = pulse_generator(i)

        # --- 2) simulate the pulse (your existing routine) -> psi_t
        psi_t = simulate_fn(pulse)   # must return array-like with psi_t[-1] final state

        # --- 3) get exact probs from final wavefunction
        final_state = np.asarray(psi_t[-1]).ravel()
        probs = measure_probs_from_statevec(final_state)

        # --- 4) possibly simulate finite-shot measurement outcomes
        counts = None
        if use_shot_noise:
            counts = simulate_shots(probs, n_shots=shots_per_eval, rng=rng)
            meas_reward, extra = measurement_reward(probs, target_state=target_state,
                                                    metric="shot_based_prob", counts=counts)
            # compute empirical CI
            ci = empirical_probs_and_ci(counts)
            extra["empirical_ci"] = (ci["ci_low"], ci["ci_high"])
        else:
            meas_reward, extra = measurement_reward(probs, target_state=target_state,
                                                    metric=metric)

        # --- 5) create an overall diagnostics dict (you can feed 'meas_reward' into optimizer)
        diagnostics = {
            "iteration": i,
            "pulse": pulse,
            "probs_exact": probs,
            "reward": meas_reward,
            "counts": counts,
            "extra": extra
        }
        results.append(diagnostics)

        if verbose:
            if counts is None:
                print(f"[{i:03d}] exact p(|{target_state}>) = {probs[target_state]:.6f} reward={meas_reward:.4f}")
            else:
                print(f"[{i:03d}] exact p(|{target_state}>) = {probs[target_state]:.6f} empirical = {counts/ counts.sum()} reward={meas_reward:.4f}")

        # --- 6) (Optionally) pass 'meas_reward' back to your optimizer/agent as the objective
        # In RL: you would convert reward to the agent's reward signal and store transition.
        # In gradient-free optimization: you would feed the scalar reward to your search algorithm.

    return results

# ---------------------------------------------------------------------------
# Optional plotting helper for counts and exact probabilities
# ---------------------------------------------------------------------------

def plot_counts_vs_probs(counts, probs=None, title="Measurement results", show=True):
    """
    Plot histogram of shot counts and overlay exact probabilities if provided.
    counts: array of length 3
    probs: optional array of exact probabilities
    """
    counts = np.asarray(counts)
    n = counts.sum()
    labels = ["|0⟩", "|1⟩", "|2⟩"]
    fig, ax = plt.subplots(figsize=(5,3.5))
    ax.bar(labels, counts, alpha=0.7)
    ax.set_ylabel("Counts")
    ax.set_title(title)
    if probs is not None:
        # overlay expected counts
        expected = probs * n
        for i, e in enumerate(expected):
            ax.plot([i], [e], marker='x', color='k', markersize=8)
            ax.text(i, e + max(n*0.01, 1), f"{probs[i]:.3f}", ha='center', va='bottom')
    ax.grid(axis='y', alpha=0.25)
    if show:
        plt.show()
    return fig, ax