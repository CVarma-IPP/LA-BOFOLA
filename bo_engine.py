# bo_engine.py
# Multi-objective Bayesian Optimization engine using Xopt for LPA GUI

from xopt import Xopt
from xopt.vocs import VOCS


def compute_objective(metrics):
    """
    Deprecated: Now using true multi-objective. This remains as reference if needed.
    """
    return 0.0


def create_xopt(active_params, evaluator, bounds, resume_file=None, acquisition_mode=None):
    """
    Initialize a multi-objective Xopt instance for three objectives:
    - spectra_score (maximize): e.g. quality of the electron spectrum
    - charge (maximize): e.g. total charge measured
    - stability (maximize): e.g. shot-to-shot repeatability

    Parameters:
    - active_params: list of parameter names (e.g. ['plasma_z', 'blade_x'])
      These are the experimental knobs you want the optimizer to vary.
    - evaluator: callable that maps list of param dicts → list of metric dicts.
      This function should run the experiment (or simulation) for each set of parameters,
      and return a dict with keys: 'spectra_score', 'charge', 'stability'.
    - bounds: dict of parameter bounds, e.g. {'plasma_z': [0, 10], ...}
      These define the allowed range for each parameter.
    - resume_file: optional JSON for loading prior state (to continue a previous run)
    - acquisition_mode: optional string to control exploration vs. exploitation
      e.g. "explore", "balanced", "exploit"
    """
    # Validate that all active parameters have bounds defined and are well-formed
    for p in active_params:
        if p not in bounds:
            raise ValueError(f"Parameter '{p}' missing from bounds.")
        if not (isinstance(bounds[p], (list, tuple)) and len(bounds[p]) == 2):
            raise ValueError(f"Bounds for '{p}' must be a list or tuple of length 2.")

    # Define the optimization problem for Xopt:
    # - variables: the tunable parameters (knobs)
    # - objectives: what we want to maximize (spectra_score, charge, stability)
    vocs = VOCS(
        variables={p: bounds[p] for p in active_params},
        objectives={
            "spectra_score": "MAXIMIZE",
            "charge": "MAXIMIZE",
            "stability": "MAXIMIZE"
        },
    )

    # Prepare the configuration for the optimizer
    xopt_config = {
        "vocs": vocs,
        "evaluator": evaluator,
    }

    # If resuming from a previous optimization, load from file
    if resume_file:
        return Xopt.from_file(resume_file)

    # Choose the acquisition function (strategy for proposing new points)
    # based on the GUI slider: explore (try new things), exploit (refine best), or balanced
    if acquisition_mode:
        # Map GUI slider to BoTorch acquisition functions
        acq_map = {
            "explore":  "qUCB",     # Upper Confidence Bound: prioritizes exploration
            "balanced": "qNEHVI",   # Noisy Expected Hypervolume Improvement: balances
            "exploit":  "qEI"       # Expected Improvement: focuses on best-so-far
        }
        if acquisition_mode not in acq_map:
            print(f"Warning: Unknown acquisition_mode '{acquisition_mode}', defaulting to 'balanced' (qNEHVI).")
        afunc = acq_map.get(acquisition_mode, "qNEHVI")

        # For UCB, you can adjust beta to control exploration strength
        gen_opts = {}
        if afunc == "qUCB":
            gen_opts["generator_options"] = {"beta": 2.0}

        xopt_config["generator"] = {
            "name":                "botorch",
            "model":               "standard",
            "acquisition_function": afunc,
            **gen_opts
        }

    # Wrap the evaluator to warn if any required objectives are missing from the results
    def wrapped_evaluator(params_list):
        results = evaluator(params_list)
        for res in results:
            for key in ["spectra_score", "charge", "stability"]:
                if key not in res:
                    print(f"Warning: Evaluator result missing '{key}'.")
        return results

    xopt_config["evaluator"] = wrapped_evaluator

    # Create and return the Xopt optimizer object
    return Xopt(**xopt_config)
