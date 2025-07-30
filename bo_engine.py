# bo_engine.py
# Multi-objective Bayesian Optimization engine using Xopt for LPA GUI

from xopt import Xopt
from xopt.vocs import VOCS


# def compute_objective(metrics):
#     """
#     Deprecated: Now using true multi-objective. This remains as reference if needed.
#     """
#     return 0.0


# bo_engine.py
# Multi-objective Bayesian Optimization engine using Xopt for LPA GUI

from xopt import Xopt
from xopt.vocs import VOCS


def create_xopt(active_params, evaluator, bounds, resume_file=None, acquisition_mode=None):
    """
    Initialize a multi-objective Xopt instance for three objectives:
    - spectra_score (maximize)
    - charge       (maximize)
    - stability    (maximize)

    Parameters:
    - active_params: list of parameter names (e.g. ['plasma_z', 'blade_x'])
    - evaluator: callable mapping list of param dicts → list of metric dicts
                 each dict must have keys 'spectra_score','charge','stability'
    - bounds: dict of parameter bounds, e.g. {'plasma_z': [0, 10], ...}
    - resume_file: optional JSON filename to load a prior Xopt run
    - acquisition_mode: one of "explore", "balanced", or "exploit"
    """
    # 1) Validate bounds
    for p in active_params:
        if p not in bounds:
            raise ValueError(f"Parameter '{p}' missing from bounds.")
        b = bounds[p]
        if not (isinstance(b, (list, tuple)) and len(b) == 2):
            raise ValueError(f"Bounds for '{p}' must be a list or tuple of length 2.")

    print(f"Creating Xopt with {len(active_params)} active parameters: {active_params}")

    # 2) Build VOCS and grid for snapping
    vocs_vars = {}
    param_grids = {}
    for p in active_params:
        b = bounds[p]
        if isinstance(b, (list, tuple)) and len(b) == 3:
            lo, hi, step = b
            vocs_vars[p] = [lo, hi]  # Always use continuous for VOCS
            # Build grid for snapping (exclude edges)
            n = int(round((hi - lo) / step)) - 1
            grid = [lo + (i + 1) * step for i in range(n)]
            param_grids[p] = grid
        else:
            vocs_vars[p] = b

    vocs = VOCS(
        variables={p: vocs_vars[p] for p in active_params},
        objectives={
            "spectra_score": "MAXIMIZE",
            "charge":         "MAXIMIZE",
            "stability":      "MAXIMIZE"
        }
    )

    print(f"VOCS created with variables: {vocs.variables}")

    def snap_to_grid(params, param_grids):
        """
        Snaps each parameter in params to the nearest value in its defined grid.
        If no grid is defined for a parameter, it returns the original value.

        Parameters:
        ----------
        - params: dict of parameter values to snap
        - param_grids: dict of parameter grids for snapping

        Returns:
        -------
        - dict of snapped parameter values
        """
        snapped = {}
        for k, v in params.items():
            if k in param_grids and param_grids[k]:
                grid = param_grids[k]
                # Find nearest grid value
                snapped[k] = min(grid, key=lambda x: abs(x - v))
            else:
                snapped[k] = v
        return snapped

    # 3) Choose generator based on acquisition_mode
    """
    Selects and configures the Bayesian optimization generator based on acquisition_mode.
    Modes:
      - 'explore': Prioritizes exploration using BayesianExplorationGenerator.
      - 'exploit': Scalarizes objectives and uses ExpectedImprovementGenerator for exploitation.
      - 'balanced' or default: Uses MOBOGenerator for multi-objective optimization.
    """
    if acquisition_mode == "explore":
        """
        Exploration mode: Uses BayesianExplorationGenerator to prioritize sampling unexplored regions.
        """
        from xopt.generators.bayesian import BayesianExplorationGenerator
        gen = BayesianExplorationGenerator(vocs=vocs)
        print("Using BayesianExplorationGenerator for exploration mode.")

    elif acquisition_mode == "exploit":
        """
        Exploitation mode: Scalarizes objectives into a single 'overall' score using geometric mean,
        then uses ExpectedImprovementGenerator to focus on maximizing this scalarized objective.
        """
        scalar_vocs = VOCS(
            variables=vocs.variables,
            objectives={"overall": "MAXIMIZE"},
        )

        def scalar_eval(param_list):
            """
            Scalarizes multi-objective results into a single 'overall' score using the geometric mean.
            Returns a list of dicts with the 'overall' key for each parameter set.
            """
            mos = evaluator(param_list) # Call the user-supplied evaluator and get multi-objective results
            out = []
            # Iterate over the multi-objective results
            # and compute the geometric mean for 'overall' score !!!Can be changed!!!
            for m in mos:
                sc, ch, st = m.get('spectra_score'), m.get('charge'), m.get('stability')
                if None in (sc, ch, st):
                    overall = None
                else:
                    overall = (sc * ch * st) ** (1.0/3.0)
                out.append({"overall": overall})
            return out

        from xopt.generators.bayesian import ExpectedImprovementGenerator
        gen = ExpectedImprovementGenerator(
            vocs=scalar_vocs,
            evaluator=scalar_eval
        )
        print("Using ExpectedImprovementGenerator for exploitation mode.")

    else:  # balanced or default
        """
        Balanced mode (default): Uses MOBOGenerator for true multi-objective optimization.
        The reference point is set to zero for all objectives.
        """
        from xopt.generators.bayesian import MOBOGenerator
        gen = MOBOGenerator(
            vocs=vocs,
            reference_point={
                "spectra_score": 0.0,
                "charge":        0.0,
                "stability":     0.0
            }
        )
        print("Using MOBOGenerator for balanced mode.")

    # 4) Wrap the user-supplied evaluator to check for missing keys
    def wrapped_evaluator(param_list):
        """
        Checks the output of the user-supplied evaluator for missing required keys.
        Snaps each parameter to the nearest grid value (if grid is defined for that parameter).
        Handles both single dict and list-of-dicts input for Xopt compatibility.
        Returns a single dict if input is a dict, or a list of dicts if input is a list.
        """
        # If param_list is a dict, treat as single evaluation
        if isinstance(param_list, dict):
            snapped = snap_to_grid(param_list, param_grids)
            results = evaluator([snapped])
            if not isinstance(results, list) or len(results) != 1:
                raise TypeError("Evaluator must return a list of one dict for single input.")
            res = results[0]
            if not isinstance(res, dict):
                raise TypeError("Evaluator result is not a dict.")
            for key in ("spectra_score", "charge", "stability"):
                if key not in res:
                    raise ValueError(f"Evaluator result missing required key '{key}'.")
            return res
        # If param_list is a list, treat as batch evaluation
        snapped_list = [snap_to_grid(p, param_grids) for p in param_list]
        results = evaluator(snapped_list)
        if not isinstance(results, list):
            raise TypeError("Evaluator must return a list of dicts for batch input.")
        for i, res in enumerate(results):
            if not isinstance(res, dict):
                raise TypeError(f"Evaluator result at index {i} is not a dict.")
            for key in ("spectra_score", "charge", "stability"):
                if key not in res:
                    raise ValueError(f"Evaluator result at index {i} missing required key '{key}'.")
        return results

    # 5) Build or resume the Xopt instance
    """
    Returns a new Xopt instance, or resumes from file if resume_file is provided.
    """
    if resume_file:
        return Xopt.from_file(resume_file)
    else:
        # Xopt expects evaluator as a dict: {'function': <callable>}
        return Xopt(vocs=vocs, evaluator={'function': wrapped_evaluator}, generator=gen)
