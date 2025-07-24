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

    # 2) Build VOCS
    vocs = VOCS(
        variables={p: bounds[p] for p in active_params},
        objectives={
            "spectra_score": "MAXIMIZE",
            "charge":         "MAXIMIZE",
            "stability":      "MAXIMIZE"
        }
    )

    # 3) Choose generator based on acquisition_mode
    if acquisition_mode == "explore":
        from xopt.generators.bayesian import BayesianExplorationGenerator
        gen = BayesianExplorationGenerator(vocs=vocs)

    elif acquisition_mode == "exploit":
        # scalarize into a single 'overall' objective using geometric mean
        scalar_vocs = VOCS(
            variables=vocs.variables,
            objectives={"overall": "MAXIMIZE"},
        )

        def scalar_eval(param_list):
            mos = evaluator(param_list)
            out = []
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

    else:  # balanced or default
        from xopt.generators.bayesian import MOBOGenerator
        gen = MOBOGenerator(
            vocs=vocs,
            reference_point={
                "spectra_score": 0.0,
                "charge":        0.0,
                "stability":     0.0
            }
        )

    # 4) Wrap the user-supplied evaluator to check for missing keys
    def wrapped_evaluator(param_list):
        results = evaluator(param_list)
        for res in results:
            for key in ("spectra_score", "charge", "stability"):
                if key not in res:
                    print(f"Warning: evaluator result missing '{key}'")
        return results

    # 5) Build or resume the Xopt instance
    if resume_file:
        return Xopt.from_file(resume_file)
    else:
        return Xopt(vocs=vocs, evaluator=wrapped_evaluator, generator=gen)
