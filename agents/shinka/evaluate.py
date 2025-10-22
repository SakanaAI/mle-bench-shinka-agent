import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from mlebench.registry import registry
from shinka.core import run_shinka_eval


AGENT_DIR = os.environ.get("AGENT_DIR", "/home/agent")
PRIVATE_DATA_DIR = "/private/data"
# COMPETITION_ID is populated for us at container runtime
COMPETITION_ID = os.getenv("COMPETITION_ID")
ANS_VAL_PATH = f"{AGENT_DIR}/validation_answer.csv"


def write_val_perf(val_perf):
    open(f"{AGENT_DIR}/val_perf.txt", "w").write(str(float(val_perf)))


def read_val_perf():
    if not os.path.isfile(f"{AGENT_DIR}/val_perf.txt"):
        return 0
    content = open(f"{AGENT_DIR}/val_perf.txt", "r").read()
    return float(content)


def get_competition():
    new_registry = registry.set_data_dir(Path(PRIVATE_DATA_DIR))
    return new_registry.get_competition(COMPETITION_ID)


def grade_validation(submission_val_path: str):
    if not os.path.isfile(ANS_VAL_PATH):
        return False, f"Validation answer file not found at: {ANS_VAL_PATH}"

    submission_val_df = pd.read_csv(submission_val_path)
    ans_val_ds = pd.read_csv(ANS_VAL_PATH)

    competition = get_competition()
    try:
        val_perf = competition.grader.grade_fn(submission_val_df, ans_val_ds)
    except Exception as e:
        return False, f"The following error occured in the grading function: {str(e)}"

    write_val_perf(val_perf)
    return True, val_perf


# -------------------------------
# Validation of run result
# -------------------------------


def validate_submission(model, split: str):
    try:
        submission_val_path = model.make_submission(split)
    except Exception as e:
        return (
            False,
            f"Calling `model.make_submission('{split}')` causes the following error: {str(e)}",
        )
    if not os.path.isfile(submission_val_path):
        return False, f"Submission file not found at: {submission_val_path}"
    return True, submission_val_path


def _validate_model(model) -> Tuple[bool, Optional[str]]:
    """Basic validation: the run result should be a model-like object."""
    if model is None:
        return False, "train_model returned None"
    if not hasattr(model, "predict"):
        return False, "train_model result has no predict(...) method"
    if not hasattr(model, "make_submission"):
        return False, "train_model result has no make_submission(...) method"

    is_passed, msg = validate_submission(model, "validation")
    if not is_passed:
        return is_passed, msg
    submission_val_path = msg

    is_passed, msg = validate_submission(model, "test")
    if not is_passed:
        return is_passed, msg

    is_passed, msg = grade_validation(submission_val_path)
    if not is_passed:
        return is_passed, msg

    return True, "train_model returns a functional model instance"


# -------------------------------
# Aggregation and submission writing
# -------------------------------


def _aggregate_and_write_submission(
    results: List[Any], results_dir: str
) -> Dict[str, Any]:
    """
    Build a submission matching the sample_submission schema if available,
    and compute simple local metrics if a validation split exists.
    """
    os.makedirs(results_dir, exist_ok=True)

    model = results[0]
    submission_test_path = model.make_submission("test")
    print(f"Test submission saved to: {submission_test_path}")

    public_metrics: Dict[str, Any] = {}

    metrics: Dict[str, Any] = {
        "combined_score": float(read_val_perf()),
        "public": public_metrics,
        "private": {
            "submission_path": submission_test_path,
        },
    }

    return metrics


# -------------------------------
# CLI + Runner glue
# -------------------------------


def _get_mle_bench_kwargs(run_index: int) -> Dict[str, Any]:
    # Pass a seed by default; target code may optionally use it
    return {"seed": run_index + 1}


def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    os.environ["RESULTS_DIR"] = results_dir

    data_dir = os.environ.get("DATA_DIR", "/home/data")
    agent_dir = os.environ.get("AGENT_DIR", "/home/agent")

    def _aggregator_with_context(runs: List[Any]) -> Dict[str, Any]:
        return _aggregate_and_write_submission(runs, results_dir=results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_mle_bench",
        num_runs=1,
        get_experiment_kwargs=_get_mle_bench_kwargs,
        validate_fn=_validate_model,
        aggregate_metrics_fn=_aggregator_with_context,
    )

    if correct:
        print("Evaluation completed successfully.")
    else:
        print(f"Evaluation failed: {error_msg}")

    print("Metrics summary:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLE-Bench evaluator using shinka.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_mle_bench')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_mle_bench",
        help="Dir to save results",
    )
    args = parser.parse_args()
    print("Running Shinka evaluation.py")
    main(args.program_path, args.results_dir)
