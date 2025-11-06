import argparse
import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from mlebench.registry import registry

from shinka.core import run_shinka_eval


# we cant access the leaderboard from inside the docker
# we manually checked and add each competition here
IS_LOWER_BETTER = {
    "spaceship-titanic": False,
    "spooky-author-identification": True,
    "nomad2018-predict-transparent-conductors": True,
    "random-acts-of-pizza": False,
}

AGENT_DIR = os.environ.get("AGENT_DIR", "/home/agent")
PRIVATE_DATA_DIR = "/private/data"
# COMPETITION_ID is populated for us at container runtime
COMPETITION_ID = os.getenv("COMPETITION_ID")
ANS_VAL_PATH = f"{AGENT_DIR}/validation_answer.csv"
ANS_TEST_PATH = glob(f"/private/data/{COMPETITION_ID}/prepared/private/*.csv")[0]


def write_perf(perf, split):
    open(f"{AGENT_DIR}/{split}_perf.txt", "w").write(str(float(perf)))


def read_perf(split):
    if not os.path.isfile(f"{AGENT_DIR}/{split}_perf.txt"):
        return -10
    content = open(f"{AGENT_DIR}/{split}_perf.txt", "r").read()
    return float(content)


def grade(submission_path: str, split: Literal["validation", "test"]):
    competition = registry.get_competition(COMPETITION_ID)
    ans_path = ANS_VAL_PATH if split == "validation" else ANS_TEST_PATH
    if not os.path.isfile(ans_path):
        return False, f"Answer file not found at: {ans_path}"

    submission_df = pd.read_csv(submission_path)
    ans_ds = pd.read_csv(ans_path)

    try:
        perf = competition.grader.grade_fn(submission_df, ans_ds)
    except Exception as e:
        return False, f"The following error occured in the grading function: {str(e)}"

    if IS_LOWER_BETTER[COMPETITION_ID]:
        perf *= -1

    write_perf(perf, split)
    return True, perf


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
    print("Validating trained model...")
    if model is None:
        return False, "train_model returns None"
    if not hasattr(model, "make_submission"):
        return False, "train_model result has no make_submission(...) method"

    for split in ["validation", "test"]:
        print("Validating submission (validation split)")
        is_passed, msg = validate_submission(model, split)
        if not is_passed:
            return is_passed, msg
        submission_path = msg
        print(f"Grading {split} submission...")
        is_passed, msg = grade(submission_path, split)
        if not is_passed:
            return is_passed, msg

    return True, "train_model returns a functional model instance"


def generate_text_feedback(code: str):
    pass


# -------------------------------
# Aggregation and submission writing
# -------------------------------


def _aggregate_and_write_submission(
    results: List[Any], use_text_feedback: bool, results_dir: str
) -> Dict[str, Any]:
    """
    Build a submission matching the sample_submission schema if available,
    and compute simple local metrics if a validation split exists.
    """
    print("Aggregating metrics...")
    model = results[0]
    text_feedback = ""
    if use_text_feedback:
        # TODO: add llm judge here
        text_feedback = generate_text_feedback(...)

    metrics: Dict[str, Any] = {
        "combined_score": float(read_perf("validation")),
        "public": {},
        "private": {"test_perf": float(read_perf("test"))},
        "text_feedback": text_feedback,
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
    use_text_feedback = os.environ.get("USE_TEXT_FEEDBACK", False)

    data_dir = os.environ.get("DATA_DIR", "/home/data")
    agent_dir = os.environ.get("AGENT_DIR", "/home/agent")

    def _aggregator_with_context(runs: List[Any]) -> Dict[str, Any]:
        return _aggregate_and_write_submission(
            runs, use_text_feedback, results_dir=results_dir
        )

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
