import argparse
from functools import partial
import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
import yaml

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
# FOR DEBUGGING, PLEASE REMOVE LATER
ANS_TEST_PATH = glob(f"/private/data/{COMPETITION_ID}/prepared/private/*.csv")[0]


def write_perf(perf, results_dir: str, split: str):
    open(f"{results_dir}/{split}_perf.txt", "w").write(str(float(perf)))


def read_perf(results_dir: str, split: str):
    if not os.path.isfile(f"{results_dir}/{split}_perf.txt"):
        return -10
    content = open(f"{results_dir}/{split}_perf.txt", "r").read()
    return float(content)


def grade(submission_path: str, results_dir: str, split: Literal["validation", "test"]):
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

    write_perf(perf, results_dir, split)
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


def _validate_model(model, results_dir: str) -> Tuple[bool, Optional[str]]:
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
        is_passed, msg = grade(submission_path, results_dir, split)
        if not is_passed:
            return is_passed, msg

    return True, "train_model returns a functional model instance"


def generate_text_feedback(code: str, rubrics: list[str]):
    raise NotImplementedError
    # TODO: manage llm calls here
    for rubric in rubrics:
        llm_judge(code, rubric)


def extract_code_proposal(path: str):
    assert path.endswith(".py"), f"Only .py files are supported, got {path}"
    content = open(path, "r").read()
    start_marker = "# EVOLVE-BLOCK-START"
    end_marker = "# EVOLVE-BLOCK-END"
    start_idx = content.find(start_marker)
    if start_idx == -1:
        raise ValueError(f"Start marker '{start_marker}' not found in {path}")
    start_idx += len(start_marker)

    end_idx = content.find(end_marker, start_idx)
    if end_idx == -1:
        raise ValueError(f"End marker '{end_marker}' not found in {path}")

    block = content[start_idx:end_idx]
    return block.strip()


# -------------------------------
# Aggregation and submission writing
# -------------------------------


def _aggregate_and_write_submission(
    results: List[Any],
    use_text_feedback: bool,
    program_path: str,
    results_dir: str,
) -> Dict[str, Any]:
    """
    Build a submission matching the sample_submission schema if available,
    and compute simple local metrics if a validation split exists.
    """
    print("Aggregating metrics...")
    assert len(results) == 1
    model = results[0]
    text_feedback = ""
    if use_text_feedback:
        print("Generating text feedback from rubrics")
        try:
            code = extract_code_proposal(program_path)
            print(f"{code=}")
            with open(f"{AGENT_DIR}/llm_judge_rubrics.yaml") as f:
                rubrics = yaml.safe_load(f)["rubrics"]
            print(f"{rubrics=}")
            text_feedback = generate_text_feedback(code, rubrics)
        except Exception as e:
            raise RuntimeError(
                f"Error during text feedback generation: {str(e)}"
            ) from e
            text_feedback = ""

    val_perf = float(read_perf(results_dir, "validation"))
    test_perf = float(read_perf(results_dir, "test"))
    metrics: Dict[str, Any] = {
        "combined_score": val_perf,
        "public": {},
        "private": {
            "test_perf": test_perf,
            "test_train_perf_diff": test_perf - val_perf,
        },
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

    def _aggregator_with_context(runs: List[Any]) -> Dict[str, Any]:
        return _aggregate_and_write_submission(
            runs, use_text_feedback, program_path, results_dir
        )

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_mle_bench",
        num_runs=1,
        get_experiment_kwargs=_get_mle_bench_kwargs,
        validate_fn=partial(_validate_model, results_dir=results_dir),
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
