import argparse
import os
from glob import glob
from pathlib import Path

from dotenv import load_dotenv


from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

env_path = Path(__file__).parent / ".env"

load_dotenv(dotenv_path=env_path, override=True)

TASK_SYS_MSG_TEMPLATE = (
    """
You are a Kaggle grandmaster, world-class machine learning engineer, and you are very good at building models and statistical methods.
Now, you are participating in a Kaggle competition. In order to win this competition, you need to refine the code block for better performance. Here is the problem statement.
<problem_statement>

{task_desc}

<end_of_problem_statement>

---

Your goal is to improve the performance of the program by suggesting improvements.

If you're given placeholder code, prioritize implementing the correct and functional data pipeline first, i.e., `make_submission`, `prepare_data`, `load_data` functions with a simple machine learning model. Once the code works (validated), focus more on improving the model performance.

When you use a data processing pipeline, make sure that the same pipeline is applied consistently across splits.
Also, make sure to avoid data leakage in the data pipeline.

Avoid using try-except statements.
If you know that a part of the code doesn't work, remove it instead of wrapping it with try-except statements.

It is usually a good idea to handle class imbalance.

The runtime is limited to one hour. Make sure that the code can be executed within one hour.

You will be given a set of performance metrics for the program.
Your goal is to maximize the `combined_score` of the program.
Try diverse approaches to solve the problem."""
).strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Shinka evolutionary agent for MLE-Bench."
    )
    parser.add_argument(
        "--num_generations", type=int, default=100, help="Number of generations to run."
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    debug = os.environ.get("DEBUG", False)
    use_text_feedback = os.environ.get("USE_TEXT_FEEDBACK", False)
    if debug:
        print("Debug mode enabled for Shinka evolution runner.")

    AGENT_DIR = os.environ.get("AGENT_DIR")
    DATA_DIR = os.environ.get("DATA_DIR")

    if not AGENT_DIR or not DATA_DIR:
        raise EnvironmentError(
            "Environment variables `AGENT_DIR` and `DATA_DIR` must be set."
        )

    description_file = Path(DATA_DIR) / "description.md"
    task_desc = description_file.read_text()

    mle_bench_task_sys_msg = TASK_SYS_MSG_TEMPLATE.format(task_desc=task_desc)
    data_directory_files = f"{glob(f'{DATA_DIR}/*')=}"
    if data_directory_files:
        mle_bench_task_sys_msg += (
            f"\n\nHere's the files listed in {{DATA_DIR}}: `{data_directory_files}`"
        )
    print("##### SYSTEM MESSAGE #####")
    print(mle_bench_task_sys_msg)
    print("=" * 60)

    llm_models = ["gemini-2.5-flash", "o4-mini", "gpt-5-mini"]

    if os.environ.get("USE_PREMIUM_MODELS", False):
        llm_models += [
            # "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", # this model doesn't work!?
            # got error `Model bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0 not supported.`
            "gemini-2.5-pro",
            "gpt-5",
        ]

    print(f"Included LLMs: {llm_models}")

    job_config = LocalJobConfig(eval_program_path=f"{AGENT_DIR}/evaluate.py")
    db_config = DatabaseConfig(
        db_path="evolution_db.sqlite",
        num_islands=2,
        archive_size=50,
        # Inspiration parameters
        elite_selection_ratio=0.3,
        num_archive_inspirations=2,
        num_top_k_inspirations=2,
        # Island migration parameters
        migration_interval=10,
        migration_rate=0.1,  # chance to migrate program to random island
        island_elitism=True,  # Island elite is protected from migration
        enforce_island_separation=True,
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )

    evo_config = EvolutionConfig(
        task_sys_msg=mle_bench_task_sys_msg,
        patch_types=["diff", "full", "cross"],
        patch_type_probs=[0.6, 0.3, 0.1],
        num_generations=args.num_generations,
        max_parallel_jobs=1,
        max_patch_resamples=3,
        max_patch_attempts=3,
        job_type="local",
        language="python",
        llm_models=llm_models,
        llm_kwargs=dict(
            temperatures=[0.0, 0.5, 1.0],
            max_tokens=2**16,  # 64k
        ),
        meta_rec_interval=5,
        meta_llm_models=["gpt-5-mini"],
        meta_llm_kwargs=dict(
            temperatures=[0.0],
            max_tokens=2**16,  # 64k
        ),
        init_program_path=f"{AGENT_DIR}/initial.py",
        results_dir=f"{AGENT_DIR}/results_mle_bench",
        max_novelty_attempts=3,
        use_text_feedback=use_text_feedback,
        llm_dynamic_selection="ucb1",
        llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
    )

    if debug:
        evo_config.num_generations = 5
        evo_config.llm_models = ["gpt-5-mini"]

    print("Running shinka agent...")
    print(f"EVO CONFIG: {evo_config}")
    print(f"JOB CONFIG: {job_config}")
    print(f"DB CONFIG: {db_config}")

    # Run evolution with defaults
    runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
    )
    runner.run()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
