import os
from pathlib import Path

from dotenv import load_dotenv


from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

env_path = Path(__file__).parent / ".env"

load_dotenv(dotenv_path=env_path, override=True)

print(os.environ.get("OPENAI_API_KEY"))

AGENT_DIR = os.environ.get("AGENT_DIR")
DATA_DIR = os.environ.get("DATA_DIR")

description_file = f"{DATA_DIR}/description.md"
# if os.environ.get("OBFUSCATE", False):
#     description_file = f"{DATA_DIR}/description_obfuscated.md"
task_desc = open(description_file).read()
print("##### TASK DESCRIPTION #####")
print(task_desc)
print("=" * 60)

mle_bench_task_sys_msg = (
    """
You are a Kaggle grandmaster, world-class machine learning engineer, and you are very good at building models and statistical methods.
Now, you are participating in a Kaggle competition. In order to win this competition, you need refine the code block for better performance. Here is the problem statement:

{task_desc}

Your goal is to improve the performance of the program by suggesting improvements.

You will be given a set of performance metrics for the program.
Your goal is to maximize the `combined_score` of the program.
Try diverse approaches to solve the problem."""
).strip()

mle_bench_task_sys_msg = mle_bench_task_sys_msg.format(task_desc=task_desc)

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
    num_generations=100,
    max_parallel_jobs=1,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=[
        # "gemini-2.5-pro",
        "gemini-2.5-flash",
        "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        "o4-mini",
        "gpt-5-mini",
        # "gpt-5",
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],
        max_tokens=32768,
    ),
    meta_rec_interval=5,
    meta_llm_models=["gpt-5-mini"],
    meta_llm_kwargs=dict(
        temperatures=[0.0],
        max_tokens=32768,
    ),
    init_program_path=f"{AGENT_DIR}/initial.py",
    results_dir=f"{AGENT_DIR}/results_mle_bench",
    max_novelty_attempts=3,
    use_text_feedback=False,
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
)


# check file exist
# print(os.listdir("/home/"))
# print(os.listdir("/home/data"))
# print(os.path.isfile("/home/instructions.txt"))
# intx = open("/home/instructions.txt").read()
# print("##### INSTRUCTION #####")
# print(intx)
# print("=" * 60)


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
