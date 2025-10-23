#!/bin/bash
method_list=(
    # parallel-llm-mcts-pymc
    parallel-llm-mcts-thompson
    # parallel-monkey
)
competition_list=(
#   spaceship-titanic.txt
  nomad2018-predict-transparent-conductors.txt
#   text-normalization-challenge-english-language.txt
#   tabular-playground-series-may-2022.txt
  random-acts-of-pizza.txt
  spooky-author-identification.txt
#   jigsaw-toxic-comment-classification-challenge.txt
#   text-normalization-challenge-russian-language.txt
#   new-york-city-taxi-fare-prediction.txt
)

# we can load the image on the compute nodes like so
docker load -i mlebench-env.tar.gz

export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent
docker build --platform=linux/amd64 -t shinka agents/shinka/ --build-arg SUBMISSION_DIR=$SUBMISSION_DIR --build-arg LOGS_DIR=$LOGS_DIR --build-arg CODE_DIR=$CODE_DIR --build-arg AGENT_DIR=$AGENT_DIR

for competition in "${competition_list[@]}"; do
    timestamp=$(date +"%Y%m%d_%H%M")
    competition_name="${competition%.*}"
    group_dir="${timestamp}_${competition_name}_shinka"
    echo "Running competition: ${1}"
    echo "Group dir: ${group_dir}"
    echo "-----------------------------------"
    # uv run run_agent.py --agent-id shinka --competition-set experiments/splits/spaceship-titanic.txt
    python run_agent.py --agent-id shinka --competition-set "experiments/splits/${competition}" --run-group ${group_dir}
    python experiments/make_submission.py --metadata "runs/${group_dir}/metadata.json" --output "runs/${group_dir}/submission.jsonl"
    submission_csv_list=$(find "runs/${group_dir}" -type f -name 'submission.csv')
    for submission_csv in ${submission_csv_list}; do
        save_dir=$(dirname "${submission_csv}")
        jq -c --arg path "${submission_csv}" '
        .submission_path = $path
        ' "runs/${group_dir}/submission.jsonl" > "${save_dir}/submission.jsonl"
        mlebench grade --submission "${save_dir}/submission.jsonl" --output-dir "${save_dir}"
    # Delete docker
    # docker ps -a --filter "name=competition-${competition}" -q | xargs docker stop
    # docker ps -a --filter "name=competition-${competition}" -q | xargs docker rm
    done
done



