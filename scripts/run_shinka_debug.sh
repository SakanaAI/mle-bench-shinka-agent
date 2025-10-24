#!/bin/bash

# we can load the image on the compute nodes like so
docker load -i mlebench-env.tar.gz

export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent
docker build --platform=linux/amd64 -t shinka agents/shinka/ --build-arg SUBMISSION_DIR=$SUBMISSION_DIR --build-arg LOGS_DIR=$LOGS_DIR --build-arg CODE_DIR=$CODE_DIR --build-arg AGENT_DIR=$AGENT_DIR


timestamp=$(date +"%Y%m%d_%H%M")
competition_name="${competition%.*}"
echo "Running competition: ${1}"
echo "-----------------------------------"
# uv run run_agent.py --agent-id shinka --competition-set experiments/splits/spaceship-titanic.txt
python run_agent.py --agent-id shinka/debug --competition-set "experiments/splits/${1}.txt"
