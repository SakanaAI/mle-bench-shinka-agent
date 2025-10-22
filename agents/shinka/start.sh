#!/bin/bash
export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent
export DATA_DIR=/home/data

set -x # Print commands and their arguments as they are executed
cd ${AGENT_DIR}

eval "$(conda shell.bash hook)" # make conda available to the shell
conda activate agent

# Check that the agent doesn't have permissions to read private dir
ls /private
# ls: cannot open directory '/private': Permission denied

# Check that the agent does have permissions to read/write everything in /home
ls /home/data
# touch $CODE_DIR/code.py
# touch $LOGS_DIR/run.log
# touch $AGENT_DIR/shinka_was_here.txt
cat /home/instructions.txt

# Use the environment-provided grading server to validate our submission
# bash /home/validate_submission.sh /home/submission/submission.csv

# symbolic linking
ln -s ${LOGS_DIR} ${AGENT_DIR}/results_mle_bench

# Ensure `${AGENT_DIR}/sample_submission.csv` exists
if [ -f "${DATA_DIR}/sample_submission.csv" ]; then
  cp -f "${DATA_DIR}/sample_submission.csv" "${AGENT_DIR}/sample_submission.csv"
elif [ -f "${DATA_DIR}/sampleSubmission.csv" ]; then
  cp -f "${DATA_DIR}/sampleSubmission.csv" "${AGENT_DIR}/sample_submission.csv"
fi


# run with timeout, and print if timeout occurs
timeout $TIME_LIMIT_SECS python ${AGENT_DIR}/run_evo.py
if [ $? -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi


cp -rf ${AGENT_DIR}/results_mle_bench/best ${CODE_DIR}
cp  ${AGENT_DIR}/results_mle_bench/best/results/submission_test.csv ${SUBMISSION_DIR}/submission.csv
