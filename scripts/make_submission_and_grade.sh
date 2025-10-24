#!/bin/bash
uv run python experiments/make_submission.py --metadata runs/${1}/metadata.json --output runs/${1}/submission.jsonl
uv run mlebench grade --submission runs/${1}/submission.jsonl --output-dir runs/${1}