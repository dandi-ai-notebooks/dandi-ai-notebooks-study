#!/bin/bash

# NOTEBOOK1=notebooks/dandisets/001349/0.250520.1729/90f7cc7c/gpt-4.1/h-1/notebook.ipynb
# NOTEBOOK2=notebooks/dandisets/001349/0.250520.1729/skip-explore/gpt-4.1/h-1/notebook.ipynb
# OUTPUT_PATH=tmp.txt

NOTEBOOK1=notebooks/dandisets/000563/0.250311.2145/75ecdb7b/gpt-4.1/h-1/notebook.ipynb
NOTEBOOK2=notebooks/dandisets/000563/0.250311.2145/skip-explore/gpt-4.1/h-1/notebook.ipynb
OUTPUT_PATH=tmp.txt

python scripts/compare_notebooks.py --notebook1-path $NOTEBOOK1 --notebook2-path $NOTEBOOK2 --output-path $OUTPUT_PATH --model openai/gpt-4.1