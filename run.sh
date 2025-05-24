#!/bin/bash

set -ex

./scripts/create_notebooks.py
./scripts/assemble_results.py
./scripts/generate_results_table.py
./scripts/compare_notebooks.py
./scripts/parse_comparisons.py