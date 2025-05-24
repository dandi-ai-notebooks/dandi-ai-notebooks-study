#!/usr/bin/env python

import yaml
import os
from pathlib import Path
from compare_notebooks import main as compare_notebook_main

def get_notebook_pairs(config):
    """Find pairs of notebooks to compare."""
    notebooks = config['notebooks']

    # First, organize notebooks by their key properties
    notebook_dict = {}
    for nb in notebooks:
        key = (
            nb['dandiset_id'],
            nb['dandiset_version'],
            nb['prompt'],
            nb['model']
        )
        skip_explore = nb.get('skip_explore', False)
        if key not in notebook_dict:
            notebook_dict[key] = {'skip_explore': None, 'normal': None}

        if skip_explore:
            notebook_dict[key]['skip_explore'] = nb
        else:
            notebook_dict[key]['normal'] = nb

    # Find valid pairs where both skip_explore and normal notebooks exist
    pairs = []
    for key, value in notebook_dict.items():
        if value['skip_explore'] and value['normal']:
            pairs.append((value['skip_explore'], value['normal']))

    return pairs

def get_notebook_path(notebook):
    """Get the path to a notebook based on its configuration."""
    dandiset_id = notebook['dandiset_id']
    version = notebook['dandiset_version']
    chat_id = notebook.get('chat_id', None)
    model = notebook['model'].split('/')[-1]
    prompt = notebook['prompt']

    # Use skip-explore or first 8 chars of chat_id
    chat_id_part = 'skip-explore' if notebook.get('skip_explore', False) else chat_id[:8]

    return f"notebooks/dandisets/{dandiset_id}/{version}/{chat_id_part}/{model}/{prompt}/notebook.ipynb"

def get_comparison_path(notebook):
    """Get the path where the comparison result should be saved."""
    dandiset_id = notebook['dandiset_id']
    version = notebook['dandiset_version']
    chat_id = notebook.get('chat_id', None)
    model = notebook['model'].split('/')[-1]
    prompt = notebook['prompt']

    chat_id_part = chat_id[:8]

    # Create the comparison path
    return f"notebook_comparisons/dandisets/{dandiset_id}/{version}/{chat_id_part}/{model}/{prompt}/comparison_with_skip_explore.txt"

def main():
    # Load configuration from notebooks.yaml
    with open('notebooks.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Get pairs of notebooks to compare
    pairs = get_notebook_pairs(config)

    print(f"Found {len(pairs)} pairs of notebooks to compare")

    # Process each pair
    for skip_explore_nb, normal_nb in pairs:
        comparison_path = get_comparison_path(normal_nb)

        # Skip if comparison already exists
        if os.path.exists(comparison_path):
            print(f"Skipping existing comparison for dandiset {normal_nb['dandiset_id']}")
            continue

        print(f"\nProcessing comparison for dandiset {normal_nb['dandiset_id']}...")

        # Get notebook paths
        notebook1_path = get_notebook_path(skip_explore_nb)
        notebook2_path = get_notebook_path(normal_nb)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(comparison_path), exist_ok=True)

        # Run the comparison
        print(f"Comparing notebooks:")
        print(f"  Notebook 1 (skip_explore): {notebook1_path}")
        print(f"  Notebook 2: {notebook2_path}")
        print(f"  Output: {comparison_path}")

        try:
            import sys
            old_argv = sys.argv
            sys.argv = [
                'compare_notebooks.py',
                '--notebook1-path', notebook1_path,
                '--notebook2-path', notebook2_path,
                '--output-path', comparison_path,
                '--model', 'openai/gpt-4.1'
            ]
            compare_notebook_main()
            sys.argv = old_argv
            print("Comparison completed successfully")
        except Exception as e:
            print(f"Error comparing notebooks: {str(e)}")

if __name__ == '__main__':
    main()
