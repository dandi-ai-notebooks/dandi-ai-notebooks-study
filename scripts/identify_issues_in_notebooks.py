#!/usr/bin/env python

import yaml
import os
from pathlib import Path
from identify_issues_in_notebook import main as identify_issues_in_notebook_main

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

def get_notebook_issues_path(notebook):
    """Get the path where the notebook issues should be saved."""
    dandiset_id = notebook['dandiset_id']
    version = notebook['dandiset_version']
    chat_id = notebook.get('chat_id', None)
    model = notebook['model'].split('/')[-1]
    prompt = notebook['prompt']

    chat_id_part = chat_id[:8] if chat_id else 'skip-explore'

    # Create the issues path
    return f"notebook_issues/dandisets/{dandiset_id}/{version}/{chat_id_part}/{model}/{prompt}/notebook_issues.txt"

def main():
    # Load configuration from notebooks.yaml
    with open('notebooks.yaml', 'r') as f:
        config = yaml.safe_load(f)

    notebooks = config['notebooks']

    print(f"Found {len(notebooks)} notebooks to process")

    # Process each pair
    for nb in notebooks:
        notebook_issues_path = get_notebook_issues_path(nb)

        # Skip if issues file already exists
        if os.path.exists(notebook_issues_path):
            print(f"Skipping existing notebook issues for dandiset {nb['dandiset_id']}")
            continue

        print(f"\nIdentifying issues for dandiset {nb['dandiset_id']}...")

        # Get notebook path
        notebook_path = get_notebook_path(nb)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(notebook_issues_path), exist_ok=True)

        # Run the process
        print(f"Identifying issues in notebook:")
        print(f"  Notebook: {notebook_path}")
        print(f"  Output: {notebook_issues_path}")

        try:
            import sys
            old_argv = sys.argv
            sys.argv = [
                'identify_issues_in_notebook.py',
                '--notebook-path', notebook_path,
                '--output-path', notebook_issues_path,
                '--model', 'anthropic/claude-3.7-sonnet'
            ]
            identify_issues_in_notebook_main()
            sys.argv = old_argv
            print("Process completed successfully")
        except Exception as e:
            print(f"Error processing notebook: {str(e)}")

if __name__ == '__main__':
    main()
