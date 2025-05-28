#!/usr/bin/env python

import os
import json
import re
import yaml
from pathlib import Path
from typing import Dict, List, Any

def parse_issue_block(block: str) -> Dict[str, Any]:
    """Parse a single issue block into a dictionary."""
    # Extract each field using regex
    type = re.search(r'<type>(.*?)</type>', block)
    description = re.search(r'<description>(.*?)</description>', block)
    severity = re.search(r'<severity>(.*?)</severity>', block, re.DOTALL)

    if not all([type, description, severity]):
        raise ValueError("Missing required fields in issue block")

    if type is None:
        raise ValueError("Type field is missing or malformed")
    if description is None:
        raise ValueError("Description field is missing or malformed")
    if severity is None:
        raise ValueError("Severity field is missing or malformed")

    return {
        'type': type.group(1).strip(),
        'description': description.group(1).strip(),
        'severity': severity.group(1).strip()
    }

def parse_notebook_issues_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into individual issue blocks
    blocks = re.findall(r'<issue>(.*?)</issue>', content, re.DOTALL)

    issues = []
    for block in blocks:
        try:
            issue_dict = parse_issue_block(block)
            issues.append(issue_dict)
        except Exception as e:
            print(f"Error parsing block in {file_path}: {str(e)}")
            continue

    return issues

def get_issues_path(notebook_config: Dict[str, Any]) -> str:
    """Construct issues file path from notebook configuration."""
    dandiset_id = notebook_config['dandiset_id']
    version = notebook_config['dandiset_version']
    chat_id = notebook_config.get('chat_id')
    model_name = notebook_config['model'].split('/')[-1]  # Extract part after '/'
    prompt = notebook_config['prompt']
    is_skip_explore = notebook_config.get('skip_explore', False)

    if is_skip_explore:
        chat_id_part = 'skip-explore'
    else:
        chat_id_part = chat_id[:8] if chat_id else 'unknown'

    return str(Path('notebook_issues') / 'dandisets' / dandiset_id / version /
              chat_id_part / model_name / prompt / 'notebook_issues.txt')

def load_notebooks_config() -> List[Dict[str, Any]]:
    """Load notebook configurations from notebooks.yaml."""
    try:
        with open('notebooks.yaml', 'r') as f:
            config = yaml.safe_load(f)
            if not config or 'notebooks' not in config:
                raise ValueError("No notebooks found in configuration")
            return config['notebooks']
    except (yaml.YAMLError, FileNotFoundError) as e:
        print(f"Error loading notebooks.yaml: {e}")
        return []

def main():
    # Load notebook configurations
    notebooks = load_notebooks_config()
    if not notebooks:
        print("No notebooks found in configuration")
        return

    # Parse each notebook's issues file
    results = []
    unique_paths = set()  # Track unique paths to avoid duplicates

    for notebook in notebooks:
        issues_path = get_issues_path(notebook)
        if issues_path in unique_paths:
            continue
        unique_paths.add(issues_path)

        try:
            if not os.path.exists(issues_path):
                print(f"Issues file not found: {issues_path}")
                continue

            print(f"Processing {issues_path}...")
            issues = parse_notebook_issues_file(issues_path)

            result = {
                'dandiset_id': notebook['dandiset_id'],
                'version': notebook['dandiset_version'],
                'chat_id': notebook.get('chat_id', 'skip-explore'),
                'model': notebook['model'],
                'prompt': notebook['prompt'],
                'issues': issues
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing {issues_path}: {str(e)}")
            continue

    # Write results to JSON file
    output = {
        'results': results
    }

    with open('notebook_issues.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"\nProcessed {len(results)} files successfully")
    print("Results saved to notebook_issues.json")

if __name__ == '__main__':
    main()
