#!/usr/bin/env python

import os
import json
import re
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

def extract_info_from_path(file_path: str) -> Dict[str, str]:
    """Extract metadata from the comparison file path."""
    # Expected path format: notebook_issues/dandisets/{dandiset_id}/{version}/{chat_id}/{model}/{prompt}/notebook_issues.txt
    parts = Path(file_path).parts
    if len(parts) < 8:
        raise ValueError(f"Invalid path structure: {file_path}")

    return {
        'dandiset_id': parts[2],
        'version': parts[3],
        'chat_id': parts[4],
        'model': parts[5],
        'prompt': parts[6]
    }

def main():
    # Find all issues files
    notebook_issues_files = []
    root_dir = Path('notebook_issues')
    if not root_dir.exists():
        print("No notebook_issues directory found")
        return

    for file_path in root_dir.rglob('notebook_issues.txt'):
        notebook_issues_files.append(str(file_path))

    print(f"Found {len(notebook_issues_files)} issues files")

    # Parse each file and collect results
    results = []
    for file_path in notebook_issues_files:
        try:
            print(f"Processing {file_path}...")
            metadata = extract_info_from_path(file_path)
            comparisons = parse_notebook_issues_file(file_path)

            result = {
                **metadata,
                'comparisons': comparisons
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
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
