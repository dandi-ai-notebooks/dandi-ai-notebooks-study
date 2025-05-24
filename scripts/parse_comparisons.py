#!/usr/bin/env python

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any

def parse_comparison_block(block: str) -> Dict[str, Any]:
    """Parse a single comparison block into a dictionary."""
    # Extract each field using regex
    question = re.search(r'<question>(.*?)</question>', block)
    question_short = re.search(r'<question_shortened>(.*?)</question_shortened>', block)
    rationale = re.search(r'<rationale>(.*?)</rationale>', block, re.DOTALL)
    preference = re.search(r'<preference>(.*?)</preference>', block)

    # Check matches and extract values safely
    try:
        if not all([question, question_short, rationale, preference]):
            raise ValueError("Missing required fields in comparison block")

        question_num = int(question.group(1)) if question else None
        question_text = question_short.group(1).strip() if question_short else None
        rationale_text = rationale.group(1).strip() if rationale else None
        pref_value = int(preference.group(1)) if preference else None

        if any(x is None for x in [question_num, question_text, rationale_text, pref_value]):
            raise ValueError("Could not extract all required fields")

        return {
            'question_number': question_num,
            'question_shortened': question_text,
            'rationale': rationale_text,
            'preference': pref_value
        }
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Error parsing comparison block: {str(e)}")

def parse_comparison_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse a comparison file into a list of comparison dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into individual comparison blocks
    blocks = re.findall(r'<comparison>(.*?)</comparison>', content, re.DOTALL)

    comparisons = []
    for block in blocks:
        try:
            comparison = parse_comparison_block(block)
            comparisons.append(comparison)
        except Exception as e:
            print(f"Error parsing block in {file_path}: {str(e)}")
            continue

    return comparisons

def extract_info_from_path(file_path: str) -> Dict[str, str]:
    """Extract metadata from the comparison file path."""
    # Expected path format: notebook_comparisons/dandisets/{dandiset_id}/{version}/{chat_id}/{model}/{prompt}/comparison_with_skip_explore.txt
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
    # Find all comparison files
    comparison_files = []
    root_dir = Path('notebook_comparisons')
    if not root_dir.exists():
        print("No notebook_comparisons directory found")
        return

    for file_path in root_dir.rglob('comparison_with_skip_explore.txt'):
        comparison_files.append(str(file_path))

    print(f"Found {len(comparison_files)} comparison files")

    # Parse each file and collect results
    results = []
    for file_path in comparison_files:
        try:
            print(f"Processing {file_path}...")
            metadata = extract_info_from_path(file_path)
            comparisons = parse_comparison_file(file_path)

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

    with open('comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"\nProcessed {len(results)} files successfully")
    print("Results saved to comparison_results.json")

if __name__ == '__main__':
    main()
