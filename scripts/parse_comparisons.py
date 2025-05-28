#!/usr/bin/env python

import os
import json
import re
import yaml
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

def get_comparison_path(notebook_config: Dict[str, Any]) -> str:
    """Construct comparison file path from notebook configuration."""
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

    return str(Path('notebook_comparisons') / 'dandisets' / dandiset_id / version /
              chat_id_part / model_name / prompt / 'comparison_with_skip_explore.txt')

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

    # Parse each notebook's comparison file
    results = []
    unique_paths = set()  # Track unique paths to avoid duplicates

    for notebook in notebooks:
        comparison_path = get_comparison_path(notebook)
        if comparison_path in unique_paths:
            continue
        unique_paths.add(comparison_path)

        try:
            if not os.path.exists(comparison_path):
                print(f"Comparison file not found: {comparison_path}")
                continue

            print(f"Processing {comparison_path}...")
            comparisons = parse_comparison_file(comparison_path)

            result = {
                'dandiset_id': notebook['dandiset_id'],
                'version': notebook['dandiset_version'],
                'chat_id': notebook.get('chat_id', 'skip-explore'),
                'model': notebook['model'],
                'prompt': notebook['prompt'],
                'comparisons': comparisons
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing {comparison_path}: {str(e)}")
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
