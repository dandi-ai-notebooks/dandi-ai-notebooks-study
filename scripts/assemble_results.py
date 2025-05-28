#!/usr/bin/env python

import yaml
import json
import os
from pathlib import Path

def load_chat_data(chat_file_path: str) -> dict:
    """Load data from chat.json file."""
    try:
        with open(chat_file_path, 'r') as f:
            data = json.load(f)
            return {
                'timestampCreated': data.get('timestampCreated'),
                'timestampUpdated': data.get('timestampUpdated'),
                'promptTokens': data.get('promptTokens'),
                'completionTokens': data.get('completionTokens'),
                'estimatedCost': data.get('estimatedCost')
            }
    except FileNotFoundError:
        print(f"Warning: Chat file not found: {chat_file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in chat file: {chat_file_path}")
        return {}

def load_info_data(info_file_path: str) -> dict:
    """Load data from info.json file."""
    try:
        with open(info_file_path, 'r') as f:
            data = json.load(f)
            return {
                'promptTokens': data.get('prompt_tokens'),
                'completionTokens': data.get('completion_tokens'),
                'estimatedCost': data.get('estimated_cost')
            }
    except FileNotFoundError:
        print(f"Warning: Info file not found: {info_file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in info file: {info_file_path}")
        return {}

def construct_chat_url(dandiset_id: str, version: str, chat_id: str) -> str:
    return f"https://dandi-ai-notebooks.github.io/dandiset-explorer/chat?dandisetId={dandiset_id}&dandisetVersion={version}&chatId={chat_id}"

def get_file_paths(dandiset_id: str, version: str, chat_id: str, model: str, prompt: str, skip_explore: bool) -> tuple:
    """Construct paths to chat.json and info.json files."""
    chat_id_8_or_skip_explore = chat_id[:8] if not skip_explore else 'skip-explore'
    model_name = model.split('/')[-1]
    base_path = Path(f"notebooks/dandisets/{dandiset_id}/{version}/{chat_id_8_or_skip_explore}/{model_name}/{prompt}")

    return (
        base_path / "chat.json" if not skip_explore else None,
        base_path / "info.json"
    )

def main():
    # Load configuration from notebooks.yaml
    try:
        with open('notebooks.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except (yaml.YAMLError, FileNotFoundError) as e:
        print(f"Error loading notebooks.yaml: {e}")
        return

    if 'notebooks' not in config:
        print("No 'notebooks' section found in YAML file")
        return

    results = {'notebooks': []}

    # Process each notebook
    for notebook in config['notebooks']:
        dandiset_id = notebook['dandiset_id']
        version = notebook['dandiset_version']
        chat_id = notebook.get('chat_id', None)
        model = notebook['model']
        prompt = notebook['prompt']
        skip_explore = notebook.get('skip_explore', False)

        # Get file paths
        chat_file_path, info_file_path = get_file_paths(
            dandiset_id, version, chat_id, model, prompt, skip_explore
        )

        # Load data from files
        if chat_file_path:
            # if not os.path.exists(chat_file_path):
            #     print('chat.json file not found. one off solution is to download it')
            #     with open(info_file_path, 'r') as f:
            #         info_data = json.load(f)
            #     def _download_file(url, file_path):
            #         import requests
            #         response = requests.get(url)
            #         with open(file_path, 'wb') as f:
            #             f.write(response.content)
            #     _download_file(info_data['chat_url'], chat_file_path)
            chat_data = load_chat_data(str(chat_file_path))
        else:
            chat_data = None
        info_data = load_info_data(str(info_file_path))

        # Calculate derived values
        chat_id_8_or_skip_explore = chat_id[:8] if not skip_explore else 'skip-explore'
        model_name = model.split('/')[-1]

        # Construct notebook entry
        notebook_entry = {
            'dandisetId': dandiset_id,
            'dandisetVersion': version,
            'notebook': {
                'notebookModel': model,
                'notebookPrompt': prompt,
                'notebookUrl': f"https://github.com/dandi-ai-notebooks/dandi-ai-notebooks-5/blob/main/notebooks/dandisets/{dandiset_id}/{version}/{chat_id_8_or_skip_explore}/{model_name}/{prompt}/notebook.ipynb",
                'promptTokens': info_data.get('promptTokens'),
                'completionTokens': info_data.get('completionTokens'),
                'estimatedCost': info_data.get('estimatedCost')
            },
            'skip_explore': skip_explore
        }
        if chat_id:
            notebook_entry['chatId'] = chat_id
            assert chat_data, f"No chat data found for {chat_id}"
            notebook_entry['chat'] = {
                'chatUrl': construct_chat_url(dandiset_id, version, chat_id),
                'chatModel': model,
                **chat_data
            }

        results['notebooks'].append(notebook_entry)

    # Write results to file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Successfully processed {len(results['notebooks'])} notebooks")

if __name__ == '__main__':
    main()
