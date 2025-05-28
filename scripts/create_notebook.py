#!/usr/bin/env python

import argparse
import os
import json
import subprocess
import re
import requests
from helpers.run_completion import run_completion


def download_chat(url: str):
    """Download chat JSON from URL and return system message."""
    print(f"Downloading chat JSON from {url}...")
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download chat JSON: {response.text}")
    chat_json = response.json()
    return chat_json


def read_prompt(path: str) -> str:
    """Read prompt from file."""
    print(f"Reading prompt from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, 'r') as f:
        return f.read()


def replace_template_vars(text: str, dandiset_id: str, version: str, chat_or_information_str: str) -> str:
    """Replace template variables in text."""
    print("Replacing template variables...")
    text = text.replace("{{ DANDISET_ID }}", dandiset_id)
    text = text.replace("{{ DANDISET_VERSION }}", version)
    text = text.replace("{{ CHAT_OR_INFORMATION }}", chat_or_information_str)
    # check to see if there are are {{ or }} left in the text that we didn't replace
    if '{{' in text or '}}' in text:
        raise ValueError("Template variables not replaced correctly. Please check the prompt.")
    return text


def write_notebook(content: str, path: str) -> str:
    """Write notebook content to .py file."""
    py_path = f'{path}/notebook.py'
    print(f"Writing notebook content to {py_path}...")
    # Create parent directory if it doesn't exist
    parent_dir = os.path.dirname(py_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(py_path, 'w') as f:
        f.write(content)
    return py_path


def execute_notebook(py_path: str) -> tuple[bool, str]:
    """Convert .py to .ipynb and execute. Returns (success, error_message)."""
    ipynb_path = py_path.replace('.py', '.ipynb')
    print(f"\nConverting {py_path} to notebook...")

    try:
        subprocess.run(['jupytext', '--to', 'notebook', py_path], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return False, f"Jupytext conversion failed: {e.stderr}"

    print(f"Executing notebook {ipynb_path}...")
    try:
        subprocess.run(['jupyter', 'execute', '--inplace', ipynb_path], check=True, capture_output=True, text=True, timeout=600)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, f"Notebook execution failed: {e.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Notebook execution timed out after 600 seconds"


def main():
    parser = argparse.ArgumentParser(description='Create and execute a Jupyter notebook with AI assistance')
    parser.add_argument('--dandiset', required=True, help='Dandiset ID')
    parser.add_argument('--version', required=True, help='Dandiset version')
    parser.add_argument('--model', required=True, help='LLM model to use')
    parser.add_argument('--chat', required=False, help='URL of chat JSON')
    parser.add_argument('--prompt', required=True, help='Path to prompt file')
    parser.add_argument('--output', required=True, help='Path of output directory')
    parser.add_argument('--skip-explore', action='store_true', help='Use skip-explore mode')

    args = parser.parse_args()

    # if output directory exists, skip
    if os.path.exists(args.output):
        print(f"Output directory {args.output} already exists. Skipping notebook creation.")
        return

    if args.skip_explore:
        chat_or_information_str = "information"
    else:
        chat_or_information_str = "chat"

    messages = []
    if not args.skip_explore:
        if not args.chat:
            raise ValueError("Chat URL is required in when not in skip-explore mode.")
        messages.append({
            "role": "system",
            "content": "You are going to be given a chat, and then the user will prompt you to create a Jupyter notebook based on this chat. They will provide detailed instructions."
        })

        # We can't just put the chat in as json because it contains encoded images.
        # So we put the chat in one message at a time
        chat = download_chat(args.chat)
        if not chat.get('finalized', False):
            raise Exception("Chat is not finalized. Please finalize the chat before using it.")
        chat_messages = chat['messages']
        for message in chat_messages:
            messages.append(message)
    else:
        # skip-explore mode
        if args.chat:
            raise ValueError("Chat URL is not allowed in skip-explore mode.")
        chat = None
        dandiset_metadata = _get_dandiset_metadata(args.dandiset, args.version)
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"The following is metadata about Dandiset {args.dandiset} version {args.version}."
                },
                {
                    "type": "text",
                    "text": json.dumps(dandiset_metadata, indent=2)
                }
            ]
        })
        dandiset_nwb_file_paths = _get_dandiset_nwb_file_paths(args.dandiset, args.version, limit=10)
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "The following is a listing of some of the files in the Dandiset."
                },
                {
                    "type": "text",
                    "text": '\n'.join(dandiset_nwb_file_paths)
                }
            ]
        })
        path1 = dandiset_nwb_file_paths[0]
        print("Getting usage script for file:", path1)
        nwb_file_usage_script = _get_nwb_file_usage_script(args.dandiset, args.version, path1)
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"The following is metadata and usage information about the file {path1}."
                },
                {
                    "type": "text",
                    "text": nwb_file_usage_script
                }
            ]
        })

    # Read and process prompt
    prompt = read_prompt(args.prompt)
    prompt = replace_template_vars(text=prompt, dandiset_id=args.dandiset, version=args.version, chat_or_information_str=chat_or_information_str)
    # chat_messages_metadata = chat['messageMetadata']  # type: ignore
    messages.append({"role": "user", "content": prompt})

    total_prompt_tokens = 0
    total_completion_tokens = 0

    attempt = 1
    while True:
        print(f"\nAttempt {attempt} to generate and execute notebook...")

        # Generate notebook content
        print("\nGenerating notebook content...")
        response, messages, prompt_tokens, completion_tokens = run_completion(
            messages=messages,
            model=args.model
        )
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        print(f'Prompt tokens used so far: {total_prompt_tokens}')
        print(f'Completion tokens used so far: {total_completion_tokens}')

        # Extract notebook content from between tags, assuming tags are on their own lines
        match = re.search(r'<notebook>\n(.*?)\n</notebook>', response, re.DOTALL)
        if not match:
            error = "Response must include <notebook> tags on their own lines"
            print(f"\nError: {error}")
            print("Adding error message and retrying...")
            messages.append({
                "role": "system",
                "content": f"Error: {error}. Please provide the notebook content between <notebook> tags."
            })
            attempt += 1
            continue

        content = match.group(1)

        # Write notebook and attempt execution
        py_path = write_notebook(content, args.output)
        success, error = execute_notebook(py_path)

        if success:
            print("\nSuccess! Notebook was created and executed without errors.")
            break

        print(f"\nError: {error}")
        print("Adding error message and retrying...")
        messages.append({
            "role": "system",
            "content": f"The notebook execution failed with the following error:\n\n{error}\n\nPlease provide a corrected version of the notebook."
        })

        attempt += 1
        if attempt > 8:
            raise RuntimeError("Failed to create and execute notebook after 8 attempts.")

    # write prompt.txt
    with open(f'{args.output}/prompt.txt', 'w') as f:
        f.write(prompt)

    # write messages.json
    with open(f'{args.output}/messages.json', 'w') as f:
        json.dump(messages, f, indent=2)

    # write chat.json
    if chat is not None:
        with open(f'{args.output}/chat.json', 'w') as f:
            json.dump(chat, f, indent=2)

    print(f'Prompt tokens used: {total_prompt_tokens}')
    print(f'Completion tokens used: {total_completion_tokens}')
    model_cost = get_model_cost(args.model)
    if model_cost[0] is not None and model_cost[1] is not None:
        estimated_cost = model_cost[0] * total_prompt_tokens / 1e6 + model_cost[1] * total_completion_tokens / 1e6
    else:
        estimated_cost = None
    if estimated_cost is not None:
        print(f'Estimated cost: ${estimated_cost:.3f}')

    with open(f'{args.output}/info.json', 'w') as f:
        json.dump({
            'model': args.model,
            'dandiset_id': args.dandiset,
            'version': args.version,
            'prompt': args.prompt,
            'skip_explore': args.skip_explore,
            'chat_url': args.chat if args.chat else '',
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens,
            'estimated_cost': estimated_cost
        }, f, indent=2)
    print(f"Notebook created and executed successfully. Output saved to {args.output}")

def _get_dandiset_metadata(dandiset_id: str, version: str) -> dict:
    """Get Dandiset metadata from Dandi API."""
    print(f"Getting metadata for Dandiset {dandiset_id} version {version}...")
    response = requests.get(f"https://api.dandiarchive.org/api/dandisets/{dandiset_id}/versions/{version}/")
    if response.status_code != 200:
        raise RuntimeError(f"Failed to get Dandiset metadata: {response.text}")
    metadata = response.json()
    # Remove any non-serializable fields
    return metadata

def _get_dandiset_nwb_file_paths(dandiset_id: str, version: str, limit: int = 10) -> list[str]:
    """Get Dandiset NWB file paths from Dandi API."""
    from dandi.dandiapi import DandiAPIClient
    from itertools import islice

    client = DandiAPIClient()
    dandiset = client.get_dandiset(dandiset_id, version)

    # List some assets in the Dandiset
    assets = dandiset.get_assets_by_glob("*.nwb")
    return [
        asset.path for asset in islice(assets, limit)
    ]

def _get_nwb_file_usage_script(dandiset_id: str, version: str, path: str) -> str:
    from get_nwbfile_info import get_nwbfile_usage_script
    from dandi.dandiapi import DandiAPIClient

    client = DandiAPIClient()
    dandiset = client.get_dandiset(dandiset_id, version)

    download_url = next(dandiset.get_assets_by_glob(path)).download_url

    usage_script = get_nwbfile_usage_script(download_url)

    lines = usage_script.split("\n")
    new_lines = []
    for i, line in enumerate(lines):
        if line.startswith("url = "):
            # remove the line url = "..." and replace it with code to get the url based on the path
            # That way, the AI will not try to use the hard-coded url

            # Look familiar?
            txt0 = f"""from dandi.dandiapi import DandiAPIClient

client = DandiAPIClient()
dandiset = client.get_dandiset("{dandiset_id}", "{version}")
url = next(dandiset.get_assets_by_glob("{path}")).download_url
"""
            for x in txt0.split('\n'):
                new_lines.append(x)
        elif line.startswith("# ") and "https://api.dandiarchive.org/" in line:
            lines[i] = "" # Hide where we display the asset URL so that the AI doesn't use it directly
        else:
            new_lines.append(line)
    usage_script = "\n".join(new_lines)
    return usage_script

def get_model_cost(model: str):
    if model == 'google/gemini-2.0-flash-001':
        return [0.1, 0.4]
    elif model == 'google/gemini-2.5-flash-preview':
        return [0.15, 0.6]
    elif model == 'google/gemini-2.5-pro-preview-03-25':
        return [1.25, 10]
    elif model == 'google/gemini-2.5-pro-preview':
        return [1.25, 10]
    elif model == 'openai/gpt-4o':
        return [2.5, 10]
    elif model == 'anthropic/claude-3.5-sonnet':
        return [3, 15]
    elif model == 'anthropic/claude-3.7-sonnet':
        return [3, 15]
    elif model == 'anthropic/claude-3.7-sonnet:thinking':
        return [3, 15]
    elif model == 'deepseek/deepseek-r1':
        return [0.55, 2.19]
    elif model == 'deepseek/deepseek-chat-v3-0324':
        return [0.27, 1.1]
    elif model == 'openai/o4-mini':
        return [1.1, 4.4]
    elif model == 'openai/o4-mini-high':
        return [1.1, 4.4]
    elif model == 'openai/gpt-4.1':
        return [2, 8]
    elif model == 'openai/o3':
        return [10, 40]
    return [None, None]


if __name__ == '__main__':
    main()
