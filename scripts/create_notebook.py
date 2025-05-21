#!/usr/bin/env python

import argparse
import os
import json
import subprocess
import re
import requests
from helpers.run_completion import run_completion


def download_chat(url: str) -> str:
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


def replace_template_vars(text: str, dandiset_id: str, version: str) -> str:
    """Replace template variables in text."""
    print("Replacing template variables...")
    text = text.replace("{{ DANDISET_ID }}", dandiset_id)
    text = text.replace("{{ DANDISET_VERSION }}", version)
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
        subprocess.run(['jupyter', 'execute', '--inplace', ipynb_path], check=True, capture_output=True, text=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, f"Notebook execution failed: {e.stderr}"


def main():
    parser = argparse.ArgumentParser(description='Create and execute a Jupyter notebook with AI assistance')
    parser.add_argument('--dandiset', required=True, help='Dandiset ID')
    parser.add_argument('--version', required=True, help='Dandiset version')
    parser.add_argument('--model', required=True, help='LLM model to use')
    parser.add_argument('--chat', required=True, help='URL of chat JSON')
    parser.add_argument('--prompt', required=True, help='Path to prompt file')
    parser.add_argument('--output', required=True, help='Path of output directory')

    args = parser.parse_args()

    # if output directory exists, skip
    if os.path.exists(args.output):
        print(f"Output directory {args.output} already exists. Skipping notebook creation.")
        return

    messages = []
    messages.append({
        "role": "system",
        "content": "You are going to be given a chat, and then the user will prompt you to create a Jupyter notebook based on this chat. They will provide detailed instructions."
    })

    # We can't just put the chat in as json because it contains encoded images.
    # So we put the chat in one message at a time (which might be confusing, but oh well)
    chat = download_chat(args.chat)
    chat_messages = chat['messages']  # type: ignore
    for message in chat_messages:
        messages.append(message)

    # chat_messages_metadata = chat['messageMetadata']  # type: ignore

    # Read and process prompt
    prompt = read_prompt(args.prompt)
    prompt = replace_template_vars(text=prompt, dandiset_id=args.dandiset, version=args.version)
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
        if attempt > 5:
            raise RuntimeError("Failed to create and execute notebook after 5 attempts.")

    # write chat.json
    with open(f'{args.output}/chat.json', 'w') as f:
        json.dump(chat, f, indent=2)

    # write prompt.txt
    with open(f'{args.output}/prompt.txt', 'w') as f:
        f.write(prompt)

    print(f'Prompt tokens used: {total_prompt_tokens}')
    print(f'Completion tokens used: {total_completion_tokens}')
    model_cost = get_model_cost(args.model)
    if model_cost[0] is not None and model_cost[1] is not None:
        print(f'Estimated cost: ${model_cost[0] * total_prompt_tokens / 1e6 + model_cost[1] * total_completion_tokens / 1e6:.3f}')

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
