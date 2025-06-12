#!/usr/bin/env python

import yaml
import json
import os

def get_user_messages(chat_json):
    """Extract user messages from chat JSON."""
    messages = []
    for msg in chat_json['messages']:
        if msg.get('role') == 'user':
            # Handle both string and list content
            content = msg.get('content', '')
            if isinstance(content, list):
                # For list content, concatenate text items
                message_text = '\n'.join(
                    item.get('text', '')
                    for item in content
                    if item.get('type') == 'text'
                )
            else:
                message_text = content
            messages.append(message_text)
    return messages

def main():
    # Load configuration from notebooks.yaml
    with open('notebooks.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if 'notebooks' not in config:
        print("No 'notebooks' section found in YAML file")
        return

    # Process each notebook configuration
    for notebook in config['notebooks']:
        # Skip entries with skip_explore=True
        if notebook.get('skip_explore', False):
            continue

        dandiset_id = notebook['dandiset_id']
        version = notebook['dandiset_version']
        chat_id = notebook['chat_id']
        model = notebook['model'].split('/')[-1]  # Extract model name after '/'

        # Construct path to local chat.json
        chat_id_8 = chat_id[:8]
        chat_path = f"notebooks/dandisets/{dandiset_id}/{version}/{chat_id_8}/{model}/{notebook['prompt']}/chat.json"

        try:
            if not os.path.exists(chat_path):
                print("")
                print(f"**Chat file not found at {chat_path} - skipping**")
                print("")
                continue

            with open(chat_path, 'r') as f:
                chat_json = json.load(f)

            if not chat_json.get('finalized', False):
                print(f"Warning: Chat for dandiset {dandiset_id} is not finalized")
                continue

            user_messages = get_user_messages(chat_json)

            print(f"**Dandiset {dandiset_id}**")
            for i, msg in enumerate(user_messages, 1):
                if i == 1:
                    continue
                if msg and msg.lower() != 'proceed':
                    print(f"- Message {i}: {msg}")
            print("")

        except Exception as e:
            print(f"Error processing dandiset {dandiset_id}: {str(e)}")

if __name__ == '__main__':
    main()
