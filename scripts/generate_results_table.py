#!/usr/bin/env python

import json

def generate_markdown_table():
    # Read the JSON file
    with open('results.json', 'r') as f:
        data = json.load(f)

    # Start the markdown table
    markdown = "| Dandiset ID | Chat Model | Notebook Model | Chat Cost | Notebook Cost | Chat |\n"
    markdown += "|------------|------------|----------------|-----------|---------------|------|\n"

    # Add each row
    for notebook in data['notebooks']:
        skip_explore = notebook.get('skip_explore', False)
        dandiset_id = notebook['dandisetId']
        if not skip_explore:
            chat_model = notebook['chat']['chatModel'].split('/')[-1]
            chat_cost = f"${notebook['chat']['estimatedCost']:.2f}"
            chat_url = notebook['chat']['chatUrl']
        else:
            chat_model = 'skip-explore'
            chat_cost = 'N/A'
            chat_url = None
        notebook_model = notebook['notebook']['notebookModel'].split('/')[-1]
        notebook_cost = f"${notebook['notebook']['estimatedCost']:.2f}"

        notebook_url = notebook['notebook']['notebookUrl']

        # Format the row with the dandiset ID linking to the notebook URL
        row = f"| [{dandiset_id}]({notebook_url}) | {chat_model} | {notebook_model} | {chat_cost} | {notebook_cost} | {'[chat](' + chat_url + ')' if chat_url else 'N/A'} |\n"
        markdown += row

    # Write to results.md
    with open('results.md', 'w') as f:
        f.write(markdown)

if __name__ == '__main__':
    generate_markdown_table()
