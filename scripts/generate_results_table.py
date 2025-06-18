#!/usr/bin/env python

import json

def generate_markdown_table():
    # Read the JSON file
    with open('results.json', 'r') as f:
        data = json.load(f)

    # Start the markdown table
    # markdown = "| Dandiset ID | Chat Model | Notebook Model | Chat Cost | Notebook Cost | Chat | Comparison |\n"
    # markdown += "|------------|------------|----------------|-----------|---------------|------|-----------|\n"
    markdown = "| Dandiset ID | Chat Model | Notebook Model | Chat Cost | Notebook Cost | Chat |\n"
    markdown += "|-------------|------------|----------------|-----------|---------------|------|\n"

    markdown_to_review = "| Dandiset | Review |\n"
    markdown_to_review += "|-------------|--------|\n"

    # Add each row
    for notebook in data['notebooks']:
        skip_explore = notebook.get('skip_explore', False)
        dandiset_id = notebook['dandisetId']
        # dandiset_version = notebook['dandisetVersion']
        # notebook_model_second_part = notebook['notebook']['notebookModel'].split('/')[-1]
        # prompt_name = notebook['notebook']['notebookPrompt']
        if not skip_explore:
            chat_model = notebook['chat']['chatModel'].split('/')[-1]
            chat_cost = f"${notebook['chat']['estimatedCost']:.2f}"
            chat_url = notebook['chat']['chatUrl']
            # chat_id_first_8 = notebook['chatId'][:8]
            # https://github.com/dandi-ai-notebooks/dandi-ai-notebooks-5/blob/main/notebook_comparisons/dandisets/001349/0.250520.1729/4befc0a1/gpt-4.1/h-2/comparison_with_skip_explore.txt
            # comparison_url = f'https://github.com/dandi-ai-notebooks/dandi-ai-notebooks-5/blob/main/notebook_comparisons/dandisets/{dandiset_id}/{dandiset_version}/{chat_id_first_8}/{notebook_model_second_part}/{prompt_name}/comparison_with_skip_explore.txt'
        else:
            chat_model = 'skip-explore'
            chat_cost = 'N/A'
            chat_url = None
            # comparison_url = None
        notebook_model = notebook['notebook']['notebookModel'].split('/')[-1]
        estimated_cost = notebook['notebook']['estimatedCost']
        notebook_cost = f"${estimated_cost:.2f}" if estimated_cost is not None else 'N/A'

        notebook_url = notebook['notebook']['notebookUrl']

        # Format the row with the dandiset ID linking to the notebook URL
        # row = f"| [{dandiset_id}]({notebook_url}) | {chat_model} | {notebook_model} | {chat_cost} | {notebook_cost} | {'[chat](' + chat_url + ')' if chat_url else 'N/A'} | {'[comparison](' + comparison_url + ')' if comparison_url else 'N/A'} |\n"
        row = f"| [{dandiset_id}]({notebook_url}) | {chat_model} | {notebook_model} | {chat_cost} | {notebook_cost} | {'[chat](' + chat_url + ')' if chat_url else 'N/A'} |\n"
        markdown += row

        to_review_url = f'https://dandi-ai-notebooks.github.io/dandi-notebook-review/review?url={notebook_url}'
        row_to_review = f"| {dandiset_id} | [notebook review]({to_review_url}) |\n"
        markdown_to_review += row_to_review

    # Write to results.md
    with open('results.md', 'w') as f:
        f.write(markdown)

    # Write to to_review.md
    with open('to_review.md', 'w') as f:
        f.write(markdown_to_review)

if __name__ == '__main__':
    generate_markdown_table()
