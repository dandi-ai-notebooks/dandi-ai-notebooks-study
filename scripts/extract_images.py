#!/usr/bin/env python3

import yaml
import json
import base64
import os
from pathlib import Path

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_model_second_part(model):
    return model.split('/')[-1]

def process_notebook(notebook_path, output_dir, notebook_info, markdown_entries):
    try:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
    except FileNotFoundError:
        print(f"Notebook not found: {notebook_path}")
        return
    except json.JSONDecodeError:
        print(f"Invalid JSON in notebook: {notebook_path}")
        return

    image_count = 0
    image_paths = []

    # Create images directory for this notebook
    images_dir = ensure_dir(output_dir / 'images')

    # Delete existing images in the directory
    for img_file in images_dir.glob('*.png'):
        try:
            img_file.unlink()
        except Exception as e:
            print(f"Error deleting file {img_file}: {e}")

    for cell_idx, cell in enumerate(nb.get('cells', [])):
        if 'outputs' not in cell:
            continue

        for output in cell['outputs']:
            if 'data' not in output:
                continue

            if 'image/png' not in output['data']:
                continue

            image_count += 1
            image_data = output['data']['image/png']

            # Save the image
            image_path = images_dir / f"{image_count}.png"
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(image_data))

            image_paths.append(image_path)
            print(f"Saved image {image_count} from notebook {notebook_path}")

    if image_paths:
        # Add entry to markdown list
        chat_id = notebook_info['chat_id'][:8] if not notebook_info.get('skip_explore', False) else 'skip-explore'
        entry = f"## Dandiset {notebook_info['dandiset_id']} (chat_id {chat_id})\n\n"
        for idx, path in enumerate(image_paths, 1):
            rel_path = os.path.relpath(path, Path('images'))
            entry += f"![Image {idx}]({rel_path})\n\n"
        markdown_entries.append(entry)

def main():
    # Read the YAML file
    with open('notebooks.yaml', 'r') as f:
        data = yaml.safe_load(f)

    markdown_entries = []

    # Process each notebook
    for notebook in data['notebooks']:
        dandiset_id = notebook['dandiset_id']
        version = notebook['dandiset_version']
        chat_id = notebook['chat_id'][:8] if not notebook.get('skip_explore', False) else 'skip-explore'
        model = get_model_second_part(notebook['model'])
        prompt = notebook['prompt']

        # Construct paths
        notebook_path = Path('notebooks') / 'dandisets' / dandiset_id / version / chat_id / model / prompt / 'notebook.ipynb'
        output_dir = ensure_dir(Path('images') / 'dandisets' / dandiset_id / version / chat_id / model / prompt)

        process_notebook(notebook_path, output_dir, notebook, markdown_entries)

    # Write markdown file
    markdown_content = "# Notebook Images\n\n" + "\n".join(markdown_entries)
    with open('images/images.md', 'w') as f:
        f.write(markdown_content)

if __name__ == '__main__':
    main()
