from typing import List, Dict, Any
import os
import json
import argparse
from helpers.run_completion import run_completion


def create_message_content_for_cell(cell: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create user message content for a given cell."""
    content: List[Dict[str, Any]] = []
    if cell["cell_type"] == "markdown":
        markdown_source = cell["source"]
        content.append(
            {"type": "text", "text": "INPUT-MARKDOWN: " + "".join(markdown_source)}
        )
    elif cell["cell_type"] == "code":
        code_source = cell["source"]
        content.append({"type": "text", "text": "INPUT-CODE: " + "".join(code_source)})
        for x in cell["outputs"]:
            output_type = x["output_type"]
            if output_type == "stream":
                text = "OUTPUT-TEXT: " + "\n".join(x["text"])
                if len(text) > 20_000:
                    text = text[:20_000] + " [OUTPUT-TRUNCATED]"
                content.append(
                    {"type": "text", "text": text}
                )
            elif output_type == "display_data" or output_type == "execute_result":
                if "image/png" in x["data"]:
                    png_base64 = x["data"]["image/png"]
                    image_data_url = f"data:image/png;base64,{png_base64}"
                    content.append({"type": "text", "text": "OUTPUT-IMAGE:"})
                    content.append(
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    )
                elif "text/plain" in x["data"]:
                    content.append(
                        {
                            "type": "text",
                            "text": "OUTPUT-TEXT: " + "".join(x["data"]["text/plain"]),
                        }
                    )
                elif "text/html" in x["data"]:
                    content.append(
                        {
                            "type": "text",
                            "text": "OUTPUT-TEXT: " + "".join(x["data"]["text/html"]),
                        }
                    )
                else:
                    print(
                        f"Warning: got output type {output_type} but no image/png data or text/plain or text/html"
                    )
            else:
                print(f"Warning: unsupported output type {output_type}")
    else:
        print(f'Warning: unsupported cell type {cell["cell_type"]}')
        content.append({"type": "text", "text": "Unsupported cell type"})
    return content

def main():
    parser = argparse.ArgumentParser(description='Compare two Jupyter notebooks')
    parser.add_argument('--notebook1-path', type=str, required=True, help='Path to the first notebook')
    parser.add_argument('--notebook2-path', type=str, required=True, help='Path to the second notebook')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the comparison result')
    parser.add_argument('--model', type=str, default='openai/gpt-4.1-mini', help='Model to use for comparison')
    args = parser.parse_args()

    notebook1_path = args.notebook1_path
    notebook2_path = args.notebook2_path
    output_path = args.output_path
    model = args.model

    with open(notebook1_path, 'r', encoding='utf-8') as f:
        notebook1 = json.load(f)
    with open(notebook2_path, 'r', encoding='utf-8') as f:
        notebook2 = json.load(f)
    cells1 = notebook1.get('cells', [])
    cells2 = notebook2.get('cells', [])

    this_dir = os.path.dirname(os.path.abspath(__file__))
    system_message_path = os.path.join(this_dir, "templates/comparison_system_message.txt")
    with open(system_message_path, "r", encoding="utf-8") as f:
        system_message = f.read()
    user_message_path = os.path.join(this_dir, "templates/comparison_user_message.txt")
    with open(user_message_path, "r", encoding="utf-8") as f:
        user_message = f.read()

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": system_message,
        }
    ]
    for notebook_num in range(1, 2 + 1):
        messages.append({
            "role": "user",
            "content": f"BEGIN NOTEBOOK {notebook_num} CONTENT",
        })
        cells = cells1 if notebook_num == 1 else cells2
        for cell in cells:
            if cell["cell_type"] not in ["code", "markdown"]:
                continue
            cell_content = create_message_content_for_cell(cell)
            messages.append(
                {
                    "role": "user",
                    "content": cell_content,
                }
            )
        messages.append({
            "role": "user",
            "content": f"END NOTEBOOK {notebook_num} CONTENT",
        })

    messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    comparison_response, _, prompt_tokens, completion_tokens = run_completion(messages=messages, model=model)

    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(comparison_response)


if __name__ == "__main__":
    main()
