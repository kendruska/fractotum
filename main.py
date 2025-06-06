import argparse
from text_processing import get_embedding_and_params
from procedural_generation import render_image, render_video
import os

def wrap_text_soft(text, width=60):
    words = text.split()
    lines, current_line = [], ""
    for word in words:
        if len(current_line) + len(word) + bool(current_line) <= width:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Text file (.txt)')
    parser.add_argument('--mode', required=True, choices=['image', 'video'], help='Output mode')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()

    formatted_text = wrap_text_soft(text.replace("\n", " ").strip())

    print("\n📜 Input text preview:\n")
    print(formatted_text)
    print("\n---\n")

    embed, params = get_embedding_and_params(text)
    base_name = os.path.splitext(args.input)[0]

    if args.mode == 'image':
        render_image(embed, params, base_name + '_fractotum.png')
    elif args.mode == 'video':
        render_video(embed, params, base_name + '_fractotum.mp4')

if __name__ == "__main__":
    main()
