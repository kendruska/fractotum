import argparse
import os
import textwrap
from text_processing import get_embedding_and_params
from procedural_generation import render_image, render_video

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
    parser.add_argument('--input', required=True, help='Path to input .txt file containing the text fragment')
    parser.add_argument('--mode', required=True, choices=['image', 'video'], help='Output mode: image or video')
    args = parser.parse_args()

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {args.input}")
        return

    formatted_text = wrap_text_soft(raw_text.replace("\n", " ").strip())

    print("\n📜 Input text preview:\n")
    print(formatted_text)
    print("\n---\n")

    embed, params = get_embedding_and_params(formatted_text)
    base_name = os.path.splitext(args.input)[0]

    output_path = f"{base_name}_fractotum.{'png' if args.mode == 'image' else 'mp4'}"
    if args.mode == 'image':
        render_image(embed, params, output_path)
    else:
        render_video(embed, params, output_path)

if __name__ == "__main__":
    main()
    print("✅ Processing complete!")