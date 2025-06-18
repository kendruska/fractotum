import argparse
from pathlib import Path
from text_processing import get_embedding_and_params
from procedural_generation import render_image, render_video
from music import render_music
from functools import partial

def wrap_text_soft(text: str, width: int = 60) -> str:
    """
    Wrap text to a specified width without breaking words.

    Args:
        text (str): The input text to wrap.
        width (int): The maximum line width.
    Returns:
        str: The wrapped text with newlines.
    """
    if not text:
        return ""
    words = text.split()
    lines = []
    current_line = words[0] if words else ""
    for word in words[1:]:
        if len(current_line) + 1 + len(word) <= width:
            current_line += f" {word}"
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)

def read_input_file(file_path: str) -> str:
    """
    Read and preprocess an input text file, returning its content as a single string.

    Args:
        file_path (str): Path to the input text file.
    Returns:
        str: The file content as a single line of text.
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as UTF-8.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().replace("\n", " ").strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file '{file_path}' not found")
    except UnicodeDecodeError:
        raise ValueError(f"Cannot decode file '{file_path}' as UTF-8")

def main():
    """
    Main entry point for the fractotum generator CLI.
    Parses arguments, reads input, generates embedding and parameters, and dispatches rendering.
    """
    parser = argparse.ArgumentParser(description="Generate fractals from text")
    parser.add_argument('--input', required=True, help='Text file (.txt)')
    parser.add_argument('--mode', required=True, 
                       choices=['image', 'video', 'audio'], 
                       help='Output mode')
    args = parser.parse_args()

    text = read_input_file(args.input)
    formatted_text = wrap_text_soft(text)

    print("\n📜 Input text preview:\n")
    print(formatted_text)
    print("\n---\n")

    embed, params = get_embedding_and_params(text)
    base_name = Path(args.input).stem

    # Use partial to avoid lambda and improve readability
    render_functions = {
        'image': partial(render_image, embed, params, f'{base_name}_fractotum.png'),
        'video': partial(render_video, embed, params, f'{base_name}_fractotum.mp4'),
        'audio': partial(render_music, embed, f'{base_name}_fractotum.mid')
    }
    
    render_func = render_functions.get(args.mode)
    if render_func:
        render_func()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()