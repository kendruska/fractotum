from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt

# Initialize model and colormap list only once
_model = SentenceTransformer("all-mpnet-base-v2")
_cmap_list = [c for c in plt.colormaps() if not c.endswith('_r')]
_cmap_count = len(_cmap_list)

def get_embedding_and_params(text):
    """
    Generate a sentence embedding from the input text and derive fractal parameters from it.

    Args:
        text (str): Input text to encode.
    Returns:
        tuple: (embedding, params) where embedding is a numpy array and params is a dict with fractal parameters.
            - embedding (np.ndarray): The sentence embedding vector.
            - params (dict): Dictionary with keys:
                - 'c' (complex): Complex parameter for fractal generation.
                - 'zoom' (float): Zoom factor for fractal.
                - 'max_iter' (int): Maximum number of iterations for fractal.
                - 'particle_density' (float): Particle density for fractal effect.
                - 'cmap' (str): Colormap name for visualization.
    """
    embedding = _model.encode(text, normalize_embeddings=True)
    idx = int(abs(embedding[0]) * _cmap_count) % _cmap_count
    cmap = _cmap_list[idx]
    params = {
        "c": complex(embedding[1] % 2 - 1, embedding[2] % 2 - 1),
        "zoom": 0.5 + abs(embedding[3]) * 0.5,
        "max_iter": int(50 + abs(embedding[4]) * 150),
        "particle_density": 0.5 + abs(embedding[5]) * 1.5,
        "cmap": cmap,
    }
    return embedding, params
