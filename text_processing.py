# text_processing.py
from sentence_transformers import SentenceTransformer
import numpy as np

def get_embedding_and_params(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embed = model.encode(text)

    c = complex((embed[0] % 2) - 1, (embed[1] % 2) - 1)
    zoom = 0.5 + abs(embed[2] % 1.5)
    max_iter = int(300 + (abs(embed[3]) * 200) % 500)
    particle_density = 0.8 + (abs(embed[4]) % 1.5)
    colormaps = ['magma', 'inferno', 'twilight_shifted', 'plasma', 'cividis']
    cmap = colormaps[int(abs(embed[5] * 10)) % len(colormaps)]

    params = {
        'c': c,
        'zoom': zoom,
        'max_iter': max_iter,
        'particle_density': particle_density,
        'cmap': cmap
    }

    return embed, params

