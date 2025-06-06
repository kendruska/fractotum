from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt

model = SentenceTransformer("all-mpnet-base-v2")

def get_embedding_and_params(text):
    embedding = model.encode(text, normalize_embeddings=True)
    embed = np.array(embedding)

    cmap_list = sorted(c for c in plt.colormaps() if not c.endswith('_r'))
    cmap = cmap_list[int(abs(embed[0]) * len(cmap_list)) % len(cmap_list)]

    params = {
        "c": complex(embed[1] % 2 - 1, embed[2] % 2 - 1),
        "zoom": 0.5 + abs(embed[3]) * 0.5,
        "max_iter": int(50 + abs(embed[4]) * 150),
        "particle_density": 0.5 + abs(embed[5]) * 1.5,
        "cmap": cmap,
    }

    return embed, params
