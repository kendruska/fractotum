# procedural_generation.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.animation import FuncAnimation, FFMpegWriter

def generate_advanced_fractal(c, zoom, max_iter, particle_density, embed, size=1000):
    x = np.linspace(-1.5 * zoom, 1.5 * zoom, size)
    y = np.linspace(-1.5 * zoom, 1.5 * zoom, size)
    X, Y = np.meshgrid(x, y)

    distortion_strength = 0.05
    X += np.sin(X * embed[6]) * distortion_strength
    Y += np.cos(Y * embed[7]) * distortion_strength

    Z = X + 1j * Y
    img = np.zeros(Z.shape)
    particles = np.zeros(Z.shape)

    for i in range(max_iter):
        mask = np.abs(Z) < 10
        angle = embed[12] % (2 * np.pi)
        Z[mask] *= np.exp(1j * angle)  # Rotational effect
        center = complex(embed[10] % 2 - 1, embed[11] % 2 - 1)
        Z += (center - Z) * 0.0025  # Gravitational pull effect
        img[mask] += particle_density
        particles[mask] += np.exp(-i * 0.01)
        Z[mask] = Z[mask] ** 2 + c

    img = np.clip(img, 0, 255)
    img_blur = gaussian_filter(img, sigma=1)

    cx, cy = size // 2, size // 2
    Yidx, Xidx = np.indices((size, size))
    rays = np.cos(0.02 * (Xidx - cx)) + np.sin(0.02 * (Yidx - cy))
    rays = gaussian_filter(rays, sigma=10)
    rays = (rays - rays.min()) / (rays.max() - rays.min())

    final = img_blur + 2 * particles + 0.7 * rays * img_blur
    final = np.clip(final, 0, np.percentile(final, 99))
    final = (final - final.min()) / (final.max() - final.min())
    return final

def render_image(embed, params, output_path):
    fractal = generate_advanced_fractal(params['c'], params['zoom'], params['max_iter'], params['particle_density'], embed)
    plt.figure(figsize=(10, 10))
    plt.imshow(fractal, cmap=params['cmap'], interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Image saved to: {output_path}")

def render_video(embed, params, output_path, frames=30):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(np.zeros((800, 800)), cmap='magma', interpolation='bilinear')
    ax.axis('off')

    def update(i):
        alpha = i / frames
        noise = np.sin(2 * np.pi * i / frames) * 0.02
        new_embed = embed + noise
        fractal = generate_advanced_fractal(params['c'], params['zoom'], params['max_iter'], params['particle_density'], new_embed, size=800)
        im.set_array(fractal)
        im.set_cmap(params['cmap'])
        return [im]

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    writer = FFMpegWriter(fps=10, bitrate=2000)
    anim.save(output_path, writer=writer)
    plt.close()
    print(f"Video saved to: {output_path}")