import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.animation import FuncAnimation, FFMpegWriter
from PIL import Image

def generate_advanced_fractal(c, zoom, max_iter, particle_density, embed, size=1000, until_iter=None):
    x = np.linspace(-1.5 * zoom, 1.5 * zoom, size)
    y = np.linspace(-1.5 * zoom, 1.5 * zoom, size)
    X, Y = np.meshgrid(x, y)

    distortion_strength = 0.05
    X += np.sin(X * embed[6]) * distortion_strength
    Y += np.cos(Y * embed[7]) * distortion_strength

    Z = X + 1j * Y
    img = np.zeros(Z.shape)
    particles = np.zeros(Z.shape)

    if until_iter is None:
        until_iter = max_iter

    for i in range(until_iter):
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

def render_video(embed, params, output_path, frames=60):
    base_fractal = generate_advanced_fractal(
        params['c'], params['zoom'], params['max_iter'],
        params['particle_density'], embed, size=800
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(base_fractal, cmap=params['cmap'], interpolation='bilinear')
    ax.axis('off')

    zoom_base = 1.0
    zoom_amp = 0.02 + abs(embed[0] % 0.05)
    zoom_freq = 1 + abs(embed[1] % 3.0)

    brightness_amp = 0.4 + abs(embed[4] % 0.5)
    brightness_freq = 0.5 + abs(embed[5] % 2.5)

    x_glitch = (embed[6] % 0.01) - 0.005
    y_glitch = (embed[7] % 0.01) - 0.005

    offset_amp = int(3 + embed[9] % 6)

    def zoom_into_array(array, zoom_factor, offset_x=0, offset_y=0):
        center = np.array(array.shape) // 2 + np.array([offset_y, offset_x])
        size = (np.array(array.shape) // zoom_factor).astype(int)
        start = np.clip(center - size // 2, 0, array.shape[0] - 1)
        end = np.clip(center + size // 2, 0, array.shape[0])
        cropped = array[start[0]:end[0], start[1]:end[1]]
        resized = Image.fromarray((cropped * 255).astype(np.uint8)).resize(array.shape[::-1], resample=Image.BICUBIC)
        return np.clip(np.array(resized) / 255, 0, 1)

    def update(i):
        t = i / frames
        zoom = zoom_base + zoom_amp * np.sin(2 * np.pi * zoom_freq * t)
        brightness = 1 + brightness_amp * np.sin(2 * np.pi * brightness_freq * t)
        brightness = max(brightness, 0.7)

        # Offset glitch effect
        dx = int(np.sin(t * 2 * np.pi) * offset_amp)
        dy = int(np.cos(t * 2 * np.pi) * offset_amp)

        zoomed = zoom_into_array(base_fractal, zoom, offset_x=dx, offset_y=dy)
        shifted = np.roll(zoomed, shift=(int(y_glitch * i * 100), int(x_glitch * i * 100)), axis=(0, 1))
        modulated = np.clip(shifted * brightness, 0, 1)
        im.set_array(modulated)
        return [im]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    writer = FFMpegWriter(fps=15, bitrate=2000)
    anim.save(output_path, writer=writer)
    plt.close('all')
    print(f"🎞️ Video saved to: {output_path}")