import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.animation import FuncAnimation, FFMpegWriter
from PIL import Image
from numba import jit

@jit(nopython=True, fastmath=True, cache=True)
def mandelbrot_core(Z, c, max_iter, particle_density):
    """
    Optimized Mandelbrot computation using numba for performance.

    Args:
        Z (np.ndarray): Complex grid for Mandelbrot calculation.
        c (complex): Complex parameter for Mandelbrot set.
        max_iter (int): Maximum number of iterations.
        particle_density (float): Density value for particle effect.
    Returns:
        tuple: (img, particles) arrays representing fractal and particle overlays.
    """
    size = Z.shape[0]
    img = np.zeros((size, size), dtype=np.float32)
    particles = np.zeros((size, size), dtype=np.float32)
    for i in range(max_iter):
        mask = np.abs(Z) < 10
        for x in range(size):
            for y in range(size):
                if mask[x, y]:
                    img[x, y] += particle_density
                    particles[x, y] += np.exp(-i * 0.01)
                    Z[x, y] = Z[x, y] ** 2 + c
    return img, particles

def generate_advanced_fractal(c, zoom, max_iter, particle_density, embed, size=1000):
    """
    Generate a fractal image with advanced effects and embedding-based distortion.

    Args:
        c (complex): Complex parameter for Mandelbrot set.
        zoom (float): Zoom factor for the fractal.
        max_iter (int): Maximum number of iterations.
        particle_density (float): Density value for particle effect.
        embed (np.ndarray): Embedding array for controlling distortion and effects.
        size (int): Output image size (pixels).
    Returns:
        np.ndarray: Final normalized fractal image.
    """
    x = np.linspace(-1.5 * zoom, 1.5 * zoom, size, dtype=np.float32)
    y = np.linspace(-1.5 * zoom, 1.5 * zoom, size, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='ij')
    distortion_strength = 0.05
    X += np.sin(X * embed[6]) * distortion_strength
    Y += np.cos(Y * embed[7]) * distortion_strength
    Z = (X + 1j * Y).astype(np.complex64)
    img, particles = mandelbrot_core(Z, c, max_iter, particle_density)
    img = np.clip(img, 0, 255)
    img_blur = gaussian_filter(img, sigma=1)
    center_coord = size // 2
    y_idx, x_idx = np.ogrid[:size, :size]
    rays = np.cos(0.02 * (x_idx - center_coord)) + np.sin(0.02 * (y_idx - center_coord))
    rays = gaussian_filter(rays, sigma=10)
    rays = (rays - rays.min()) / (rays.max() - rays.min())
    final = img_blur + 2 * particles + 0.7 * rays * img_blur
    final = np.clip(final, 0, np.percentile(final, 99))
    final = (final - final.min()) / (final.max() - final.min())
    return final

def render_image(embed, params, output_path):
    """
    Render a fractal image from embedding and parameters, and save to file.

    Args:
        embed (np.ndarray): Embedding array for fractal control.
        params (dict): Dictionary of fractal parameters (c, zoom, max_iter, particle_density, cmap).
        output_path (str): Path to save the output image.
    Returns:
        None
    """
    fractal = generate_advanced_fractal(params['c'], params['zoom'], params['max_iter'], params['particle_density'], embed)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(fractal, cmap=params['cmap'], interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    print(f"✅ Image saved to: {output_path}")

def render_video(embed, params, output_path, frames=60):
    """
    Render a zooming fractal video from embedding and parameters, and save to file.

    Args:
        embed (np.ndarray): Embedding array for fractal control.
        params (dict): Dictionary of fractal parameters (c, zoom, max_iter, particle_density, cmap).
        output_path (str): Path to save the output video.
        frames (int): Number of frames in the video.
    Returns:
        None
    """
    base_fractal = generate_advanced_fractal(params['c'], params['zoom'], params['max_iter'], params['particle_density'], embed, size=600)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(base_fractal, cmap=params['cmap'], interpolation='bilinear')
    ax.axis('off')
    zoom_factors = 1 + 0.02 * np.arange(frames) / frames
    brightness_values = 1 + 0.5 * np.sin(2 * np.pi * np.arange(frames) / frames)
    def zoom_into_array(array, zoom_factor):
        center = np.array(array.shape) // 2
        size_crop = (np.array(array.shape) / zoom_factor).astype(int)
        start = np.maximum(center - size_crop // 2, 0)
        end = np.minimum(center + size_crop // 2, array.shape)
        cropped = array[start[0]:end[0], start[1]:end[1]]
        resized = np.array(Image.fromarray((cropped * 255).astype(np.uint8)).resize(array.shape[::-1], resample=Image.LANCZOS))
        return np.clip(resized / 255, 0, 1)
    def update(i):
        zoomed = zoom_into_array(base_fractal, zoom_factors[i])
        modulated = np.clip(zoomed * brightness_values[i], 0, 1)
        im.set_array(modulated)
        return [im]
    anim = FuncAnimation(fig, update, frames=frames, blit=True, interval=50)
    writer = FFMpegWriter(fps=20, bitrate=3000)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"🎞️ Video saved to: {output_path}")
