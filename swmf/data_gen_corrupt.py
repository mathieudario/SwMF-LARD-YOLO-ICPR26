# -*- coding: utf-8 -*-

"""
Code adapted from https://github.com/hendrycks/robustness/tree/master
Benchmarking Neural Network Robustness to Common Corruptions and Perturbations (ICLR 2019), Hendryck & Dietterich
"""

from pathlib import Path
import os
import numpy as np
from PIL import Image
import skimage as sk
from skimage.filters import gaussian
import cv2
from scipy.ndimage import zoom as scizoom
import tqdm


MAX_SEVERITY = 3  # 3 levels of severity


# ========== Distortion Helpers ==========
def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=512, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


# ========== Distortions ==========
def gaussian_noise(x, severity=1):
    c = [0.05, 0.09, 0.14][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [500, 250, 100, 75, 50][severity - 1]
    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.01, .02, .03, .05, .07][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [0.06, 0.1, 0.12, 0.16, 0.2][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    c = [.4, .6, 0.7, .8, 1][severity - 1]
    x = gaussian(np.array(x) / 255., sigma=c)
    return np.clip(x, 0, 1) * 255


def frosted_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.2, 1, 1), (0.4, 2, 2), (0.7, 3, 3)][severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0]) * 255)

    h, w = x.shape[:2]
    for _ in range(c[2]):
        # Generate random offsets for all pixels
        dx = np.random.randint(-c[1], c[1], size=(h, w))
        dy = np.random.randint(-c[1], c[1], size=(h, w))

        # Calculate new indices
        h_prime = np.clip(np.arange(h)[:, None] + dy, 0, h - 1)
        w_prime = np.clip(np.arange(w)[None, :] + dx, 0, w - 1)

        # Swap pixels using advanced indexing
        x[:], x[h_prime, w_prime] = x[h_prime, w_prime], x[:]

    return np.clip(gaussian(x / 255., sigma=c[0]), 0, 1) * 255


def defocus_blur(x, severity=1):
    img_size = x.shape[0]
    c = [(0.002, 0.4), (0.006, 0.5), (0.010, 0.6)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0]*img_size, alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x32x32 -> 32x32x3

    return np.clip(channels, 0, 1) * 255


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.06, 0.01), np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.01), np.arange(1, 1.26, 0.01)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
    c = [(0.5, 2.5), (0.9, 2.0), (1.3, 1.5)][severity - 1]
    
    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:x.shape[0], :x.shape[1]][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def brightness(x, severity=1):
    c = [0.25, 0.40, 0.55][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.
    x = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
    return np.clip(x, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.75, 0.5, 0.4, 0.3, 0.15][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def pixelate(x, severity=1):
    c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]
    img = Image.fromarray(x)
    img = img.resize((int(x.shape[1] * c), int(x.shape[0] * c)), Image.BOX)
    img = img.resize((x.shape[1], x.shape[0]), Image.BOX)
    return np.array(img)


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)][severity - 1]
    x = np.array(x) / 255.
    x = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [80, 65, 58, 50, 40][severity - 1]
    img = Image.fromarray(x)
    img.save("temp.jpg", "JPEG", quality=c)
    return np.array(Image.open("temp.jpg"))


# ========== Main Routine ==========
def create_dataset_corrupt(src_path, verbose=False):
    if not isinstance(src_path, Path):
        src_path = Path(src_path).resolve()

    if not src_path.exists():
        raise ValueError(f"Invalid path {src_path}")
    
    dst_path = src_path.parent / (src_path.stem + "_corrupt")

    if not dst_path.exists():
        dst_path.mkdir(exist_ok=True, parents=True)

    image_files = [f for f in os.listdir(src_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    corruptions = {
        'Gaussian_Noise': gaussian_noise,
        # 'Shot_Noise': shot_noise,
        # 'Impulse_Noise': impulse_noise,
        # 'Speckle_Noise': speckle_noise,
        # 'Gaussian_Blur': gaussian_blur,
        'Defocus_Blur': defocus_blur,
        'Frosted_Blur': frosted_blur,
        # 'Zoom_Blur': zoom_blur,
        'Fog': fog,
        'Brightness': brightness,
        # 'Contrast': contrast,
        # 'Saturate': saturate,
        # 'Pixelate': pixelate,
        # 'JPEG': jpeg_compression,
    }

    for corrupt_id, corrupt_fn in corruptions.items():
        if verbose:
            print(f'Creating images for the corruption: {corrupt_id}')

        for severity in range(1, MAX_SEVERITY + 1):
            if verbose:
                print(f"Severity {severity}")

            severity_dst_path = dst_path / f"{corrupt_id}_severity_{severity}"
            severity_dst_path.mkdir(exist_ok=True, parents=True)

            for img_file in tqdm.tqdm(image_files):
                img_path = src_path / img_file
                img = Image.open(img_path)
                corrupted_img = corrupt_fn(np.array(img), severity)
                corrupted_img = Image.fromarray(corrupted_img.astype('uint8'))
                corrupted_img.save(severity_dst_path / img_file)
                