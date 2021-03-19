import cv2
import imageio
from tqdm import tqdm

from basic_transform import (blur, contrast_and_brighten, gamma_saturation,
                             lcd_overlay, noise, rotate, scanner_like, shadow,
                             watermark, wrinkles)
from distortion import distort, perspective, stretch

augmentations = [
    rotate,
    shadow,
    watermark,
    wrinkles,
    lcd_overlay,
    gamma_saturation,
    contrast_and_brighten,
    scanner_like,
    distort,
    perspective,
    stretch,
    blur,
    noise,
]

image = cv2.imread("data/sample.jpg")


def create_gif(image_list, gif_name, duration=1):
    frames = []
    for image in image_list:
        frames.append(image)
    imageio.mimsave(gif_name, frames, "GIF", duration=duration)
    return


for aug in tqdm(augmentations):
    aug_list = []
    for i in range(5):
        result = aug(image)
        aug_list.append(result[:, :, ::-1])
    create_gif(aug_list, f"output/{aug.__name__}.gif")
