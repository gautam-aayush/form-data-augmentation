from linear_augmentation import *
from non_linear_augmentation import *
import cv2
import imageio
from tqdm import tqdm

augmentations = [
    rotate,
    add_shadow,
    add_watermark,
    add_wrinkles,
    add_lcd_overlay,
    gamma_saturation,
    contrast_and_brighten,
    scanner_like,
    distort,
    perspective,
    stretch,
    blur,
    add_noise,
]

image = cv2.imread(
    "/home/aayush-fm/Documents/fuse/fuse-extract-v2-ai/fuse/notebooks/exploratory/data/W2/horizontal/w2_horizontal_052.jpg"
)


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
