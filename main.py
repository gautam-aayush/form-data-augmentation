from distortion import perspective
from pathlib import Path

import click
import cv2
import numpy as np
from pdf2image import convert_from_path
from tqdm import tqdm

from basic_transform import (contrast_and_brighten, gamma_saturation,
                             lcd_overlay, rotate, scanner_like, shadow,
                             virtual_background, watermark, wrinkles, shearing, displace)
from composite_transform import (background_with_lcd_stretch,
                                 rotation_with_lcd, wrinkle_with_noise)


def get_image(filename, page=1):
    if filename.lower().endswith("pdf"):
        image = convert_from_path(filename, 600)
        image = np.asarray(image[0])
    else:
        image = cv2.imread(filename)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, img_gray


augmentations = [
    rotate,
    shadow,
    # watermark,
    wrinkles,
    shearing,
    displace,
    # lcd_overlay,
    # gamma_saturation,
    contrast_and_brighten,
    scanner_like,
    # virtual_background,
    # rotation_with_lcd,
    wrinkle_with_noise,
    perspective,
    # background_with_lcd_stretch,
]


@click.command()
@click.option("--data-root", type=click.Path(exists=True), default="data/")
@click.option("--output-dir", type=click.Path(exists=True), default="output/")
@click.option("--aug-prob",type=float, default=1.0)
def main(data_root, output_dir, aug_prob):
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    # list all jpg, pdf or png files
    aug_files = list(data_root.rglob("*.[jp][pnd][gf]"))

    for file in tqdm(aug_files):
        org_img, _ = get_image(str(file))
        data_inner = file.parts[1:-1]
        if not data_inner:
            data_inner = ""
        else:
            data_inner = Path(*data_inner)
        for i, aug in enumerate(augmentations):
            if np.random.rand() < aug_prob:
                result = aug(org_img)
                filename = file.parts[-1]
                new_filename = f"{filename.split('.')[0]}_aug_{i+1}.jpg"
                output_path = Path(output_dir, data_inner, new_filename)
                cv2.imwrite(str(output_path), result)


if __name__ == "__main__":
    main()
