from .linear_augmentation import *
from .non_linear_augmentation import *
import click
from pathlib import Path
from tqdm import tqdm
import numpy as np


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
    add_virtual_background,
]


@click.command()
@click.option("--data_root", type=click.Path(exists=True))
@click.option("--output_dir", type=click.Path(exists=True))
@click.option("--aug_prob", default=0.1)
def main(data_root, output_dir, aug_prob):
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    aug_files = list(data_root.rglob("*.jpg"))

    for file in tqdm(aug_files):
        result, _ = get_image(str(file))
        is_augmented = False
        for aug in augmentations:
            if np.random.rand() < aug_prob:
                is_augmented = True
                result = aug(result)
        data_inner = file.parts[1:-1]
        if not data_inner:
            data_inner = ""
        else:
            data_inner = Path(*data_inner)
        if is_augmented:
            print("Augmented")
        else:
            print("Not Augmented")
        filename = file.parts[-1]
        new_filename = filename.split(".")[0] + "_aug.jpg"
        # import ipdb
        # ipdb.set_trace()
        output_path = Path(output_dir, data_inner, new_filename)
        cv2.imwrite(str(output_path), result)


if __name__ == "__main__":
    main()
