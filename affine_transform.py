from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def displacement(
    image: np.ndarray, horizontal_scale: float = 0.1, vertical_scale: float = 0.1
) -> np.ndarray:
    """
    Displaces an image horzontally and vertically by respective scales
    Args:
        image (np.ndarray): BGR image
        horizontal_scale (float, optional): Fraction of original image width to displace.
        Defaults to 0.1.
        vertical_scale (float, optional): Fraction of original image width to displace.
        Defaults to 0.1.

    Returns:
        np.ndarray: Displaced image
    """
    height, width = image.shape[:2]
    horizontal = int(horizontal_scale * width)
    vertical = int(vertical_scale * height)
    translation_mat = np.array([[1, 0, horizontal], [0, 1, vertical]], dtype=np.float32)
    border_color = (255, 255, 255)  # white borders
    displaced_image = cv2.warpAffine(
        image,
        translation_mat,
        (width + horizontal, height + vertical),
        borderValue=border_color,
    )
    return displaced_image


def rotation(image: np.ndarray, angle: int = 90, same: bool = False) -> np.ndarray:
    """
    Rotates an image through given angle
    Args:
        image (np.ndarray): BGR image
        angle (int, optional): Rotation angle (counterclockwise) about its center. Defaults to 90.
        same (bool, optional): When True the output image is of the same size as input
        however, some portion of the original image may be lost. When False, height and width
        are adjusted to preserve original image content.
        Defaults to False.

    Returns:
        np.ndarray: Rotated image
    """
    height, width = image.shape[:2]
    centerX = (width - 1) / 2
    centerY = (height - 1) / 2
    rotation_mat = cv2.getRotationMatrix2D((centerX, centerY), angle, 1)

    if same:
        new_width = width
        new_height = height
    else:
        cos = np.abs(rotation_mat[0, 0])
        sin = np.abs(rotation_mat[0, 1])
        # compute the new bounding dimensions of the image
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        # adjust the rotation matrix to take into account translation
        rotation_mat[0, 2] += (new_width / 2) - centerX
        rotation_mat[1, 2] += (new_height / 2) - centerY

    border_color = (255, 255, 255)  # white borders
    rotated_image = cv2.warpAffine(
        image, rotation_mat, (new_width, new_height), borderValue=border_color
    )

    return rotated_image


def shear(image: np.ndarray, shear_X: float = 0.1, shear_Y: float = 0.1) -> np.ndarray:
    """
    Shears the image along x and y directions
    Args:
        image (np.ndarray): BGR image
        shear_X (float, optional): Value of horizontal shear. Defaults to 0.1.
        shear_Y (float, optional): Value of vertical shear. Defaults to 0.1.

    Returns:
        np.ndarray: Sheared image
    """
    width, height = image.shape[:2]

    # increase image height and width to preserve image content
    new_width = int(2 * width)
    new_height = int(2 * height)
    M2 = np.float32([[1, shear_Y, 0], [shear_X, 1, 0]])
    #     M2[0,2] = -M2[0,1] * W/2
    #     M2[1,2] = -M2[1,0] * H/2
    centerX = (width - 1) / 2
    centerY = (height - 1) / 2
    M2[0, 2] += (new_width / 2) - centerX
    M2[1, 2] += (new_height / 2) - centerY

    sheared_image = cv2.warpAffine(image, M2, (new_width, new_height))
    return sheared_image


def arg_to_string(arg):
    result = "("
    i = 0
    for key, value in arg.items():
        if i != 0:
            result += "_"
        result += key
        result += "_" + str(value)
        i += 1
    result += ")"
    return result


if __name__ == "__main__":
    data_dir = Path("../fuse/notebooks/exploratory/data/Ncell-Phase3")
    output_dir = Path("../fuse/notebooks/exploratory/outputs/Ncell/augmented_data2")
    output_dir.mkdir(exist_ok=True)
    # transformations = [displacement, rotation, noise, shear]
    # transformation_names = ['displacement', 'rotation', 'gaussian_noise', 'shear']
    # transform_params = {'displacement': [{'horizontal': 100, 'vertical': 100}, {'horizontal': 200, 'vertical': 200},
    #                                      {'horizontal': 500, 'vertical': 500}],
    #                     'rotation': [{'angle': 30}, {'angle': 45}, {'angle': 60}, {'angle': 90}],
    #                     'gaussian_noise': [{'var': 10}, {'var': 20}, {'var': 30}],
    #                     'shear': [{'shear_X': 0.1, 'shear_Y': 0.1}, {'shear_X': 0.1, 'shear_Y': 0.2},
    #                               {'shear_X': 0.2, 'shear_Y': 0.1}]}
    #
    # for file in tqdm(list(data_dir.glob('*.jpg'))):
    #     for i, transformation in enumerate(transformations):
    #         for arg in transform_params[transformation_names[i]]:
    #             image = cv2.imread(str(file))
    #             transformed_img = transformation(image, **arg)
    #             new_filename = f'{transformation_names[i]}_{arg_to_string(arg)}_{file.parts[-1]}'
    #             cv2.imwrite(str(Path(output_dir, new_filename)), transformed_img)

    transformations = [rotation]
    transformation_names = ["rotation", "shear"]
    transform_params = {"rotation": [{"angle": 11}, {"angle": 180}]}

    for file in tqdm(list(data_dir.glob("*.jpg"))):
        for i, transformation in enumerate(transformations):
            for arg in transform_params[transformation_names[i]]:
                image = cv2.imread(str(file))
                transformed_img = transformation(image, **arg)
                new_filename = (
                    f"{transformation_names[i]}_{arg_to_string(arg)}_{file.parts[-1]}"
                )
                cv2.imwrite(str(Path(output_dir, new_filename)), transformed_img)
