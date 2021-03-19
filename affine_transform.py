import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path


def displacement(image, horizontal, vertical):
    height, width = image.shape[:2]
    translation_mat = np.array([[1, 0, horizontal], [0, 1, vertical]], dtype=np.float32)
    displaced_image = cv2.warpAffine(
        image, translation_mat, (width + horizontal, height + vertical)
    )
    return displaced_image


def rotation(image, angle=90, same=False):
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

    rotated_image = cv2.warpAffine(
        image, rotation_mat, (new_width, new_height), borderValue=(255, 255, 255)
    )

    return rotated_image



def shear(image, shear_X, shear_Y):
    width, height = image.shape[:2]
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
