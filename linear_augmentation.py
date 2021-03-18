import cv2
import numpy as np
from pathlib import Path
from affine_transform import rotation
from utility import _perspective_warp, _add_texture, _generate_shadow_coordinates
from non_linear_augmentation import distort

PATH_TO_WRINKLED_TEXTURE = Path(
    "augmentation-helpers/overlays/wrinkle"
)
PATH_TO_MONITOR_TEXTURE = Path(
    "augmentation-helpers/overlays/monitor"
)
PATH_TO_BG_IMAGES = Path(
    "augmentation-helpers/background"
)

assert PATH_TO_WRINKLED_TEXTURE.exists()
assert PATH_TO_MONITOR_TEXTURE.exists()
assert PATH_TO_BG_IMAGES.exists()


def add_noise(image: np.ndarray, noise_typ: str = None) -> np.ndarray:
    """
    Adds noise to an image. Avaiable noise_types "gauss",
    "s&p" (salt and pepper)
    Args:
        image (np.ndarray): BGR image on which to add noise
        noise_typ (str, optional): type of noise to add: "gauss" or "s&p".
        Defaults to None.

    Returns:
        np.ndarray: BGR image with noise added
    """
    noise_types = ["gauss", "s&p"]
    if not noise_typ:
        noise_typ = np.random.choice(noise_types)
    if noise_typ == "gauss":
        height, width, ch = image.shape
        mean = 0  # gaussian mean
        var = 30  # gaussian variance
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (height, width, ch))
        noisy = image + gauss
        return noisy.astype(np.uint8)
    elif noise_typ == "s&p":
        height, width, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004  # fraction of image to be converted to noise
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        # get random coordinates for sale noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        # get random coordinated for pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0

        return out


def add_shadow(image: np.ndarray, no_of_shadows: int = 1) -> np.ndarray:
    """Add shadow to an image by decreasing lightness of
    random polygonal regions in an image
    Note: As the number of shadows increase, there are chances of overlapping
    of shadows which causes the brightness of overlapped region to decrease further
    Args:
        image (np.ndarray): BGR image to add shadow on
        no_of_shadows (int, optional): Number of shadows to add. Defaults to 1.

    Returns:
        np.ndarray: image with shadows
    """
    # convert to HLS
    image_HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    mask = np.zeros_like(image)
    imshape = image.shape[:2]
    vertices_list = _generate_shadow_coordinates(imshape, no_of_shadows)
    # get list of shadow vertices
    for vertices in vertices_list:
        # add all shadow polygons on empty mask,
        # single 255 denotes only blue channel
        cv2.fillPoly(mask, vertices, 255)
        # if blue channel is hot, lower the birghtness for light channel
        image_HLS[:, :, 1][mask[:, :, 0] == 255] = (
            image_HLS[:, :, 1][mask[:, :, 0] == 255] * 0.75
        )
        # convert to BGR
        image_BGR = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2BGR)
    return image_BGR


def add_virtual_background(image: np.ndarray, bg_image: np.ndarray=None, scale: float=1.25) -> np.ndarray:
    """[summary]
    Adds a background to an image
    Args:
        image (np.ndarray): [description]
        bg_image (np.ndarray, optional): [description]. Defaults to None.
        scale (float, optional): [description]. Defaults to 1.25.

    Returns:
        np.ndarray: [description]
    """
    files = sorted(PATH_TO_BG_IMAGES.glob("*.jpg"))
    index = np.random.randint(0, len(files))
    bg_image = cv2.imread(str(files[index]))

    org_height, org_width = image.shape[:2]
    new_img_height, new_img_width = int(org_height * scale), int(org_width * scale)
    bg_image = cv2.resize(bg_image, (new_img_width, new_img_height))

    warped_fg_image, pts = _perspective_warp(bg_image, image)
    tmp_img = np.ones(bg_image.shape, dtype=np.uint8) * 255
    filled_img = cv2.fillPoly(tmp_img, np.int32([pts]), (0, 0, 0))
    masked = cv2.bitwise_and(bg_image, filled_img)
    full_n_final = cv2.bitwise_or(masked, warped_fg_image)

    return full_n_final


def add_watermark(image):
    texts = ["confidential", "fusemachines", "official", "W2-Tax"]
    text = texts[np.random.randint(0, len(texts))]
    loc = np.random.randint(image.shape[0] // 4, image.shape[1] // 2, 2)

    image_with_text = cv2.putText(
        np.zeros(image.shape, image.dtype),
        text,
        tuple(loc),
        cv2.FONT_HERSHEY_PLAIN,
        20,
        tuple(map(int, np.random.randint(0, 255, 3))),
        15,
        cv2.LINE_AA,
    )
    rotation_angle = np.random.rand() * 90 - 90
    image_with_text = rotation(image_with_text, rotation_angle, same=True)
    image_with_text += image
    #     print(image_with_text.shape, image.shape)
    final_image = cv2.addWeighted(image, 0.7, image_with_text, 0.3, 0)
    return final_image


def add_wrinkles(image):
    files = sorted(PATH_TO_WRINKLED_TEXTURE.glob("*.jpg"))
    index = np.random.randint(0, len(files))
    overlay = cv2.imread(str(files[index]))
    textured = _add_texture(image, overlay)
    distorted = distort(textured)
    final_img = add_noise(distorted)
    return final_img


def add_lcd_overlay(image):
    files = sorted(PATH_TO_MONITOR_TEXTURE.glob("*.jpg"))
    index = np.random.randint(0, len(files))
    overlay = cv2.imread(str(files[index]))
    return _add_texture(image, overlay)


def rotate(image):
    random = np.random.rand()
    if random < 0.3:
        angle = np.random.rand() * 10 - 10
    elif random < 0.7:
        angle = 90
    else:
        angle = 180
    rotated_img = rotation(image, angle)
    return rotated_img


def blur(image, sigma_x=100, sigma_y=100):
    blurred_image = cv2.GaussianBlur(image, (5, 5), sigma_x, sigma_y)
    return blurred_image


def contrast_and_brighten(image, contrast=None, brightness=None):
    if not contrast:
        contrast = np.random.rand(1) + 0.5
    if not brightness:
        brightness = np.random.randint(-50, 100)
    new_image = np.clip(image.astype(np.int64) * contrast + brightness, 0, 255)
    new_image = new_image.astype(image.dtype)
    return new_image


def scanner_like(image):
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        image_grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    image_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return image_bgr


def gamma_saturation(image, gamma=None):
    if not gamma:
        if np.random.rand(1) < 0.5:
            # lower saturation
            gamma = np.random.rand(1)
        else:
            # increase saturation
            gamma = np.random.randint(1, 11)
    lookup_table = np.zeros((1, 256), np.uint8)
    for i in range(256):
        lookup_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    new_img = cv2.LUT(image, lookup_table)
    return new_img
