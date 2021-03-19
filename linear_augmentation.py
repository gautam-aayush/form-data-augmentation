import cv2
import numpy as np
from pathlib import Path
from affine_transform import rotation
from utility import _perspective_warp, _add_texture, _generate_shadow_coordinates
from non_linear_augmentation import distort

PATH_TO_WRINKLED_TEXTURE = Path("augmentation-helpers/overlays/wrinkle")
PATH_TO_MONITOR_TEXTURE = Path("augmentation-helpers/overlays/monitor")
PATH_TO_BG_IMAGES = Path("augmentation-helpers/background")

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


def add_virtual_background(
    image: np.ndarray,
    bg_image: np.ndarray = None,
    scale: float = 1.25,
    interactive: bool = False,
) -> np.ndarray:
    """
    Adds a background to an image by warping the image into the perspective of background.
    Args:
        image (np.ndarray): BGR image, foreground
        bg_image (np.ndarray, optional): BGR image background. Defaults to None.
        If None, a random image from a predefined list is chosen as background
        scale (float, optional): scale of background with respect to foreground. Defaults to 1.25.
        interactive (bool, optional): If True, an interactive window allows to choose
        the perspective points in the background image, otherwise random points are chosen.
        Defaults to False.

    Returns:
        np.ndarray: [description]
    """
    if not bg_image:
        files = sorted(PATH_TO_BG_IMAGES.glob("*.jpg"))
        chosen_file = np.random.choice(files)
        bg_image = cv2.imread(str(chosen_file))

    # resize bg image to approproate scale
    fg_height, fg_width = image.shape[:2]
    new_height, new_width = int(fg_height * scale), int(fg_width * scale)
    bg_image = cv2.resize(bg_image, (new_width, new_height))

    # warp image in perspective of background
    warped_fg_image, pts = _perspective_warp(bg_image, image)

    # create a white image like with same shape as bg
    img_buffer = np.ones(bg_image.shape, dtype=np.uint8) * 255
    # blacken the part where fg image goes to create a warped image template
    warp_template = cv2.fillPoly(img_buffer, np.int32([pts]), (0, 0, 0))

    # mask out bg image with the warp_template
    masked = cv2.bitwise_and(bg_image, warp_template)
    # combine the warped_fg_image to the masked bg
    final_image = cv2.bitwise_or(masked, warped_fg_image)

    return final_image


def add_watermark(image: np.ndarray, text: str = None) -> np.ndarray:
    """
    Add watermark text to an image
    Args:
        image (np.ndarray): BGR image
        text (str, optional): text for watermark. Defaults to None.
        When None, a random text is chosen from a pre-defined list

    Returns:
        np.ndarray: BGR image with watermark added
    """
    texts = ["confidential", "fusemachines", "official", "W2-Tax"]
    if not text:
        text = np.random.choice(texts)

    # choose a random location for watermark
    loc = np.random.randint(image.shape[0] // 4, image.shape[1] // 2, 2)

    # write text in solid on an all black image
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

    # rotate the text at a random angle
    rotation_angle = np.random.rand() * 90 - 90
    image_with_text = rotation(image_with_text, rotation_angle, same=True)

    # add rotated text to image
    image_with_text += image
    # again add the original image to the image with text with
    # different weights to get a semi transparent look
    alpha = 0.7
    beta = 1 - alpha
    final_image = cv2.addWeighted(image, alpha, image_with_text, beta, 0)
    return final_image


def add_wrinkles(image: np.ndarray, wrinkled_overlay: np.ndarray = None) -> np.ndarray:
    """
    Adds wrinkles to an image
    Args:
        image (np.ndarray): Original BGR image
        wrinkled_overlay (np.ndarray, optional): Wrinkled texture image to overlay.
        Defaults to None.

    Returns:
        np.ndarray: wrinkled image
    """
    if not wrinkled_overlay:
        # randomly choose a texture for overlay
        files = sorted(PATH_TO_WRINKLED_TEXTURE.glob("*.jpg"))
        chosen_file = np.random.choice(files)
        wrinkled_overlay = cv2.imread(str(chosen_file))
    # add wrinkled texture
    textured = _add_texture(image, wrinkled_overlay)
    # get a distortion in text
    distorted = distort(textured)
    # add noise
    final_img = add_noise(distorted)
    return final_img


def add_lcd_overlay(image: np.ndarray, overlay: np.ndarray = None) -> np.ndarray:
    """
    Add a LCD texture to an image
    Args:
        image (np.ndarray): BGR image
        overlay (np.ndarray, optional): BGR image with LCD texture.
        Defaults to None.

    Returns:
        np.ndarray: [description]
    """
    if not overlay:
        # randomly choose a texture for overlay
        files = sorted(PATH_TO_MONITOR_TEXTURE.glob("*.jpg"))
        index = np.random.randint(0, len(files))
        overlay = cv2.imread(str(files[index]))
    return _add_texture(image, overlay)


def rotate(image: np.ndarray, angle: int = None) -> np.ndarray:
    """[summary]

    Args:
        image (np.ndarray): image to be rotated
        angle (int, optional): angle to rotate. Defaults to None.

    Returns:
        np.ndarray: rotated image
    """
    if not angle:
        # Randomly choose an angle
        random = np.random.rand()
        # 50% of the time choose an angle between -10 and 10 degrees
        if random < 0.5:
            angle = np.random.rand() * 10 - 10
        elif random < 0.75:
            # 25% of the time 90 degrees
            angle = 90
        else:
            # 25% of the time 180 degrees
            angle = 180
    rotated_img = rotation(image, angle)
    return rotated_img


def blur(image: np.ndarray, sigma_x: int= None, sigma_y: int=None) -> np.ndarray:
    """
    Applies Gussian blur to an image
    Args:
        image (np.ndarray): BGR image
        sigma_x (int, optional): Standard deviation along x-axis.
        When None a value is randomly chosen. Defaults to None.
        sigma_y (int, optional): Standard deviation along y-axis.
        When None a value is randomly chosen. Defaults to None.

    Returns:
        np.ndarray: [description]
    """
    if not sigma_x:
        sigma_x = np.random.randint(50,200)
    if not sigma_y:
        sigma_y = np.random.randint(50,200)
    blurred_image = cv2.GaussianBlur(image, (5, 5), sigma_x, sigma_y)
    return blurred_image


def contrast_and_brighten(
    image: np.ndarray, contrast: float = None, brightness: int = None
) -> np.ndarray:
    """
    Use alpha-beta method to contrast and brightness images
    Args:
        image (np.ndarray): BGR image
        contrast (float, optional): Contrast value (multiplicative factor) to be applied. 
        Good results for values between 0.5 and 3.0. Defaults to None.
        brightness (int, optional): Brightness value to be added. Negative values decrease brightness.
        Good results between -50 to 100. Defaults to None.

    Returns:
        np.ndarray: image with brightness and contrast values altered.
    """
    if not contrast:
        contrast = np.random.rand(1) + 0.5
    if not brightness:
        brightness = np.random.randint(-50, 100)

    # g(x,y) = contrast * f(x,y) + brightness
    new_image = np.clip(image.astype(np.int64) * contrast + brightness, 0, 255)
    new_image = new_image.astype(image.dtype)
    return new_image


def scanner_like(image: np.ndarray) -> np.ndarray:
    """
    Binarizes image and thresholds it to get a photocopier/scanner like look.
    Args:
        image (np.ndarray): BGR image

    Returns:
        np.ndarray: BGR image
    """
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        image_grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    image_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return image_bgr


def gamma_saturation(image: np.ndarray, gamma: float = None) -> np.ndarray:
    """
    Applies gamma saturation to an image.
    Args:
        image (np.ndarray): [BGR image
        gamma (float, optional): Values between 0 and 1 decrease contrast.
        Values greater than 1 increase contrast. Defaults to None.

    Returns:
        np.ndarray: BGR image
    """
    if not gamma:
        if np.random.rand(1) < 0.5:
            # lower saturation
            gamma = np.random.rand(1)
        else:
            # increase saturation
            gamma = np.random.randint(1, 11)
    lookup_table = np.zeros((1, 256), np.uint8)
    for i in range(256):
        # output_intensity = (input_intensity/255)** gamma × 255
        lookup_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    new_img = cv2.LUT(image, lookup_table)
    return new_img
