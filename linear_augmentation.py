import cv2
import numpy as np
from pathlib import Path
from .affine_transform import rotation
from .utility import _perspective_warp, _add_texture, _generate_shadow_coordinates
from .non_linear_augmentation import distort

PATH_TO_WRINKLED_TEXTURE = Path(
    "fuse/notebooks/exploratory/data/augmentation-helpers/overlays/wrinkle"
)
PATH_TO_MONITOR_TEXTURE = Path(
    "fuse/notebooks/exploratory/data/augmentation-helpers/overlays/monitor"
)
PATH_TO_BG_IMAGES = Path(
    "fuse/notebooks/exploratory/data/augmentation-helpers/background"
)

assert PATH_TO_WRINKLED_TEXTURE.exists()
assert PATH_TO_MONITOR_TEXTURE.exists()
assert PATH_TO_BG_IMAGES.exists()


def add_noise(image, noise_typ=None):
    noise_types = ["gauss", "s&p"]
    if not noise_typ:
        noise_typ = noise_types[np.random.randint(0, len(noise_types))]
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 30
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy.astype(np.uint8)
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out


def add_shadow(image, no_of_shadows=1):
    image_HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    ## Conversion to HLS
    mask = np.zeros_like(image)
    imshape = image.shape
    vertices_list = _generate_shadow_coordinates(imshape, no_of_shadows)
    # 3 getting list of shadow vertices
    for vertices in vertices_list:
        cv2.fillPoly(
            mask, vertices, 255
        )  ## adding all shadow polygons on empty mask, single 255 denotes only red channel
        image_HLS[:, :, 1][mask[:, :, 0] == 255] = (
            image_HLS[:, :, 1][mask[:, :, 0] == 255] * 0.75
        )  ## if red channel is hot, image's "Lightness" channel's brightness is lowered
        image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2BGR)  ## Conversion to RGB
    return image_RGB


def add_virtual_background(image, scale=1.25):
    files = sorted(PATH_TO_BG_IMAGES.glob("*.jpg"))
    index = np.random.randint(0, len(files))
    bg_image = cv2.imread(str(files[index]))

    org_height, org_width = image.shape[:2]
    new_img_height, new_img_width = int(org_height * scale), int(org_width * scale)

    bg_image = _perspective_warp(bg_image)

    bg_image = cv2.resize(bg_image, (new_img_width, new_img_height))

    loc_x = np.random.randint(0, new_img_width - org_width)
    loc_y = np.random.randint(0, new_img_height - org_height)

    bg_image[loc_y : loc_y + org_height, loc_x : loc_x + org_width, :] = image[:, :, :]

    return bg_image


def order_points(pts):
    # order points in the order top-left, top-right, bottom-right
    # bottom-left
    final_rect = np.zeros((4, 2))

    sums = np.sum(pts, axis=1)

    final_rect[0] = pts[np.argmin(sums)]
    final_rect[2] = pts[np.argmax(sums)]

    diff = np.diff(pts, axis=1)

    final_rect[1] = pts[np.argmin(diff)]
    final_rect[3] = pts[np.argmax(diff)]

    return final_rect


def get_perspective_points(image, interactive=False, min_height=None, min_width=None):
    positions = []
    if interactive:
        window_name = "Select perspective points"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(
            window_name, partial(draw_circle, positions=positions, image=image)
        )
        while True:
            cv2.imshow(window_name, image)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
    else:
        # get a rectangle with min_height and min_width
        print(image.shape[1], image.shape[0])
        assert int((image.shape[1] - min_width)) / 2 > 0
        x1 = np.random.randint(0, int((image.shape[1] - min_width) / 2))
        y1 = np.random.randint(0, int((image.shape[0] - min_height) / 2))
        x2 = np.random.randint(min_width, image.shape[1])
        y2 = np.random.randint(min_height, image.shape[0])

        positions = [
            (x1, y1 + np.random.randint(int(-0.1 * min_height), int(0.1 * min_height))),
            (x1, y2),
            (x2, y1),
            (x2 + np.random.randint(int(-0.1 * min_width), int(0.1 * min_width)), y2),
        ]
    return order_points(positions)


def draw_circle(event, x, y, flags, param, positions, image):
    # If event is Left Button Click then store the coordinate in the lists
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
        positions.append([x, y])


def perspective_warp(image, fg_image):
    pts1 = get_perspective_points(
        image, min_height=fg_image.shape[0], min_width=fg_image.shape[1]
    )

    pts2 = order_points(
        np.array(
            [
                [0, 0],
                [fg_image.shape[1], 0],
                [0, fg_image.shape[0]],
                [fg_image.shape[1], fg_image.shape[0]],
            ]
        )
    )
    h, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    warped_image = cv2.warpPerspective(fg_image, h, (image.shape[1], image.shape[0]))
    #     plt.imshow(warped_image[:,:,::-1])
    return warped_image, pts1


def add_virtual_background2(image, scale=1.25):
    files = sorted(PATH_TO_BG_IMAGES.glob("*.jpg"))
    index = np.random.randint(0, len(files))
    bg_image = cv2.imread(str(files[index]))

    org_height, org_width = image.shape[:2]
    new_img_height, new_img_width = int(org_height * scale), int(org_width * scale)
    bg_image = cv2.resize(bg_image, (new_img_width, new_img_height))

    warped_fg_image, pts = perspective_warp(bg_image, image)
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
