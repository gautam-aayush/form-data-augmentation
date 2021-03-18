import cv2
from functools import partial
import numpy as np
from typing import List, Tuple


def _add_texture(image: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """
    Overlays overlay on top of image to get the texture of overlay on the image
    Args:
        image (np.ndarray): original image
        overlay (np.ndarray): overlay image

    Returns:
        np.ndarray: image with overlay added
    """
    alpha = 0.7
    beta = 1 - alpha
    height, width = image.shape[:2]
    # resize overlay to the size of original image
    overlay = cv2.resize(overlay, (width, height))
    texturized_image = cv2.addWeighted(image, alpha, overlay, beta, 0)
    return texturized_image


def _get_perspective_points(
    image: np.ndarray,
    min_height: int = None,
    min_width: int = None,
    interactive: bool = False,
) -> np.ndarray:
    """
    Gets perspective points from an image for perspective warp. When interactive is True
    an interactive window allows selection of points from the image,
    otherwise a random perspective rectangle with min_height and min_width are selected
    Args:
        image (np.ndarray): BGR image on which perspective points are to be selected
        interactive (bool, optional): When True, an interactive window allows
        manual selection of perspective points. Defaults to False.
        min_height (int, optional): minimum height of the perspective trapezoid.
        Required when interactive is False. Defaults to None.
        min_width (int, optional): minimum width of the perspective trapezoid.
        Required when interactive is False. Defaults to None.

    Returns:
        np.ndarray: Array of perspective points
    """
    positions = []
    if interactive:
        window_name = "Select perspective points"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(
            window_name, partial(_draw_circle, positions=positions, image=image)
        )
        while True:
            cv2.imshow(window_name, image)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
    else:
        # get a rectangle with min_height and min_width
        image_height, image_width = image.shape[:2]
        assert int((image.shape[1] - min_width)) / 2 > 0

        # random rectangular portion of the image, with min_height and min_width
        x1 = np.random.randint(0, (image_width - min_width) // 2)
        y1 = np.random.randint(0, (image_height - min_height) // 2)
        x2 = np.random.randint(x1 + min_width, image_width)
        y2 = np.random.randint(y1 + min_height, image_height)

        # change in height and width from the default rectangle
        delta = 0.1
        delta_height = np.random.randint(-delta * min_height, delta * min_height)
        delta_width = np.random.randint(-delta * min_width, delta * min_width)

        positions = [
            (x1, y1 + delta_height),
            (x1, y2),
            (x2, y1),
            (x2 + delta_width, y2),
        ]
    return _order_points(positions)


def _perspective_warp(
    bg_image: np.ndarray, fg_image: np.ndarray, interactive: bool = False
) -> np.ndarray:
    """
    Perspective warp foreground image on the perspective of background image
    Args:
        bg_image (np.ndarray): [description]
        fg_image (np.ndarray): [description]
        interactive (bool, optional): Defaults to False.

    Returns:
        np.ndarray: [description]
    """
    pts1 = _get_perspective_points(
        bg_image,
        min_height=fg_image.shape[0],
        min_width=fg_image.shape[1],
        interactive=interactive,
    )

    pts2 = _order_points(
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
    warped_image = cv2.warpPerspective(
        fg_image, h, (bg_image.shape[1], bg_image.shape[0])
    )
    return warped_image, pts1


def _draw_circle(
    event: int, x: int, y: int, flags: int, param, positions: List, image: np.ndarray
):
    """
    Callback function to draw a circle on the given image, when event is triggered
    Args:
        event (int): cv2 event
        x (int): x coordinate
        y (int): y coordinate
        flags (int): [description]
        param ([type]): [description]
        positions (List): empty list as a reference
        image (np.ndarray): BGR image
    """
    # If event is Left Button Click then store the coordinate in the lists
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
        positions.append([x, y])


def _order_points(pts: List) -> np.ndarray:
    """
    Order the points of a rectangle in top-left, top-right, bottom-right
    and bottom-left order
    Args:
        pts (List): List of points to be ordered

    Returns:
        np.ndarray: Array of ordered points
    """
    final_rect = np.zeros((4, 2))

    sums = np.sum(pts, axis=1)

    # top left
    final_rect[0] = pts[np.argmin(sums)]
    # bottom right
    final_rect[2] = pts[np.argmax(sums)]

    diff = np.diff(pts, axis=1)

    # top right
    final_rect[1] = pts[np.argmin(diff)]
    # bottom left
    final_rect[3] = pts[np.argmax(diff)]

    return final_rect


def _generate_shadow_coordinates(
    imshape: Tuple[int], no_of_shadows: int = 1
) -> List[np.ndarray]:
    """
    Generates 2D coordinates for a polygon of random dimensionality
    where the value of the coordinates are limited by imshape
    Args:
        imshape (Tuple[int]): maximum values for x and y coordinate
        no_of_shadows (int, optional): Number of polygons to generate.
        Defaults to 1.

    Returns:
        List[np.ndarray]: List of polygon coordiantes
    """
    vertices_list = []
    x_lim, y_lim = imshape
    for index in range(no_of_shadows):
        vertex = []
        min_vertices, max_vertices = 3, 5
        for dimensions in range(np.random.randint(min_vertices, max_vertices)):
            vertex.append(
                (y_lim * np.random.uniform(), x_lim // 3 + x_lim * np.random.uniform(),)
            )
        # polygon vertices
        vertices = np.array([vertex], dtype=np.int32)
        vertices_list.append(vertices)
    return vertices_list
