import cv2
from functools import partial
import numpy as np


def _add_texture(image, overlay):
    overlay = cv2.resize(overlay, image.shape[:2][::-1])
    texturized_image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    return texturized_image


def _get_perspective_points(image, interactive=False):
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
        (x1, x2), (y1, y2) = np.random.randint(0, image.shape[1], 2), np.random.randint(
            0, image.shape[0], 2
        )
        positions = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    return _order_points(positions)


def _draw_circle(event, x, y, flags, param, positions, image):
    # If event is Left Button Click then store the coordinate in the lists
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
        positions.append([x, y])


def _perspective_warp(image):
    pts1 = _get_perspective_points(image)
    width1 = np.linalg.norm(pts1[0] - pts1[1])
    width2 = np.linalg.norm(pts1[2] - pts1[3])
    new_width = int(max(width1, width2))

    height1 = np.linalg.norm(pts1[0] - pts1[3])
    height2 = np.linalg.norm(pts1[1] - pts1[2])
    new_height = int(max(height1, height2))

    pts2 = _order_points(
        np.array([[0, 0], [new_width, 0], [0, new_height], [new_width, new_height]])
    )
    h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    warped_image = cv2.warpPerspective(image, h, (new_width, new_height))
    return warped_image


def _order_points(pts):
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


def _generate_shadow_coordinates(imshape, no_of_shadows=1):
    vertices_list = []
    for index in range(no_of_shadows):
        vertex = []
        for dimensions in range(
            np.random.randint(3, 5)
        ):  ## Dimensionality of the shadow polygon
            vertex.append(
                (
                    imshape[1] * np.random.uniform(),
                    imshape[0] // 3 + imshape[0] * np.random.uniform(),
                )
            )
        vertices = np.array([vertex], dtype=np.int32)  ## single shadow vertices
        vertices_list.append(vertices)
    return vertices_list  ## List of shadow vertices
