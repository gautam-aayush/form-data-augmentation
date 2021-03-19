from basic_transform import (lcd_overlay, noise, rotate, virtual_background,
                             wrinkles)
from distortion import distort, perspective, stretch


def rotation_with_lcd(image):
    rotated_img = rotate(image)
    final_img = lcd_overlay(rotated_img)
    return final_img


def wrinkle_with_noise(image):
    wrinkled_img = wrinkles(image)
    final_img = noise(wrinkled_img)
    return final_img


def background_with_lcd_stretch(image):
    img_with_bg = virtual_background(image)
    img_with_overlay = lcd_overlay(img_with_bg)
    final_img = stretch(img_with_overlay)
    return final_img
