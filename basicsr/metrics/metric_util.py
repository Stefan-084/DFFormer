import numpy as np


def reorder_image(img, input_order="HWC"):
    if input_order not in ["HWC", "CHW"]:
        raise ValueError(f"Wrong input_order {input_order}. Supported input orders are 'HWC' and 'CHW'.")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == "CHW":
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    img = img.astype(np.float32)
    if img.max() <= 1:
        img = img * 255.0
    if img.ndim == 3 and img.shape[2] == 3:
        # Input is BGR here, matching the historical BasicSR behavior.
        y = 16.0 + (24.966 * img[..., 0] + 128.553 * img[..., 1] + 65.481 * img[..., 2]) / 255.0
        img = y[..., None]
    return img
