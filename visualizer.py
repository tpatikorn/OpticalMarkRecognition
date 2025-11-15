import os
from typing import Tuple

import cv2
import numpy as np


def visualize_float_grid(image_path: str,
                         float_grid: np.ndarray,
                         output_filename: str,
                         color_bgr: Tuple[int, int, int] = (0, 0, 255)):
    """
    Visualizes the float grid by coloring the corresponding cells in an image with transparent red.

    Args:
        image_path: Path to the base image (e.g., 'debug_output/page_grid.png').
        float_grid: A 2D NumPy array of float values (0.0 to 1.0) or True/False.
        output_filename: The name of the output image file (will be saved in DEBUG_OUTPUT_DIR).
        color_bgr: the color to be painted in bgr
    """
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Create a blank overlay image (same size as img, all zeros)
    # This overlay will store the red rectangles
    overlay = np.zeros_like(img, dtype=np.uint8)

    cell_h = img.shape[0] // float_grid.shape[0]
    cell_w = img.shape[1] // float_grid.shape[1]

    for r in range(float_grid.shape[0]):
        for c in range(float_grid.shape[1]):
            ratio = float_grid[r, c]

            y1 = cell_h * r
            y2 = cell_h * (r + 1)
            x1 = cell_w * c
            x2 = cell_w * (c + 1)

            # BGR format: Blue, Green, Red
            fill_color = (int(np.clip((255 - color_bgr[0]) * ratio, 0, 255)),
                          int(np.clip((255 - color_bgr[1]) * ratio, 0, 255)),
                          int(np.clip((255 - color_bgr[2]) * ratio, 0, 255)))

            # Draw a solid red rectangle on the overlay
            # Fill from (x1+1, y1+1) to (x2-1, y2-1) to avoid overwriting grid lines.
            cv2.rectangle(overlay, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), fill_color, -1)

    # Blend the overlay with the original image
    # Use a constant alpha for the transparency of the red rectangles.
    # Alpha value between 0.0 (fully transparent) and 1.0 (fully opaque).
    # Let's use 0.5 for 50% transparency.
    final_img = cv2.bitwise_not(img)
    final_img = cv2.addWeighted(overlay, 1, final_img, 1, 0)
    final_img = cv2.bitwise_not(final_img)

    output_path = os.path.join(output_filename)
    cv2.imwrite(output_path, final_img)
