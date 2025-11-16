import os

import cv2
import fitz
import numpy as np

from constants import PDF_RENDER_DPI


def extract_images_from_folder(raw_folder: str, output_folder: str):
    """
    Extract each file in a folder as greyscale images into output_folder.
    If a file is an image, it'll be simply copied to output_folder.
    If a file is a pdf, each page will be copied to output_folder as separate images.
    :param raw_folder: input folder path
    :param output_folder: output folder path
    """
    for filename in os.listdir(raw_folder):
        filepath = os.path.join(raw_folder, filename)
        print(f"Starting OMR processing for: {filepath}")
        is_pdf = filepath.lower().endswith(".pdf")
        if is_pdf:
            images = pdf_to_images(filepath)
        else:
            img = cv2.imread(filepath)
            images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]
        for _index, img in enumerate(images):
            base_filename = os.path.splitext(filename)[0]
            cv2.imwrite(os.path.join(output_folder, f"{base_filename}_{_index:02}.jpg"), img)


def pdf_to_images(filepath):
    images = []
    doc = fitz.open(filepath)
    for page_num in range(len(doc)):
        print(f"--- Processing {filepath} Page {page_num + 1} ---")
        page = doc.load_page(page_num)
        images.append(pdf_page_to_image(page))
    doc.close()
    return images


def pdf_page_to_image(page: fitz.Page) -> np.ndarray:
    """
    Renders a PDF page as a high-resolution grayscale image.

    Args:
        page: A fitz.Page object from PyMuPDF.

    Returns:
        A 2D NumPy array (grayscale image).
    """
    # Render page to a pixmap (image)
    pix = page.get_pixmap(dpi=PDF_RENDER_DPI)

    # Convert pixmap samples to a NumPy array
    # Pix.samples is a 1D bytes object (R,G,B, R,G,B, ...)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    # Convert from RGB to grayscale for processing
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return gray_img
