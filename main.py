import os

from constants import *
from omr import AnswerSheet, AnswerSheetTemplate
from omr import CellSpan
from omr_util import extract_images_from_folder

# Process the PDF
# omr_results = process_omr(PDF_FILE, mark_threshold=0.1)
image_folder = "images"
pages_folder = "pages"
GRID_ROWS = 51
GRID_COLS = 33

sheet_template = AnswerSheetTemplate(
    grid_cols=GRID_COLS,
    grid_rows=GRID_ROWS,
    id_span=CellSpan((7, 2), (16, 13), True, span_type="id"),
    answer_spans=[
        CellSpan((20, 2), (49, 7), False,
                 span_type="choose_one", answer_keys=[_i % 6 for _i in range(30)]),
        CellSpan((20, 10), (49, 15), False,
                 span_type="choose_one", answer_keys=[_i % 6 for _i in range(30)]),
        CellSpan((20, 18), (49, 23), False,
                 span_type="choose_any",
                 answer_keys=[[_i % 6, (_i + 1) % 6, (_i + 4) % 6] for _i in range(30)]),
        CellSpan((20, 26), (49, 31), False,
                 span_type="choose_one", answer_keys=[_i % 6 for _i in range(30)])])


def grade_these(template, pages, ignore_border_px=4, mark_threshold=0.1, highlight_folder=None):
    graded = []
    for image_filename in os.listdir(pages):
        print("grading: ", image_filename)
        sheet_img = AnswerSheet(
            template=template,
            filepath=os.path.join(pages, image_filename),
            highlight_folder=highlight_folder,
            ignore_border_px=ignore_border_px,
            mark_threshold=mark_threshold)
        graded.append(sheet_img)
    return graded


if __name__ == "__main__":
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
    os.makedirs(pages_folder, exist_ok=True)

    extract_images_from_folder(image_folder, pages_folder)

    results = grade_these(sheet_template, pages_folder, 4, 0.1)
    for result in results:
        print(result.filepath, result.student_id, result.all_scores)
