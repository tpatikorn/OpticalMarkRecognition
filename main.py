import os

from constants import *
from omr import AnswerSheet, AnswerSheetTemplate
from omr import CellSpan
from omr_util import extract_images_from_folder
from visualizer import visualize_float_grid

# Process the PDF
# omr_results = process_omr(PDF_FILE, mark_threshold=0.1)
image_folder = "images"
pages_folder = "pages"
GRID_ROWS = 51
GRID_COLS = 33

os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
os.makedirs(pages_folder, exist_ok=True)

extract_images_from_folder(image_folder, pages_folder)
results = []

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
                 span_type="choose_one", answer_keys = [_i % 6 for _i in range(30)])],
    correct_score=1,
    incorrect_score=-0.2)

for image_filename in os.listdir(pages_folder):
    sheet_img = AnswerSheet(
        template=sheet_template,
        filepath=os.path.join(pages_folder, image_filename),
        ignore_border_px=4,
        mark_threshold=0.1)
    results.append(sheet_img)

result_0 = results[0]

# Demonstrate the function with the first page's results
visualize_float_grid(os.path.join(DEBUG_OUTPUT_DIR, f"page_1_aligned.png"),
                     result_0.float_grid,
                     os.path.join(DEBUG_OUTPUT_DIR, f"page_1_vis.png"), )

result_file = "debug_output/page_grid_output.png"
print(result_0.student_id)
print(result_0.all_answers)
print(result_0.all_scores)
