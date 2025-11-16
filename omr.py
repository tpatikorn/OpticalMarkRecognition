import os
from typing import Union, List, Tuple, Literal, Any

import cv2
import numpy as np
from numpy import ndarray

from constants import DEBUG_OUTPUT_DIR, DEFAULT_MARK_THRESHOLD, GREEN, DARK_GREEN, RED
from visualizer import visualize_float_grid


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 corner points in top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left has the smallest sum (x+y)
    # Bottom-right has the largest sum (x+y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has the smallest difference (x-y)
    # Bottom-left has the largest difference (x-y)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def apply_threshold(float_grid: np.ndarray, threshold: float) -> np.ndarray:
    """
    Converts the float grid to a boolean grid based on a threshold.

    Args:
        float_grid: The (ROWS, COLS) array of float values.
        threshold: The cutoff value (e.g., 0.25).

    Returns:
        A (ROWS, COLS) NumPy array of booleans.
    """
    # This is a fast, vectorized NumPy operation
    boolean_grid = float_grid > threshold
    return boolean_grid


class Grading:
    def __init__(self,
                 choose_one_correct=1,
                 choose_one_incorrect=0,
                 choose_any_correct=1,
                 choose_any_per_incorrect_choice=-0.2,
                 choose_any_per_missing_choice=-0.2,
                 min_score=0):
        self.choose_one_correct = choose_one_correct
        self.choose_one_incorrect = choose_one_incorrect
        self.choose_any_correct = choose_any_correct
        self.choose_any_per_incorrect_choice = choose_any_per_incorrect_choice
        self.choose_any_per_missing_choice = choose_any_per_missing_choice
        self.min_score = min_score

    def get_choose_any_score(self, num_incorrect, num_missing):
        return max((self.choose_any_correct +
                    (num_incorrect * self.choose_any_per_incorrect_choice) +
                    (num_missing * self.choose_any_per_missing_choice)), self.min_score)


class CellSpan:
    def __init__(self,
                 top_left: Tuple[int, int],
                 bot_right: Tuple[int, int],
                 vertical: bool,
                 span_type: Literal["id", "choose_one", "choose_any"],
                 answer_keys: List[int | List[int]] = None,
                 grading: Grading = None):
        self.top_left = top_left
        self.bot_right = bot_right
        self.vertical = vertical
        self.span_type = span_type
        self.answer_keys = answer_keys
        if grading:
            self.grading = grading
        else:
            self.grading = Grading()


class AnswerSheetTemplate:
    def __init__(self,
                 grid_rows: int,
                 grid_cols: int,
                 id_span: CellSpan,
                 answer_spans: List[CellSpan]):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.id_span = id_span
        self.answer_spans = answer_spans


class AnswerSheet(AnswerSheetTemplate):
    def __init__(self, filepath: str,
                 template: AnswerSheetTemplate,
                 highlight_folder=DEBUG_OUTPUT_DIR,
                 new_cell_size: int = 20,
                 ignore_border_px: int = 4,
                 mark_threshold: float = 0.1):
        # AnswerSheetTemplate
        super().__init__(template.grid_rows, template.grid_cols, template.id_span, template.answer_spans)
        self.template = template

        # raw stuff and param for processing it
        self.filepath = filepath
        self.output_path = os.path.join(highlight_folder, os.path.basename(filepath))
        self.new_cell_size = new_cell_size
        self.ignore_border_px = ignore_border_px
        self.mark_threshold = mark_threshold

        # output/intermediate steps
        self.raw_img = None
        self.aligned_img = None
        self.float_grid = None

        # ultra processed stuff
        self.student_id = None
        self.all_answers = []
        self.all_scores = 0
        self.process()

    def process(self):
        self.raw_img = cv2.imread(self.filepath, cv2.IMREAD_GRAYSCALE)
        self.aligned_img = self.align_image()
        cv2.imwrite(self.output_path, self.aligned_img)  # make aligned ver. as a base highlighted version
        self.float_grid = self.analyze_grid(self.ignore_border_px)

        self.student_id = self.extract_student_id()
        self.all_answers = self.extract_all_answers()

    @property
    def new_width(self):
        return self.grid_cols * self.new_cell_size

    @property
    def new_height(self):
        return self.grid_rows * self.new_cell_size

    def align_image(self) -> Union[ndarray, None]:
        """
        Finds the largest rectangle (the outer border) in the image and performs
        a perspective warp to get a top-down, aligned view.

        Returns:
            A new NumPy array (grayscale image) that is aligned and cropped,
            or None if no 4-point border is found.
        """
        # 1. Pre-process the image
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.raw_img, (5, 5), 0)

        # Use adaptive thresholding to get a clean binary image
        # THRESH_BINARY_INV: Inverts the image (border becomes white)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # 2. Find contours
        # RETR_EXTERNAL: Only find the outermost contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            print("Error: No contours found.")
            return None

        # Sort contours by area in descending order and keep the largest one
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        outer_border_contour = contours[0]

        # 3. Approximate the contour
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(outer_border_contour, True)
        approx = cv2.approxPolyDP(outer_border_contour, 0.02 * perimeter, True)

        # 4. Check if it's a rectangle
        if len(approx) == 4:
            # We found our 4-corner border
            corners = approx.reshape(4, 2)
            ordered_corners = order_points(corners)

            # Define the 4 destination points for the warp
            # This matches the (width, height) we set in CONFIG
            dst_pts = np.array([
                [0, 0],  # Top-left
                [self.new_width - 1, 0],  # Top-right
                [self.new_width - 1, self.new_height - 1],  # Bottom-right
                [0, self.new_height - 1]  # Bottom-left
            ], dtype="float32")

            # 5. Perform the perspective warp
            # Get the transformation matrix
            _m = cv2.getPerspectiveTransform(ordered_corners.astype("float32"), dst_pts)

            # Apply the matrix to the *original grayscale image*
            self.aligned_img = cv2.warpPerspective(self.raw_img, _m, (self.new_width, self.new_height))
            return self.aligned_img
        else:
            print(f"Error: Found contour with {len(approx)} points, not 4. Cannot align.")
            return None

    def analyze_grid(self, ignore_border_px: int) -> np.ndarray:
        """
        Converts the aligned image into a 2D array of float values (0.0 - 1.0)
        representing the "filled-ness" of each cell.
        Also draws the grid on the aligned image and saves it.

        Returns:
            A (ROWS, COLS) NumPy array of floats.
        """
        # Create an empty grid to store the float values
        float_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=float)

        # Create a color version of the aligned image to draw the grid on
        grid_debug_image = cv2.cvtColor(self.aligned_img, cv2.COLOR_GRAY2BGR)

        # 1. Invert the cell: Marks become white (255), paper becomes black (0)
        image_inv = cv2.bitwise_not(self.aligned_img)
        _, image_thresh = cv2.threshold(
            image_inv, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # --- Robust "Filled-ness" Calculation ---

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                # Calculate the boundaries for the current cell
                y1 = self.new_cell_size * r
                y2 = self.new_cell_size * (r + 1)
                x1 = self.new_cell_size * c
                x2 = self.new_cell_size * (c + 1)

                # Extract the cell from the image
                cell = image_thresh[y1 + ignore_border_px: y2 - ignore_border_px,
                x1 + ignore_border_px: x2 - ignore_border_px]

                # 3. Calculate the ratio
                # Total pixels in the cell
                total_pixels = cell.size
                # Count the "mark" pixels (non-zero pixels)
                filled_pixels = cv2.countNonZero(cell)

                # The filled ratio is our float value
                filled_ratio = filled_pixels / total_pixels

                float_grid[r, c] = filled_ratio

                # Draw grid lines on the debug image
                # Draw horizontal lines
                cv2.line(grid_debug_image, (x1, y1), (x2, y1), GREEN, 1)  # Green line
                # Draw vertical lines
                cv2.line(grid_debug_image, (x1, y1), (x1, y2), GREEN, 1)  # Green line

        # Draw the rightmost and bottommost lines to complete the grid
        cv2.line(
            grid_debug_image,
            (0, self.new_height - 1),
            (self.new_width - 1, self.new_height - 1),
            (0, 255, 0),
            1)
        cv2.line(
            grid_debug_image,
            (self.new_width - 1, 0),
            (self.new_width - 1, self.new_height - 1),
            (0, 255, 0),
            1)

        # Save the grid debug image
        grid_save_path = os.path.join(DEBUG_OUTPUT_DIR, f"page_grid.png")
        cv2.imwrite(grid_save_path, grid_debug_image)

        return float_grid

    def extract_argmax(self, span: CellSpan) -> tuple[List[int], int]:
        values = self.float_grid[span.top_left[0]: span.bot_right[0] + 1, span.top_left[1]: span.bot_right[1] + 1]
        # find the indices of True and return all such indices
        argmax_axis = 0 if span.vertical else 1
        output = np.argmax(values, axis=argmax_axis)
        scores = 0

        correct_grid = np.zeros(self.float_grid.shape, dtype=np.uint8)
        incorrect_grid = np.zeros(self.float_grid.shape, dtype=np.uint8)
        for _index, _value in enumerate(output):
            if span.vertical:
                _row = span.top_left[0] + _value
                _col = span.top_left[1] + _index
            else:
                _row = span.top_left[0] + _index
                _col = span.top_left[1] + _value
            if span.answer_keys and (_value == span.answer_keys[_index]):
                correct_grid[_row, _col] = 1
                scores += span.grading.choose_one_correct
            else:
                incorrect_grid[_row, _col] = 1
                scores += span.grading.choose_one_incorrect
        visualize_float_grid(self.output_path, correct_grid, self.output_path, color_bgr=DARK_GREEN)
        visualize_float_grid(self.output_path, incorrect_grid, self.output_path, color_bgr=RED)
        return output.tolist(), round(scores, 2)

    def extract_threshold(self, span: CellSpan, threshold: float = None) -> Tuple[List[List[int]], int]:
        values = self.float_grid[span.top_left[0]: span.bot_right[0] + 1, span.top_left[1]: span.bot_right[1] + 1]

        if threshold is None:
            threshold = DEFAULT_MARK_THRESHOLD
        # for each row, return a list of indices of elt that >= threshold
        indices = []
        for row in values:
            indices.append([i for i, val in enumerate(row) if val >= threshold])

        scores = 0

        correct_grid = np.zeros(self.float_grid.shape, dtype=np.uint8)
        incorrect_grid = np.zeros(self.float_grid.shape, dtype=np.uint8)
        unanswered_grid = np.zeros(self.float_grid.shape, dtype=np.uint8)
        for _index, _values in enumerate(indices):
            _answered = set(_values)
            _correct = set(span.answer_keys[_index])
            union_answer = _answered.union(_correct)
            intersect_answer = _answered.intersection(_correct)
            incorrect = _answered - _correct
            unanswered = _correct - _answered
            scores += span.grading.get_choose_any_score(len(incorrect), len(unanswered))
            for _value in union_answer:
                if span.vertical:
                    _row = span.top_left[0] + _value
                    _col = span.top_left[1] + _index
                else:
                    _row = span.top_left[0] + _index
                    _col = span.top_left[1] + _value
                if _value in intersect_answer:
                    correct_grid[_row, _col] = 1
                elif _value in incorrect:
                    incorrect_grid[_row, _col] = 1
                elif _value in unanswered:
                    unanswered_grid[_row, _col] = 1
        visualize_float_grid(self.output_path, correct_grid, self.output_path, color_bgr=DARK_GREEN)
        visualize_float_grid(self.output_path, incorrect_grid, self.output_path, color_bgr=RED)
        visualize_float_grid(self.output_path, unanswered_grid, self.output_path, color_bgr=RED, thickness=2)
        return indices, round(scores, 2)

    def extract_student_id(self) -> str:
        choices = self.extract_argmax(self.id_span)[0]
        return "".join([str(_) for _ in choices])

    def extract_choices(self, span: CellSpan, threshold=None) -> Tuple[List[int] | List[List[int]], int]:
        match span.span_type:
            case "choose_any":
                # for each row, return a list of indices of elt that >= threshold
                return self.extract_threshold(span, threshold)
            case "choose_one" | _:  # when all else fail, fallback to choose one
                return self.extract_argmax(span)

    def extract_all_answers(self, threshold=None) -> tuple[list[Any], int]:
        """
        extract all answers as one big List and set it to self.all_answers.
        for each elt:
        - case span_type == choose_one, that elt type will be int
        - case span_type == choose_any, that elt type will be List[int]
        :param threshold:  only for "choose_any" span type
        :return: self.all_answers
        """
        self.all_answers = []
        self.all_scores = 0
        for answer_span in self.answer_spans:
            this_answer = self.extract_choices(answer_span, threshold)
            self.all_answers += this_answer[0]
            self.all_scores += this_answer[1]
        self.all_scores = round(self.all_scores, 2)
        return self.all_answers, self.all_scores
