#!/usr/bin/env python

import time
from typing import Tuple, Union

import cv2
import lack
import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D  # this may need to move to the beginning

slog = lack.LoggerManager(namespace="spateo")


def split_image(
    img: Union[str, np.ndarray],
    window_size: int = 1000,
    overlap_size: int = 200,
    save_fig: bool = False,
    root_dir: str = "./",
) -> Tuple[list, list]:
    """Split a large image into several small images.

    Args:
        img: The input whole image. It should be a one-channel image. File extension must be tif.
        window_size: Split whole image with this size.
        overlap_size: The overlap size of adjacent figures.
        save_fig: If save small figures.
        root_dir: The root directory for the splitted figures to be saved.

    Returns:
        Return a tuple including firstly the list contains all figures and then the list of the corresponding figure
        names.
    """

    figures = []
    figure_names = []
    if isinstance(img, str):
        img = cv2.imread(img, 2)  # tag: cv::IMREAD_ANYDEPTH = 2

    for y in range(0, img.shape[0] - overlap_size, window_size):
        row = []
        row_name = []
        for x in range(0, img.shape[1] - overlap_size, window_size):
            row.append(img[y : y + window_size + overlap_size, x : x + window_size + overlap_size])
            row_name.append(
                f"yrange_{y}-{y + window_size + overlap_size}_xrange_{x}-{x + window_size + overlap_size}.tif"
            )
        figures.append(row)
        figure_names.append(row_name)

    if save_fig:
        for y in range(len(figures)):
            for x in range(len(figures[y])):
                cv2.imwrite(root_dir + figure_names[y][x], figures[y][x])

    return figures, figure_names


def run_stardist(
    img: str,
    window_size: int = 1000,
    overlap_size: int = 200,
    save_fig: bool = False,
    norm_first: bool = False,
    combine_only: bool = False,
    model: str = "2D_versatile_fluo",
    nms_thresh: Union[None, float] = None,
    prob_thresh: Union[None, float] = None,
    pd_kwargs: dict = {},
) -> np.ndarray:
    """Segment cells with staining figure by stardist (https://github.com/stardist/stardist).

    Args:
        img: Directory and file name pointing to the input staining image. np.uint8. File extension must be tif.
        window_size: Sliding window size for splitting `img`.
        overlap_size: Overlap size of two adjacent figures.
        save_fig: Whether or not to save small figures.
        norm_first: Whether to normalize `img` before splitting it into several small figures. If True, each small
            figure will be normalized before running stardist. If False, each small figure won't be normalized in
            advance before running stardist.
        combine_only: Whether to only combine results of all small figures and don't run stardist.
        model: The pretrained model from StarDist2D. There are three available models
            (the name and the alias are listed below:
            1. '2D_versatile_fluo': 'Versatile (fluorescent nuclei)'
            2. '2D_versatile_he':  'Versatile (H&E nuclei)'
            3. '2D_paper_dsb2018': 'DSB 2018 (from StarDist 2D paper)'
        prob_thresh : Consider only object candidates from pixels with predicted object probability
            above this threshold (also see `optimize_thresholds`).
        nms_thresh : Perform non-maximum suppression that considers two objects to be the same
            when their area/surface overlap exceeds this threshold (also see `optimize_thresholds`).
        pd_kwargs: additional arguments passed to the `model.predict_instances` function.

    Returns:
        A np.ndarray (np.int32) indicates cell labels, with background (non-cell) as zero.
    """

    if norm_first:
        # tag: cv::IMREAD_GRAYSCALE = 0
        img = normalize(cv2.imread(img, 0))  # input img: np.uint8; output img: np.float32

    figures, figure_names = split_image(img=img, window_size=window_size, overlap_size=overlap_size, save_fig=save_fig)
    figure_names = [[j.replace(".tif", ".stardist.tif") for j in i] for i in figure_names]

    if pd_kwargs is not None:
        pd_kwargs_dict = {
            "axes": None,
            "normalizer": None,
            "sparse": True,
            "prob_thresh": None,
            "nms_thresh": None,
            "scale": None,
            "n_tiles": None,
            "show_tile_progress": True,
            "verbose": False,
            "predict_kwargs": None,
            "nms_kwargs": None,
            "overlap_label": None,
        }
        pd_kwargs_dict.update(pd_kwargs)

    if not combine_only:
        model = StarDist2D.from_pretrained(model)
        for y in range(len(figures)):
            for x in range(len(figures[y])):
                if not norm_first:
                    figures[y][x], _ = model.predict_instances(
                        normalize(figures[y][x]), nms_thresh=pd_kwargs_dict, prob_thresh=pd_kwargs_dict, **pd_kwargs
                    )
                else:
                    figures[y][x], _ = model.predict_instances(
                        figures[y][x], nms_thresh=nms_thresh, prob_thresh=prob_thresh, **pd_kwargs
                    )

                # labels.dtype int32
                if save_fig:
                    cv2.imwrite(figure_names[y][x], figures[y][x].astype(np.float32))

    slog.main_info(time.asctime(time.localtime(time.time())))
    slog.main_info("finish stardist")
    # print(time.asctime(time.localtime(time.time())))
    # print("finish stardist", flush=True)

    if combine_only:
        com = combine_image(figures=figures, overlap_size=overlap_size, figure_names=figure_names)
    else:
        com = combine_image(figures=figures, overlap_size=overlap_size, figure_names=None)

    slog.main_info(time.asctime(time.localtime(time.time())))
    slog.main_info("finish combining image")
    # print(time.asctime(time.localtime(time.time())))
    # print("finish combine", flush=True)

    return com


def combine_image(
    figures: list,
    overlap_size: int,
    figure_names: list = None,
    root_dir: str = "./",
) -> np.ndarray:
    """Combine all small images into a whole image considering overlappings of adjacent images.

    Args:
        figures: A 2-dimensional list contains np.array (np.int32) of all small images.
        overlap_size: Overlap size of two adjacent images.
        figure_names: A 2-dimensional list contains file names of all small images.
        root_dir: The root directory for the splitted figures to be saved.

    Returns:
        Combined array, np.int32.
    """

    if figure_names:
        for y in range(len(figure_names)):
            for x in range(len(figure_names[y])):
                figures[y][x] = cv2.imread(root_dir + figure_names[y][x], 2).astype(np.int32)

    _rename_cell_labels(figures)

    rows = []
    for row in figures:
        if len(row) == 1:
            rows.append(row[0])
        else:
            com = _combine_h(row[0], row[1], overlap_size)
            for i in range(2, len(row)):
                com = _combine_h(com, row[i], overlap_size)
            rows.append(com)

        # print(time.asctime(time.localtime(time.time())))
        # print("finish combine a left right", flush=True)

    if len(rows) == 1:
        return rows[0]
    else:
        com = _combine_v(rows[0], rows[1], overlap_size)
        for i in range(2, len(rows)):
            com = _combine_v(com, rows[i], overlap_size)
            # print(time.asctime(time.localtime(time.time())))
            # print("finish combine a up down", flush=True)
        return com


def _rename_cell_labels(figures: list):
    """Rename cell labels to make sure a label only exists in one figure.

    Args:
        figures: All small figures, each figure is np.int32.

    Returns:
        None. Change value in-place.
    """

    max_labels = 0
    for row in figures:
        for one in row:
            one[:] = np.where(one > 0, one + max_labels, one)
            max_labels = max(np.max(one), max_labels)


def _combine_v(up: np.ndarray, down: np.ndarray, overlap_size: int) -> np.ndarray:
    """Combine two np.ndarray vertically.

    Args:
        up: The up array, np.int32.
        down: The down array, np.int32.
        overlap_size: Overlap size of two in shape[0]

    Returns:
        Combined array, np.int32.
    """

    # 1. process up window
    up_to_win = up[0 : up.shape[0] - min(overlap_size, down.shape[0]), :]  # get up window
    up_cell_ids = np.unique(up_to_win)  # retrieve up window labels
    up_win_to_bot = up[up.shape[0] - min(overlap_size, down.shape[0]) :, :]  # get up overlapping region
    # if overlap shares label with up window, assign to the label from up window
    up_win_to_bot = np.where(np.isin(up_win_to_bot, up_cell_ids), up_win_to_bot, 0)
    up = np.vstack((up_to_win, up_win_to_bot))  # combine up window and overlapping region

    # 2. process padding near the overlapping region
    pad = min(down.shape[0], 10)
    line_up = up[
        up.shape[0] - min(overlap_size, down.shape[0]) : up.shape[0] - min(overlap_size, down.shape[0]) + pad, :
    ]
    line_down = down[0:pad, :]
    line_ov = np.where(line_up > 0, line_down, 0)  # set the labels of the padding to the bottom window.

    # 3. process down window (if the labels are assigned in the up window, ignore the ones from bottom window)
    down_cell_remove_ids = np.unique(line_ov)
    down = np.where(np.isin(down, down_cell_remove_ids), 0, down)  # remove labelings for the down region

    # 4. create a full up/bottom window with overlapping
    up_exp = np.vstack((up, np.zeros([max(0, down.shape[0] - overlap_size), up.shape[1]], dtype=np.int32)))
    down_exp = np.vstack(
        (np.zeros([max(up.shape[0] - down.shape[0], up.shape[0] - overlap_size), down.shape[1]], dtype=np.int32), down)
    )

    # 5. combine
    com = np.where(up_exp > 0, up_exp, down_exp)

    return com


def _combine_h(left: np.ndarray, right: np.ndarray, overlap_size: int) -> np.ndarray:
    """Combine two np.ndarray horizontally.

    Args:
     left: The left array, np.int32.
     right: The right array, np.int32.
     overlap_size: Overlap size of two in shape[1]

    Returns:
        Combined array, np.int32.
    """

    left_cell_ids = np.unique(left[:, 0 : left.shape[1] - min(overlap_size, right.shape[1])])
    left = np.where(np.isin(left, left_cell_ids), left, 0)

    pad = min(right.shape[1], 10)
    line_left = left[
        :, left.shape[1] - min(overlap_size, right.shape[1]) : left.shape[1] - min(overlap_size, right.shape[1]) + pad
    ]
    line_right = right[:, 0:pad]
    line_ov = np.where(line_left > 0, line_right, 0)
    right_cell_remove_ids = np.unique(line_ov)
    right = np.where(np.isin(right, right_cell_remove_ids), 0, right)

    left_exp = np.hstack((left, np.zeros([left.shape[0], max(0, right.shape[1] - overlap_size)], dtype=np.int32)))
    right_exp = np.hstack(
        (
            np.zeros(
                [left.shape[0], max(left.shape[1] - right.shape[1], left.shape[1] - overlap_size)], dtype=np.int32
            ),
            right,
        )
    )

    com = np.where(left_exp > 0, left_exp, right_exp)

    return com


def main():
    # print(time.asctime(time.localtime(time.time())))
    # print("start",flush=True)
    # figures, figure_names = split_image(img="/jdfssz1/ST_SUPERCELLS/P20Z10200N0059/panhailin/identifyCell/PD_epilepsy/st/control1/FP200000616TL_C1_web_0/FP200000616TL_C1.ssDNA.PS.xrange_63-19308_yrange_3608-18668.tif",window_size=1000)
    # print(time.asctime(time.localtime(time.time())))
    # print("finish split image",flush=True)
    # com = run_stardist(figures, figure_names, overlap_size=200, save_fig=False, combine_only=True)
    # cv2.imwrite("stardist.combined.tif", com.astype(np.float32))

    com = run_stardist(
        img="/jdfssz1/ST_SUPERCELLS/P20Z10200N0059/panhailin/identifyCell/PD_epilepsy/st/control1/FP200000616TL_C1_web_0/FP200000616TL_C1.ssDNA.PS.xrange_63-19308_yrange_3608-18668.tif",
        window_size=1000,
        overlap_size=200,
        save_fig=True,
        norm_first=True,
    )
    # com = run_stardist(img="/jdfssz1/ST_SUPERCELLS/P20Z10200N0059/panhailin/identifyCell/PD_epilepsy/st/control1/FP200000616TL_C1_web_0/FP200000616TL_C1.ssDNA.PS.xrange_63-19308_yrange_3608-18668.tif",window_size=1000, overlap_size=200, save_fig=False, combine_only=True)
    cv2.imwrite("stardist.combined.tif", com.astype(np.float32))


if __name__ == "__main__":
    main()
