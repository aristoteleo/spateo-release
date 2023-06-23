import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["pdf.fonttype"] = 42


def get_fb_dis(image_tif, labels_tif):
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "You need to install the package `opencv-python`." "\nInstall via `pip install opencv-python`"
        )
    data = cv2.imread(image_tif, 2)
    labels = cv2.imread(labels_tif, 2)
    cell_sigs = np.bincount(data[labels > 0]) / np.sum(labels > 0)
    bg_sigs = np.bincount(data[labels == 0]) / np.sum(labels == 0)
    # fig, ax = plt.subplots()
    # ax.bar(range(len(cell_sigs)), cell_sigs, label="cell", alpha=0.6, color="red")
    # ax.bar(range(len(bg_sigs)), bg_sigs, label="bg", alpha=0.6, color='blue')
    # ax.set_xlabel("signal intensity")
    # ax.set_ylabel("density")
    # plt.legend()
    # plt.savefig("sig.dis.pdf")
    if len(bg_sigs) < len(cell_sigs):
        bg_sigs = list(bg_sigs)
        bg_sigs.extend([0 for i in range(len(cell_sigs) - len(bg_sigs))])
    if len(bg_sigs) > len(cell_sigs):
        cell_sigs = list(cell_sigs)
        cell_sigs.extend([0 for i in range(len(bg_sigs) - len(cell_sigs))])
    stat_df = pd.DataFrame({"signal": range(len(cell_sigs)), "cell_sigs": cell_sigs, "bg_sigs": bg_sigs}).set_index(
        "signal"
    )
    return stat_df


def cell_area_dis(labels_tifs):
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "You need to install the package `opencv-python`." "\nInstall via `pip install opencv-python`"
        )
    all_areas = []
    for labels_tif in labels_tifs:
        labels = cv2.imread(labels_tif, 2)
        all_areas.extend(list(np.bincount(labels.flatten())[1:]))
    all_areas = np.array(all_areas)
    all_areas = all_areas[all_areas > 0]
    area_dis = np.bincount(all_areas)
    area_df = pd.DataFrame({"area": range(len(area_dis)), "cell_num": area_dis, "prob": area_dis / np.sum(area_dis)})
    # fig, ax = plt.subplots()
    # ax.bar('area', 'cell_num', data=area_df)
    # ax.set_xlabel('area')
    # ax.set_ylabel('cell number')
    # plt.savefig("cell_area.dis.pdf")
    return area_df


def c_to_a_ratio_dis(labels_tif):
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "You need to install the package `opencv-python`." "\nInstall via `pip install opencv-python`"
        )
    labels = cv2.imread(labels_tif, 2)
    cell_labels = [i for i in np.unique(labels) if i > 0]
    ratios = []
    for c in cell_labels:
        one = np.where(labels == c, 1, 0).astype(np.uint8)
        contours, _ = cv2.findContours(one, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        per = cv2.arcLength(contours[0], True)
        area = np.sum(one == 1)
        ratios.append(per / area)
    ratios = np.array(ratios)
    return ratios


def ltos_ratio_dis(labels_tifs):
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "You need to install the package `opencv-python`." "\nInstall via `pip install opencv-python`"
        )
    ratios = []
    for labels_tif in labels_tifs:
        labels = cv2.imread(labels_tif, 2)
        cell_labels = [i for i in np.unique(labels) if i > 0]

        for c in cell_labels:
            one = np.where(labels == c, 1, 0).astype(np.uint8)
            contours, _ = cv2.findContours(one, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            rect = cv2.minAreaRect(contours[0])
            if rect[1][0] * rect[1][1] == 0:
                continue
            ltos = np.max(rect[1]) / np.min(rect[1])
            ratios.append(ltos)
    ratios = np.array(ratios)
    return ratios


def get_fb_dis_window(image_tif, labels_tif, win=200):
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "You need to install the package `opencv-python`." "\nInstall via `pip install opencv-python`"
        )
    data = cv2.imread(image_tif, 2)
    labels = cv2.imread(labels_tif, 2)
    c = 0
    cell_df = None
    bg_df = None
    for i in range(0, data.shape[0], win):
        for j in range(0, data.shape[1], win):
            d = data[i : i + win, j : j + win]
            l = labels[i : i + win, j : j + win]
            cell_sigs = np.bincount(d[l > 0]) / np.sum(l > 0)
            bg_sigs = np.bincount(d[l == 0]) / np.sum(l == 0)
            cell_one_df = pd.DataFrame({"signal": range(len(cell_sigs)), "cell_sigs": cell_sigs}).set_index("signal")
            bg_one_df = pd.DataFrame({"signal": range(len(bg_sigs)), "bg_sigs": bg_sigs}).set_index("signal")

            if c == 0:
                cell_df = cell_one_df
                bg_df = bg_one_df
            if c > 0:
                cell_df = pd.concat((cell_df, cell_one_df), axis=1)
                bg_df = pd.concat((bg_df, bg_one_df), axis=1)
            c += 1
    cell_df = cell_df.fillna(0).T
    bg_df = bg_df.fillna(0).T
    cell_mean_df = pd.DataFrame({"prob": cell_df.mean(axis=0)})
    bg_mean_df = pd.DataFrame({"prob": bg_df.mean(axis=0)})
    return cell_df, bg_df, cell_mean_df, bg_mean_df
