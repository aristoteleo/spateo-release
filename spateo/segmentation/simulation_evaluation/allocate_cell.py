# from cProfile import label
import os
import pickle

# from select import select
import cv2
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

# image = np.zeros([15,15])
# cv2.ellipse(img=image, center=(5,14), axes=(4,2), angle=0.0, startAngle=0, endAngle=360, color=1, thickness=-1)
# print(image)


class Cell:
    def __init__(self, center, axes, color, angle):
        self.center = center
        self.axes = axes
        self.color = color
        self.angle = angle

    def set_center(self, center):
        self.center = center


def get_cell_pos(area_df, ltos, cell_num=100, height=500, width=500, seed=1, max_iter=20000, shift_length=100):
    labels = np.zeros([height, width], dtype=np.uint16)

    areas = select_area(area_df, cell_num, seed)
    # ctoas = select_ctoa(c_to_a_ratios, cell_num, seed)
    # axes = get_axes_from_area_and_ctoa(areas, ctoas, seed)
    axes = get_axes_from_area_and_ltos(areas, ltos, seed)
    centers = get_center(height, width, cell_num, seed)
    colors = [i for i in range(1, cell_num + 1)]
    np.random.seed(seed)
    angles = np.random.rand(cell_num) * 360

    cells = []
    for i in range(len(colors)):
        cells.append(Cell(centers[i], axes[i], colors[i], angles[i]))

    shift_cells(cells, labels, max_iter, seed, shift_length)

    return labels


def shift_cells(cells, labels, max_iter, seed, shift_length=10):
    cv2.ellipse(
        img=labels,
        center=cells[0].center,
        axes=cells[0].axes,
        color=cells[0].color,
        angle=0.0,
        startAngle=0,
        endAngle=360,
        thickness=-1,
    )
    deal_list = cells[1:]
    c = 0
    np.random.seed(seed)
    center_shifts = np.random.randint(-shift_length, shift_length + 1, 2 * max_iter + 2).reshape(-1, 2)

    while deal_list:

        c += 1
        one = deal_list.pop(0)
        labels_tmp = labels.copy()
        cv2.ellipse(
            img=labels_tmp,
            center=one.center,
            axes=one.axes,
            color=one.color,
            angle=one.angle,
            startAngle=0,
            endAngle=360,
            thickness=-1,
        )
        if (labels[labels_tmp == one.color] > 0).any():
            tmp = np.array(one.center) - center_shifts[c]
            tmp[tmp < 0] = 0
            tmp[0] = np.min([labels.shape[1], tmp[0]])
            tmp[1] = np.min([labels.shape[0], tmp[1]])
            one.set_center(tuple(tmp))
            deal_list.append(one)
        else:
            labels[:] = labels_tmp

        if c >= max_iter:
            print("max iteration has reached, pleas check the result.")
            deal_list = []


def get_center(height, width, cell_num, seed):
    import numpy as np

    np.random.seed(seed)
    heights = np.random.randint(height, size=cell_num)
    widths = np.random.randint(width, size=cell_num)
    return list(zip(heights, widths))


def select_area(area_df, cell_num, seed):
    np.random.seed(seed)
    area_df = area_df[area_df["prob"] > 0]
    areas = np.array([row["area"] for index, row in area_df.iterrows() for i in range(int(row["cell_num"]))])
    while len(areas) < cell_num:
        areas = np.tile(areas, 2)

    np.random.shuffle(areas)
    areas = areas[0:cell_num]
    return areas


def select_ctoa(c_to_a_ratios, cell_num, seed):
    while cell_num > len(c_to_a_ratios):
        c_to_a_ratios = np.tile(c_to_a_ratios, 2)
    np.random.seed(seed)
    np.random.shuffle(c_to_a_ratios)
    c_to_a_ratios = c_to_a_ratios[0:cell_num]
    return c_to_a_ratios


def get_axes_from_area_and_ctoa(areas, ctoas, seed):
    # S=pi*a*b
    # C=2pib + 4(a-b)
    # R = C/S
    # x = RS
    # y = S/pi
    # long = np.sqrt(y-np.pi*y/2+x/4)
    # short = y/longs

    x = ctoas * areas
    y = areas / np.pi

    longs = np.sqrt(y - np.pi * y / 2 + x / 4)
    shorts = y / longs
    axes = list(zip(longs, shorts))
    return axes


def get_axes_from_area_and_ltos(areas, ltos, seed):
    # S = pi*a*b
    # R = a/b
    # b = np.sqrt(S/(R*pi))
    # a = np.sqrt(S/(R*pi)) * R

    np.random.seed(seed)
    while len(areas) > len(ltos):
        ltos = np.tile(ltos, 2)
    ltos = ltos[0 : len(areas)]

    shorts = np.sqrt(areas / (ltos * np.pi))
    longs = (shorts * ltos).astype(np.uint16)
    shorts = shorts.astype(np.uint16)
    axes = list(zip(longs, shorts))
    return axes


def add_sig_to_cell(labels, cell_mean_df, bg_mean_df, seed):
    np.random.seed(seed)

    cell_mean_df = cell_mean_df[cell_mean_df["prob"] > 0]
    cells = np.array([index for index, row in cell_mean_df.iterrows() for i in range(int(row["prob"] * 1000))])
    while np.sum(labels > 0) > len(cells):
        cells = np.tile(cells, 2)
    np.random.shuffle(cells)

    cells = cells[0 : np.sum(labels > 0)]

    bg_mean_df = bg_mean_df[bg_mean_df["prob"] > 0]
    bgs = np.array([index for index, row in bg_mean_df.iterrows() for i in range(int(row["prob"] * 1000))])
    while np.sum(labels == 0) > len(bgs):
        bgs = np.tile(bgs, 2)
    np.random.shuffle(bgs)
    bgs = bgs[0 : np.sum(labels == 0)]

    sigs = np.zeros_like(labels, dtype=np.int16)
    sigs[labels > 0] = cells
    sigs[labels == 0] = bgs

    return sigs


def simulate_cell_and_sig(
    area_df,
    ltos,
    cell_sig_df,
    bg_sig_df,
    prefix,
    cell_num=100,
    height=500,
    width=500,
    seed=1,
    max_iter=20000,
    shift_length=100,
):
    labels = get_cell_pos(
        area_df=area_df,
        ltos=ltos,
        cell_num=cell_num,
        height=height,
        width=width,
        seed=seed,
        max_iter=max_iter,
        shift_length=shift_length,
    )
    sigs = add_sig_to_cell(labels, cell_sig_df, bg_sig_df, seed)
    # adata = AnnData(X=csr_matrix(sigs))
    # adata.layers['labels'] = labels

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    out_file = prefix + "/seed" + str(seed) + ".txt"
    x, y = np.where(sigs > 0)
    df = pd.DataFrame({"geneID": "Malat1", "x": x, "y": y, "MIDCounts": sigs[sigs > 0]})
    df.to_csv(out_file, sep="\t", index=False)

    labels_file = prefix + "/seed" + str(seed) + ".labels.pkl"
    o = open(labels_file, "wb")
    pickle.dump(labels, o)
    o.close()
