import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_mutual_info_score


def cal_ami(a1, a2):
    labels1 = a1.astype(np.int16)
    labels2 = a2.astype(np.int16)
    ami = adjusted_mutual_info_score(labels1.flatten(), labels2.flatten())
    return ami


def cal_f1score(a1, a2, binary=True):
    labels1 = a1.astype(np.int16)
    labels2 = a2.astype(np.int16)
    if binary:
        labels1[labels1 > 0] = 1
        labels2[labels2 > 0] = 1
    ami = f1_score(labels1.flatten(), labels2.flatten())
    return ami


def cal_precision(a1, a2, tau=0.5):
    labels1 = a1.astype(np.int16)  # prediction
    labels2 = a2.astype(np.int16)  # ground true

    tps = []
    nonfns = []

    pre_ids = [i for i in np.unique(labels1) if i > 0]
    for id in pre_ids:
        gt_ids = [i for i in np.unique(labels2[labels1 == id]) if i > 0]
        for gt_id in gt_ids:
            iou = np.sum((labels1 == id) & (labels2 == gt_id)) / np.sum((labels1 == id) | (labels2 == gt_id))
            if iou >= tau:
                tps.append(id)
                nonfns.append(gt_id)

    tp = len(set(tps))

    fp = len(pre_ids) - tp

    fn = len([i for i in np.unique(labels2) if i > 0]) - len(set(nonfns))

    p = tp / (tp + fp + fn)

    return p
