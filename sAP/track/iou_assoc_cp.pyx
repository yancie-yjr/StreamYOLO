import numpy as np
import pycocotools.mask as maskUtils

def iou_assoc(bboxes1, labels1, tracks1, tkidx, bboxes2, labels2, match_iou_th, no_unmatched1=False):
    # iou-based association
    # shuffle all elements so that matched stays in the front
    # bboxes are in the form of a list of [l, t, w, h]
    m, n = len(bboxes1), len(bboxes2)
        
    _ = n*[0]
    ious = maskUtils.iou(bboxes1, bboxes2, _)

    match_fwd = m*[None]
    matched1 = []
    matched2 = []
    unmatched2 = []

    for j in range(n):
        best_iou = match_iou_th
        match_i = None
        for i in range(m):
            if match_fwd[i] is not None \
                or labels1[i] != labels2[j] \
                or ious[i, j] < best_iou:
                continue
            best_iou = ious[i, j]
            match_i = i
        if match_i is None:
            unmatched2.append(j)
        else:
            matched1.append(match_i)
            matched2.append(j)
            match_fwd[match_i] = j

    if no_unmatched1:
        order1 = matched1
    else:
        unmatched1 = list(set(range(m)) - set(matched1))
        order1 = matched1 + unmatched1
    order2 = matched2 + unmatched2

    n_matched = len(matched2)
    n_unmatched2 = len(unmatched2)
    tracks2 = np.concatenate((tracks1[order1][:n_matched],
        np.arange(tkidx, tkidx + n_unmatched2, dtype=tracks1.dtype)))
    tkidx += n_unmatched2

    return order1, order2, n_matched, tracks2, tkidx
