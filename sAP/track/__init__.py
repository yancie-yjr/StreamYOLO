
'''
Tracking module
'''

import numpy as np
import mmcv, cv2

import pycocotools.mask as maskUtils
from det import imwrite

def vis_track(img, bboxes, tracks, labels, class_names,
    masks=None, scores=None, score_th=0,
    out_scale=1, out_file=None):

    if out_scale != 1:
        img = mmcv.imrescale(img, out_scale, interpolation='bilinear')

    bboxes = np.asarray(bboxes)
    labels = np.asarray(labels)
    tracks = np.asarray(tracks)
    if masks is not None:
        masks = np.asarray(masks)

    empty = len(bboxes) == 0
    if not empty and scores is not None and score_th > 0:
        sel = scores >= score_th
        bboxes = bboxes[sel]
        labels = labels[sel]
        scores = scores[sel]
        tracks = tracks[sel]
        if masks is not None:
            masks = masks[sel]
        empty = len(bboxes) == 0

    if empty:
        if out_file is not None:
            imwrite(img, out_file)
        return img

    if out_scale != 1:
        bboxes = out_scale*bboxes
        # we don't want in-place operations like bboxes *= out_scale

    if masks is not None:
        for mask in masks:
            color = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8
            )
            m = maskUtils.decode(mask)
            if out_scale != 1:
                m = mmcv.imrescale(
                    m.astype(np.uint8), out_scale,
                    interpolation='nearest'
                )
            m = m.astype(np.bool)
            img[m] = 0.5*img[m] + 0.5*color


    thickness = 1
    font_scale = 0.5

    bboxes = bboxes.round().astype(np.int32)
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        color = vis_track.palettes[tracks[i] % vis_track.palettes.shape[0]]
        color = color.tolist()

        lt = (bbox[0], bbox[1])
        rb = (bbox[2], bbox[3])
        cv2.rectangle(
            img, lt, rb, color, thickness=thickness
        )
        if class_names is None:
            label_text = f'class {label}'
        else:
            label_text = class_names[label]
        if scores is not None:
            label_text += f'|{scores[i]:.02f}'
        cv2.putText(
            img, label_text, (bbox[0], bbox[1] - 2),
            cv2.FONT_HERSHEY_COMPLEX, font_scale,
            color,
        )

    if out_file is not None:
        imwrite(img, out_file)
vis_track.palettes = np.random.randint(0, 256, (100, 3), dtype=np.uint8)


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

def iou_assoc_no_tracks(bboxes1, labels1, bboxes2, labels2, match_iou_th, no_unmatched1=False):
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

    return order1, order2, n_matched

def track_based_shuffle(tracks1, tracks2, no_unmatched1=False):
    # shuffle all elements so that matched stays in the front
    in1 = np.in1d(tracks1, tracks2)
    in2 = np.in1d(tracks2, tracks1)
    matched1 = np.nonzero(in1)[0]
    matched2 = np.nonzero(in2)[0]
    n_matched = len(matched1)
    if no_unmatched1:
        order1 = matched1
    else:
        unmatched1 = np.nonzero(~in1)[0]
        order1 = np.concatenate((matched1, unmatched1))
    unmatched2 = np.nonzero(~in2)[0]
    order2 = np.concatenate((matched2, unmatched2))
    return order1, order2, n_matched