''' 
Forecasting module
This file contains helper functions for forecasting
'''

import numpy as np
import cv2

import pycocotools.mask as maskUtils


def extrap_clean_up_single(bbox, w_img, h_img, min_size=75):
    # bbox in ltwh
    if bbox[2] <= 0 or bbox[3] <=0:
        return None

    # convert [l, t, w, h] to [l, t, r, b]
    bbox[2:] = bbox[:2] + bbox[2:]

    # clip to the image
    bbox[[0, 2]] = bbox[[0, 2]].clip(0, w_img)
    bbox[[1, 3]] = bbox[[1, 3]].clip(0, h_img)

    # convert [l, t, r, b] to [l, t, w, h]
    bbox[2:] = bbox[2:] - bbox[:2]

    # int conversion is neccessary, otherwise, there are very small w, h that round up to 0
    if bbox[2].astype(np.int)*bbox[3].astype(np.int) < min_size:
        return None

    return bbox

def extrap_clean_up(bboxes, w_img, h_img, min_size=75, lt=False):
    # bboxes in the format of [cx or l, cy or t, w, h]
    wh_nz = bboxes[:, 2:] > 0
    keep = np.logical_and(wh_nz[:, 0], wh_nz[:, 1])

    if lt:
        # convert [l, t, w, h] to [l, t, r, b]
        bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
    else:
        # convert [cx, cy, w, h] to [l, t, r, b]
        bboxes[:, :2] = bboxes[:, :2] - bboxes[:, 2:]/2
        bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]

    # clip to the image
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, w_img)
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, h_img)

    # convert [l, t, r, b] to [l, t, w, h]
    bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]

    # int conversion is neccessary, otherwise, there are very small w, h that round up to 0
    keep = np.logical_and(keep, bboxes[:, 2].astype(np.int)*bboxes[:, 3].astype(np.int) >= min_size)
    bboxes = bboxes[keep]
    return bboxes, keep


def bbox_extrap_lin(bboxes1, bboxes2, tk, w_img, h_img, min_size=75):
    bboxes3 = bboxes2 + tk*(bboxes2 - bboxes1)

    return extrap_clean_up(bboxes3, w_img, h_img, min_size, lt=True)

def bbox_extrap_quad(bboxes1, bboxes2, v1, v2, tk, t32, w_img, h_img, min_size=75):
    n_v1, n_v2 = len(v1), len(v2)
    assert n_v1 <= n_v2
    assert len(bboxes1) == len(bboxes2)
    assert n_v2 <= len(bboxes2)
    
    if n_v1 == 0 or n_v2 == 0:
        return bbox_extrap_lin(bboxes1, bboxes2, tk, w_img, h_img, min_size)

    v2 = v2[:n_v1]
    v3 = v2 + tk*(v2 - v1)

    # create a copy and convert list to numpy array
    bboxes1 = np.array(bboxes1)
    bboxes2 = np.array(bboxes2)

    # convert [l, t, w, h] to [cx, cy, w, h]
    bboxes1[:, :2] = bboxes1[:, :2] + bboxes1[:, 2:]/2
    bboxes2[:, :2] = bboxes2[:, :2] + bboxes2[:, 2:]/2

    bboxes3 = bboxes2[:n_v1] + v3*t32
    if len(bboxes1) > n_v1:
        bboxes3_lin = bboxes2[n_v1:] + tk*(bboxes2[n_v1:] - bboxes1[n_v1:])
        bboxes3 = np.vstack((bboxes3, bboxes3_lin))

    return extrap_clean_up(bboxes3, w_img, h_img, min_size)


def warp_mask_to_box(masks1, bboxes1, bboxes2):
    # create a copy and convert list to numpy array
    # bboxes in ltwh
    bboxes1 = np.array(bboxes1).astype(np.int)
    bboxes2 = np.array(bboxes2).astype(np.int)

    masks2 = []
    for m1, b1, b2 in zip(masks1, bboxes1, bboxes2):
        m1 = maskUtils.decode(m1)
        h_img, w_img = m1.shape
        m1 = m1[b1[1]: b1[1] + b1[3], b1[0]: b1[0] + b1[2]]
        m1 = cv2.resize(
            m1.astype(np.uint8),
            tuple(b2[2:].tolist()),
            interpolation=cv2.INTER_NEAREST,
        )
        m2 = np.zeros((h_img, w_img), dtype=np.uint8)
        bottom = min(b2[1] + b2[3], h_img)
        right = min(b2[0] + b2[2], w_img)
        m2[b2[1]: bottom, b2[0]: right] = m1[:bottom - b2[1], :right - b2[0]] 
        rle = maskUtils.encode(
            np.array(m2[:, :, np.newaxis], order='F'))[0]
        masks2.append(rle)

    return np.asarray(masks2)

