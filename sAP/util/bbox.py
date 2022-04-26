'''
Utility function for bounding boxes
'''

import numpy as np

def ltwh2ltrb_(bboxes):
    if len(bboxes):
        if bboxes.ndim == 1:
            bboxes[2:] += bboxes[:2]
        else:
            bboxes[:, 2:] += bboxes[:, :2]
    return bboxes

def ltrb2ltwh_(bboxes):
    if len(bboxes):
        if bboxes.ndim == 1:
            bboxes[2:] -= bboxes[:2]
        else:
            bboxes[:, 2:] -= bboxes[:, :2]
    return bboxes

def ltwh2cxywh_(bboxes):
    if len(bboxes):
        if bboxes.ndim == 1:
            bboxes[:2] += bboxes[2:]/2
        else:
            bboxes[:, :2] += bboxes[:, 2:]/2
    return bboxes

def cxywh2ltwh_(bboxes):
    if len(bboxes):
        if bboxes.ndim == 1:
            bboxes[:2] -= bboxes[2:]/2
        else:
            bboxes[:, :2] -= bboxes[:, 2:]/2
    return bboxes

def wh2sr_(bboxes):
    if bboxes.ndim == 1:
        squeeze = True
        bboxes = bboxes[None, :]
    else:
        squeeze = False
    wh = bboxes[:, 2:].copy()
    bboxes[:, 2] = np.sqrt(wh[:, 0]*wh[:, 1])
    bboxes[:, 3] = wh[:, 0]/wh[:, 1]
    if squeeze:
        bboxes = bboxes[0]
    return bboxes

def sr2wh_(bboxes):
    if bboxes.ndim == 1:
        squeeze = True
        bboxes = bboxes[None, :]
    else:
        squeeze = False
    sr = bboxes[:, 2:].copy()
    sr[:, 0] = sr[:, 0]*sr[:, 0]
    stimesr = sr[:, 0]*sr[:, 1]
    valid = stimesr > 0

    bboxes[:, 2:] = 0
    bboxes[valid, 2] = np.sqrt(stimesr[valid])
    bboxes[valid, 3] = sr[valid, 0]/bboxes[valid, 2]
    if squeeze:
        bboxes = bboxes[0]
    return bboxes

def ltwh2ltrb(bboxes):
    bboxes = bboxes.copy()
    return ltwh2ltrb_(bboxes)

def ltrb2ltwh(bboxes):
    bboxes = bboxes.copy()
    return ltrb2ltwh_(bboxes)

def ltwh2cxywh(bboxes):
    bboxes = bboxes.copy()
    return ltwh2cxywh_(bboxes)

def cxywh2ltwh(bboxes):
    bboxes = bboxes.copy()
    return cxywh2ltwh_(bboxes)

def wh2sr(bboxes):
    bboxes = bboxes.copy()
    return wh2sr_(bboxes)

def sr2wh(bboxes):
    bboxes = bboxes.copy()
    return sr2wh_(bboxes)

