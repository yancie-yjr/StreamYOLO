'''
Visualize detection output given a confidence threshold
'''

import argparse, json, pickle
from os.path import join, isfile

from tqdm import tqdm
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
from skimage.segmentation import find_boundaries

from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

# the line below is for running both in the current directory 
# and the repo's root directory
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
from det import imread, imwrite
from vis.make_videos_numbered import worker_func as make_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--result-path', type=str, default=None)
    parser.add_argument('--gt', action='store_true', default=False)
    parser.add_argument('--vis-dir', type=str, required=True)
    parser.add_argument('--vis-scale', type=float, default=1)
    parser.add_argument('--seq', type=str, default=None)
    parser.add_argument('--score-th', type=float, default=0.3)
    parser.add_argument('--make-video', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

class_palette = {
    0: (196, 48, 22),   # person
    1: (63, 199, 10),   # bicycle
    2: (29, 211, 224),  # car
    3: (163, 207, 52),  # motorcycle
    5: (40, 23, 227),   # bus
    7: (29, 91, 224),   # truck
    9: (235, 197, 9),   # traffic_light
    10: (144, 39, 196), # fire_hydrant
    11: (196, 0, 0),    # stop_sign
}

def vis_obj_fancy(
    img, bboxes, labels, class_names,
    color_palette, color_scheme='class',
    masks=None, scores=None, score_th=0.3, tracks=None,
    iscrowd=None,
    out_scale=1, out_file=None,
    show_label=True, show_score=True,
    font=None,
):
    thickness = 2
    alpha = 0.2

    if isinstance(img, str):
        img = imread(img)
        img = np.array(img) # create a writable copy
    if out_scale != 1:
        img = cv2.resize(
            img,
            fx=out_scale, fy=out_scale,
            interpolation=cv2.INTER_LINEAR,
        )
    bboxes = np.asarray(bboxes)
    labels = np.asarray(labels)
    if masks is not None:
        masks = np.asarray(masks)

    empty = len(bboxes) == 0
    if not empty and scores is not None and score_th > 0:
        sel = scores >= score_th
        bboxes = bboxes[sel]
        labels = labels[sel]
        if masks is not None:
            masks = masks[sel]
        empty = len(bboxes) == 0
    
    if empty:
        if out_file is not None:
            imwrite(img, out_file)
        return img

    if out_scale != 1:
        bboxes = out_scale*bboxes

    bboxes = bboxes.round().astype(np.int32)

    # draw masks or filled rectangles
    if masks is None:
        img_filled = img.copy()
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            if color_scheme == 'class':
                color = color_palette[label]
            cv2.rectangle(
                img, 
                (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                color, thickness=-1,
            )
        img = cv2.addWeighted(img_filled, (1 - alpha), img, alpha, 0)
        # draw box contours
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            if color_scheme == 'class':
                color = color_palette[label]
            cv2.rectangle(
                img, 
                (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                color, thickness=thickness,
            )
    else:
        for i, (mask, label) in enumerate(zip(masks, labels)):
            if color_scheme == 'class':
                color = np.array(color_palette[label])
            m = maskUtils.decode(mask)
            if out_scale != 1:
                m = cv2.resize(
                    m.astype(np.uint8), 
                    fx=out_scale, fy=out_scale,
                    interpolation=cv2.INTER_NEAREST,
                )
            m = m.astype(np.bool)
            img[m] = (1 - alpha)*img[m] + alpha*color
            b = find_boundaries(m)
            img[b] = color

    # put label text
    if show_label or show_score:
        if font is not None:
            # using TrueType supported in PIL
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)

        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            if color_scheme == 'class':
                color = color_palette[label]

            lt = (bbox[0], bbox[1])

            if show_label:
                if class_names is None:
                    label_text = f'class {label}'
                else:
                    label_text = class_names[label]
            else:
                label_text = ''
            if show_score and scores is not None:
                score_text = f'{scores[i]:.02f}'
            else:
                score_text = ''
            if label_text and score_text:
                text = label_text + ' - ' + score_text
            elif label_text:
                text = label_text
            elif score_text:
                text = score_text
            else:
                text = ''
            
            if font is None:
                cv2.putText(
                    img, text, (lt[0], lt[1] - 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.7, color,
                    thickness=1,
                )
            else:
                draw.text(
                    (lt[0], lt[1] - font.size),
                    text, (*color, 1), # RGBA
                    font=font,
                )
        if font is not None:
            img = np.asarray(img)

    if out_file is not None:
        imwrite(img, out_file)
    return img

def main():
    opts = parse_args()

    mkdir2(opts.vis_dir)

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    coco_subset = np.asarray(db.dataset['coco_subset'])
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    color_palette = [
        class_palette[k] for k in coco_subset if k in class_palette
    ]

    if opts.gt:
        results_ccf = db.dataset['annotations']
    else:
        results_ccf = pickle.load(open(opts.result_path, 'rb'))

    if opts.seq is not None:
        if opts.seq.isdigit():
            idx = int(opts.seq)
        else:
            idx = seqs.index(opts.seq)
        seqs = [seqs[idx]]
    else:
        idx = None

    # font_path = r'C:\Windows\Fonts\AdobeGurmukhi-Regular.otf'
    # font = ImageFont.truetype(font_path, size=40)
    font = None

    for sid, seq in enumerate(tqdm(seqs)):
        if idx is not None:
            sid = idx
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        for ii, img in enumerate(frame_list):
            img_path = join(opts.data_root, seq_dirs[sid], img['name'])
            dets = [r for r in results_ccf if r['image_id'] == img['id']]
            bboxes = np.array([d['bbox'] for d in dets])
            masks = None
            if len(bboxes):
                bboxes[:, 2:] += bboxes[:, :2]
                if 'segmentation' in dets[0]:
                    masks = [d['segmentation'] for d in dets]

            labels = np.array([d['category_id'] for d in dets])
            if opts.gt:
                iscrowd = np.array([d['iscrowd'] for d in dets])
                scores = None
            else:
                iscrowd = None
                scores = np.array([d['score'] for d in dets])
            vis_path = join(opts.vis_dir, seq, '%06d.jpg' % (ii + 1)) # img['name'][:-3] + 'jpg'
            if opts.overwrite or not isfile(vis_path):
                vis_obj_fancy(
                    img_path, bboxes, labels, class_names,
                    color_palette, masks=masks,
                    show_label=False,
                    scores=scores, show_score=False,
                    iscrowd=iscrowd, score_th=opts.score_th,
                    out_scale=opts.vis_scale,
                    out_file=vis_path,
                    font=font,
                )

        if opts.make_video:
            seq_dir_out = join(opts.vis_dir, seq)
            out_path = seq_dir_out + '.mp4'
            if opts.overwrite or not isfile(out_path):
                make_video((seq_dir_out, opts))

    if not opts.make_video:
        print(f'python vis/make_videos_numbered.py "{opts.vis_dir}" --fps {opts.fps}')

if __name__ == '__main__':
    main()