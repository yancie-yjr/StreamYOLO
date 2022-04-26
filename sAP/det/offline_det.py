'''
Offline detection
Run a given detector on every image in a dataset with no time constraint
'''

import argparse, json, pickle
from os.path import join, isfile

from tqdm import tqdm
import numpy as np

import torch

from pycocotools.coco import COCO

# the line below is for running in both the current directory 
# and the repo's root directory
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
from util.bbox import ltrb2ltwh_
from det import imread, parse_det_result, vis_det, eval_ccf
from det.det_apis import init_detector, inference_detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--in-scale', type=float, default=None)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--no-class-mapping', action='store_true', default=False)
    parser.add_argument('--cpu-pre', action='store_true', default=False)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--vis-dir', type=str, default=None)
    parser.add_argument('--vis-scale', type=float, default=1)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--no-eval', action='store_true', default=False)
    parser.add_argument('--eval-mask', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def main():
    assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing

    opts = parse_args()

    mkdir2(opts.out_dir)
    vis_out = bool(opts.vis_dir)
    if vis_out:
        mkdir2(opts.vis_dir)

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    coco_mapping = None if opts.no_class_mapping else db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    model = init_detector(opts)
    results_raw = []        # image based, all 80 COCO classes
    results_ccf = []        # instance based

    for iid, img in tqdm(db.imgs.items()):
        img_name = img['name']

        sid = img['sid']
        seq_name = seqs[sid]

        img_path = join(opts.data_root, seq_dirs[sid], img_name)
        I = imread(img_path)
        
        result = inference_detector(model, I, gpu_pre=not opts.cpu_pre)
        results_raw.append(result)
        bboxes, scores, labels, masks = \
            parse_det_result(result, coco_mapping, n_class)

        if vis_out:
            vis_path = join(opts.vis_dir, seq_name, img_name[:-3] + 'jpg')
            if opts.overwrite or not isfile(vis_path):
                vis_det(
                    I, bboxes, labels,
                    class_names, masks, scores,
                    out_scale=opts.vis_scale,
                    out_file=vis_path
                )

        # convert to coco fmt
        n = len(bboxes)
        if n:
            ltrb2ltwh_(bboxes)

        for i in range(n):
            result_dict = {
                'image_id': iid,
                'bbox': bboxes[i],
                'score': scores[i],
                'category_id': labels[i],
            }
            if masks is not None:
                result_dict['segmentation'] = masks[i]
            results_ccf.append(result_dict)

    out_path = join(opts.out_dir, 'results_raw.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump(results_raw, open(out_path, 'wb'))

    out_path = join(opts.out_dir, 'results_ccf.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump(results_ccf, open(out_path, 'wb'))

    if not opts.no_eval:
        eval_summary = eval_ccf(db, results_ccf)
        out_path = join(opts.out_dir, 'eval_summary.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump(eval_summary, open(out_path, 'wb'))
        if opts.eval_mask:
            print('Evaluating instance segmentation')
            eval_summary = eval_ccf(db, results_ccf, iou_type='segm')
            out_path = join(opts.out_dir, 'eval_summary_mask.pkl')
            if opts.overwrite or not isfile(out_path):
                pickle.dump(eval_summary, open(out_path, 'wb'))

    if vis_out:
        print(f'python vis/make_videos.py "{opts.vis_dir}"')

if __name__ == '__main__':
    main()