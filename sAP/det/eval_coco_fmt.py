'''
A wrapper for evaluating detection results with cocoapi
'''

import argparse, pickle

from os.path import join, isfile

from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

# the line below is for running in both the current directory 
# and the repo's root directory
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
from det import eval_ccf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--result-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--per-class', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--eval-mask', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def main():
    opts = parse_args()

    mkdir2(opts.out_dir)
    db = COCO(opts.annot_path)

    eval_summary = eval_ccf(db, opts.result_path, None, opts.per_class)
    out_path = join(opts.out_dir, 'eval_summary.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump(eval_summary, open(out_path, 'wb'))
    if opts.eval_mask:
        print('Evaluating instance segmentation')
        eval_summary = eval_ccf(db, opts.result_path, iou_type='segm')
        out_path = join(opts.out_dir, 'eval_summary_mask.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump(eval_summary, open(out_path, 'wb'))

if __name__ == '__main__':
    main()