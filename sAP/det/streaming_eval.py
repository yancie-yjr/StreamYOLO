'''
Streaming evaluation
Given real-time or simulated real-time detection outputs,
it pairs them with the ground truth and evaluates the pairs

Note that this script does not need to run in real-time
'''

import argparse, json, pickle
from os.path import join, isfile

from tqdm import tqdm
import numpy as np

from pycocotools.coco import COCO

# the line below is for running in both the current directory 
# and the repo's root directory
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
from util.bbox import ltrb2ltwh
from det import imread, parse_det_result, vis_det, eval_ccf
from track import vis_track


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--eta', type=float, default=0, help='eta >= -1')
    parser.add_argument('--result-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--vis-dir', type=str, default=None)
    parser.add_argument('--vis-scale', type=float, default=1)
    parser.add_argument('--no-class-mapping', action='store_true', default=False)    
    parser.add_argument('--no-eval', action='store_true', default=False)
    parser.add_argument('--use-parsed', action='store_true', default=False)
    parser.add_argument('--eval-mask', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def main():
    opts = parse_args()

    out_dir = mkdir2(opts.out_dir) if opts.out_dir else opts.result_dir
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

    results_ccf = []
    in_time = 0
    miss = 0
    mismatch = 0

    print('Pairing the output with the ground truth')

    for sid, seq in enumerate(tqdm(seqs)):
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        
        results = pickle.load(open(join(opts.result_dir, seq + '.pkl'), 'rb'))
        # use raw results when possible in case we change class subset during evaluation
        if opts.use_parsed:
            results_parsed = results['results_parsed']
        else:
            results_raw = results.get('results_raw', None)
            if results_raw is None:
                results_parsed = results['results_parsed']
        timestamps = results['timestamps']
        input_fidx = results['input_fidx']

        tidx_p1 = 0
        for ii, img in enumerate(frame_list):
            # pred, gt association by time
            t = (ii - opts.eta)/opts.fps
            while tidx_p1 < len(timestamps) and timestamps[tidx_p1] <= t:
                tidx_p1 += 1
            if tidx_p1 == 0:
                # no output
                miss += 1
                bboxes, scores, labels  = [], [], []
                masks, tracks = None, None
            else:
                tidx = tidx_p1 - 1
                ifidx = input_fidx[tidx]
                in_time += int(ii == ifidx)
                mismatch += ii - ifidx

                if opts.use_parsed or results_raw is None:
                    result = results_parsed[tidx]
                    bboxes, scores, labels, masks = result[:4]
                    if len(result) > 4:
                        tracks = result[4]
                    else:
                        tracks = None
                else:
                    result = results_raw[tidx]
                    bboxes, scores, labels, masks = \
                        parse_det_result(result, coco_mapping, n_class)
                    tracks = None
                    
            if vis_out:
                img_path = join(opts.data_root, seq_dirs[sid], img['name'])
                I = imread(img_path)
                vis_path = join(opts.vis_dir, seq, img['name'][:-3] + 'jpg')
                if opts.overwrite or not isfile(vis_path):
                    if tracks is None:
                        vis_det(
                            I, bboxes, labels,
                            class_names, masks, scores,
                            out_scale=opts.vis_scale,
                            out_file=vis_path,
                        )
                    else:
                        vis_track(
                            I, bboxes, tracks, labels,
                            class_names, masks, scores,
                            out_scale=opts.vis_scale,
                            out_file=vis_path,
                        )

            # convert to coco fmt
            n = len(bboxes)
            if n:
                bboxes_ltwh = ltrb2ltwh(bboxes)

            for i in range(n):
                result_dict = {
                    'image_id': img['id'],
                    'bbox': bboxes_ltwh[i],
                    'score': scores[i],
                    'category_id': labels[i],
                }
                if masks is not None:
                    result_dict['segmentation'] = masks[i]

                results_ccf.append(result_dict)

    out_path = join(out_dir, 'results_ccf.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump(results_ccf, open(out_path, 'wb'))

    out_path = join(out_dir, 'eval_assoc.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump({
            'miss': miss,
            'in_time': in_time,
            'mismatch': mismatch,
        }, open(out_path, 'wb'))

    if not opts.no_eval:
        eval_summary = eval_ccf(db, results_ccf)
        out_path = join(out_dir, 'eval_summary.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump(eval_summary, open(out_path, 'wb'))
        if opts.eval_mask:
            print('Evaluating instance segmentation')
            eval_summary = eval_ccf(db, results_ccf, iou_type='segm')
            out_path = join(out_dir, 'eval_summary_mask.pkl')
            if opts.overwrite or not isfile(out_path):
                pickle.dump(eval_summary, open(out_path, 'wb'))

    if vis_out:
        print(f'python vis/make_videos.py "{opts.vis_dir}" --fps {opts.fps}')

if __name__ == '__main__':
    main()