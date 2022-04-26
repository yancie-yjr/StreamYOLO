'''
Simulated real-time detection with infinite GPUs
'''

import argparse, json, pickle
from os.path import join, isfile, basename

from tqdm import tqdm
import numpy as np

import torch

from pycocotools.coco import COCO

# the line below is for running in both the current directory 
# and the repo's root directory
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2, print_stats
from util.bbox import ltwh2ltrb_
from util.runtime_dist import dist_from_dict
from det import imread, parse_det_result, result_from_ccf
from det.det_apis import init_detector, inference_detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--in-scale', type=float, default=None)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--cached-res', type=str, default=None)
    parser.add_argument('--runtime', type=str, required=True)
    parser.add_argument('--perf-factor', type=float, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def main():
    opts = parse_args()

    mkdir2(opts.out_dir)

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    coco_mapping = db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    if opts.cached_res:
        cache_in_ccf = '_ccf' in basename(opts.cached_res)
        if cache_in_ccf:
            # speed up based on the assumption of sequential storage
            cache_end_idx = 0
        cached_res = pickle.load(open(opts.cached_res, 'rb'))
    else:
        assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing
        model = init_detector(opts)

    np.random.seed(opts.seed)
    runtime = pickle.load(open(opts.runtime, 'rb'))
    runtime_dist = dist_from_dict(runtime, opts.perf_factor)

    runtime_all = []
    n_processed = 0
    n_total = 0

    for sid, seq in enumerate(tqdm(seqs)):
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        n_frame = len(frame_list)
        n_total += n_frame

        if not opts.cached_res:
            # load all frames in advance
            frames = []
            for img in frame_list:
                img_path = join(opts.data_root, seq_dirs[sid], img['name'])
                frames.append(imread(img_path))

        timestamps = []
        results_raw = []
        results_parsed = []
        input_fidx = []
        runtime = []
        if opts.cached_res and cache_in_ccf:
            results_raw = None
        else:
            results_raw = []

        for ii in range(n_frame):
            t = ii/opts.fps

            if opts.cached_res:
                img = frame_list[ii]
                if cache_in_ccf:
                    cache_end_idx, bboxes, scores, labels, masks = \
                        result_from_ccf(cached_res, img['id'], cache_end_idx)
                    ltwh2ltrb_(bboxes)
                else:
                    result = cached_res[img['id']]
                    bboxes, scores, labels, masks = \
                        parse_det_result(result, coco_mapping, n_class)
            else:
                frame = frames[ii]
                result = inference_detector(model, frame)
                bboxes, scores, labels, masks = \
                    parse_det_result(result, coco_mapping, n_class)

            rt_this = runtime_dist.draw()
            t_finishing = t + rt_this

            timestamps.append(t_finishing)
            if results_raw is not None:
                results_raw.append(result)
            results_parsed.append((bboxes, scores, labels, masks))
            input_fidx.append(ii)
            runtime.append(rt_this)

        # since parallel excecution, the order is not guaranteed
        idx = np.argsort(timestamps)
        timestamps = [timestamps[i] for i in idx]
        if results_raw is not None:
            results_raw = [results_raw[i] for i in idx]
        results_parsed = [results_parsed[i] for i in idx]
        input_fidx = [input_fidx[i] for i in idx]
        runtime = [runtime[i] for i in idx]
        
        out_path = join(opts.out_dir, seq + '.pkl')
        if opts.overwrite or not isfile(out_path):
            out_dict = {
                'results_parsed': results_parsed,
                'timestamps': timestamps,
                'input_fidx': input_fidx,
                'runtime': runtime,
            }
            if results_raw is not None:
                out_dict['results_raw'] = results_raw
            pickle.dump(out_dict, open(out_path, 'wb'))

        runtime_all += runtime
        n_processed += len(results_parsed)

    runtime_all_np = np.array(runtime_all)
    n_small_runtime = (runtime_all_np < 1.0/opts.fps).sum()

    out_path = join(opts.out_dir, 'time_info.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump({
            'runtime_all': runtime_all,
            'n_processed': n_processed,
            'n_total': n_total,
            'n_small_runtime': n_small_runtime,
        }, open(out_path, 'wb'))  

    # convert to ms for display
    s2ms = lambda x: 1e3*x

    print(f'{n_processed}/{n_total} frames processed')
    print_stats(runtime_all_np, 'Runtime (ms)', cvt=s2ms)
    print(f'Runtime smaller than unit time interval: '
        f'{n_small_runtime}/{n_processed} '
        f'({100.0*n_small_runtime/n_processed:.4g}%)')

if __name__ == '__main__':
    main()