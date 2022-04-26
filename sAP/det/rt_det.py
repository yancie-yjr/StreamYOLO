'''
Real-time detection
Given a real-time stream of input, it runs the detector and stores the timestamped output
'''

import argparse, json, pickle

from os.path import join, isfile, basename
from glob import glob
from time import perf_counter

from tqdm import tqdm
import numpy as np

import torch

from pycocotools.coco import COCO

# the line below is for running in both the current directory 
# and the repo's root directory
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2, print_stats
from det import imread, parse_det_result
from det.det_apis import init_detector, inference_detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--det-stride', type=float, default=1)
    parser.add_argument('--in-scale', type=float, default=None)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--no-class-mapping', action='store_true', default=False)
    parser.add_argument('--cpu-pre', action='store_true', default=False)
    parser.add_argument('--dynamic-schedule', action='store_true', default=False)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def main():
    assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing

    opts = parse_args()
    mkdir2(opts.out_dir)

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    coco_mapping = None if opts.no_class_mapping else db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    model = init_detector(opts)

    # warm up the GPU
    img = db.imgs[0]
    w_img, h_img = img['width'], img['height']
    _ = inference_detector(model, np.zeros((h_img, w_img, 3), np.uint8))
    torch.cuda.synchronize()

    runtime_all = []
    n_processed = 0
    n_total = 0

    for sid, seq in enumerate(tqdm(seqs)):
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        
        # load all frames in advance
        frames = []
        for img in frame_list:
            img_path = join(opts.data_root, seq_dirs[sid], img['name'])
            frames.append(imread(img_path))
        n_frame = len(frames)
        n_total += n_frame
        
        timestamps = []
        results_raw = []
        results_parsed = []
        input_fidx = []
        runtime = []
        last_fidx = None
        if not opts.dynamic_schedule:
            stride_cnt = 0
        
        t_total = n_frame/opts.fps
        t_start = perf_counter()
        while 1:
            t1 = perf_counter()
            t_elapsed = t1 - t_start
            if t_elapsed >= t_total:
                break

            # identify latest available frame
            fidx_continous = t_elapsed*opts.fps
            fidx = int(np.floor(fidx_continous))
            if fidx == last_fidx:
                continue
            
            last_fidx = fidx
            if opts.dynamic_schedule:
                fidx_remainder = fidx_continous - fidx
                if fidx_remainder > 0.5:
                    continue
            else:
                if stride_cnt % opts.det_stride == 0:
                    stride_cnt = 1
                else:
                    stride_cnt += 1
                    continue

            frame = frames[fidx]
            result = inference_detector(model, frame, gpu_pre=not opts.cpu_pre)
            bboxes, scores, labels, masks = \
                parse_det_result(result, coco_mapping, n_class)

            torch.cuda.synchronize()

            t2 = perf_counter()
            t_elapsed = t2 - t_start
            if t_elapsed >= t_total:
                break

            timestamps.append(t_elapsed)
            results_raw.append(result)
            results_parsed.append((bboxes, scores, labels, masks))
            input_fidx.append(fidx)
            runtime.append(t2 - t1)

        out_path = join(opts.out_dir, seq + '.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump({
                'results_raw': results_raw,
                'results_parsed': results_parsed,
                'timestamps': timestamps,
                'input_fidx': input_fidx,
                'runtime': runtime,
            }, open(out_path, 'wb'))

        runtime_all += runtime
        n_processed += len(results_raw)

    runtime_all_np = np.asarray(runtime_all)
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