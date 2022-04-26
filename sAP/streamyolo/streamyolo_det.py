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
from torchvision.ops import batched_nms
import cv2
from yolox.exp import get_exp
from yolox.utils import fuse_model
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--det-stride', type=float, default=1)
    parser.add_argument('--in_scale', type=float, default=0.5)
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



def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def preproc(img, input_size, swap=(2, 0, 1)):
    resized_img = cv2.resize(img, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR,)
    resized_img = resized_img.transpose(swap)
    return resized_img

def inference(outputs, conf_thre=0.01, nms_thresh=0.65, in_scale = 0.5):
    box_corner = outputs.new(outputs.shape)
    box_corner[:, 0] = outputs[:, 0] - outputs[:, 2] / 2
    box_corner[:, 1] = outputs[:, 1] - outputs[:, 3] / 2
    box_corner[:, 2] = outputs[:, 0] + outputs[:, 2] / 2
    box_corner[:, 3] = outputs[:, 1] + outputs[:, 3] / 2
    outputs[:, :4] = box_corner[:, :4]

    class_conf, class_pred = torch.max(outputs[:, 5:], 1, keepdim=True)
    conf_mask = (outputs[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
    detections = torch.cat((outputs[:, :5], class_conf, class_pred.float()), 1)
    detections = detections[conf_mask]

    nms_out_index = batched_nms(
        detections[:, :4],
        detections[:, 4] * detections[:, 5],
        detections[:, 6],
        nms_thresh,
    )

    detections = detections[nms_out_index].cpu().detach().numpy()
    return detections[:, :4] / in_scale, detections[:, 4] * detections[:, 5], detections[:, 6].astype(np.int32), None





def main():
    # assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing

    opts = parse_args()
    mkdir2(opts.out_dir)

    db = COCO(opts.annot_path)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    ###model 
    exp = get_exp(opts.config, None)
    model = exp.get_model()
    model.cuda()
    model.eval()
    ckpt = torch.load(opts.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print("loaded checkpoint done.")
    # model = fuse_model(model)
    model.eval()
    model.half()
    # tensor_type = torch.cuda.FloatTensor
    tensor_type = torch.cuda.HalfTensor

    # warm up the GPU
    img = db.imgs[0]
    w_img, h_img = img['width'], img['height']
    tmp_image = torch.ones(1, 3, int(h_img/2), int(w_img/2)).type(tensor_type)
    buffer_ = None
    for i in range(10):
        _, _ = model(tmp_image, buffer=buffer_, mode='on_pipe')

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
            frames.append(cv2.imread(img_path))
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

        buffer = None  # buffer feature

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
            h_img, w_img = int(1200 * opts.in_scale), int(1920 * opts.in_scale)
            frame = preproc(frame, input_size=(h_img, w_img))  # [3,600,960]
            with torch.no_grad():
                frame = torch.from_numpy(frame).unsqueeze(0).type(tensor_type)    # [1,3,600,960]
                result, buffer = model(frame, buffer=buffer, mode='on_pipe')
                bboxes, scores, labels, masks = inference(result[0])

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