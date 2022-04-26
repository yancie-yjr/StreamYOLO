'''
Our meta-detector Streamer
It converts a detector into a streaming perception system with a fixed output rate
'''

import argparse, json, pickle

from os.path import join, isfile, basename
from glob import glob
from time import perf_counter
import multiprocessing as mp
import traceback

from tqdm import tqdm
import numpy as np

import torch

from pycocotools.coco import COCO

# the line below is for running in both the current directory 
# and the repo's root directory
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2, print_stats
from util.bbox import ltrb2ltwh_, ltwh2ltrb_
from util.runtime_dist import dist_from_dict
from det import imread, parse_det_result
from det.det_apis import init_detector, inference_detector
from track import track_based_shuffle
# from track import iou_assoc
from track.iou_assoc_cp import iou_assoc

from forecast import extrap_clean_up
from forecast.pps_forecast_kf import \
    bbox2z, bbox2x, x2bbox, make_F, make_Q, \
    batch_kf_predict_only, batch_kf_predict, \
    batch_kf_update


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--eta', type=float, default=0, help='eta >= -1') # frame

    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--in-scale', type=float, default=None)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--cpu-pre', action='store_true', default=False)
    
    parser.add_argument('--dynamic-schedule', action='store_true', default=False)
    parser.add_argument('--runtime', type=str, required=True)
    parser.add_argument('--perf-factor', type=float, default=1)

    parser.add_argument('--match-iou-th', type=float, default=0.3)
    parser.add_argument('--forecast-rt-ub', type=float, default=0.003) # seconds

    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def det_process(opts, frame_recv, det_res_send, w_img, h_img):
    try:
        model = init_detector(opts)

        # warm up the GPU
        _ = inference_detector(model, np.zeros((h_img, w_img, 3), np.uint8))
        torch.cuda.synchronize()

        while 1:
            fidx = frame_recv.recv()
            if type(fidx) is list:
                # new video, read all images in advance
                frame_list = fidx
                frames = [imread(img_path) for img_path in frame_list]
                # signal ready, no errors
                det_res_send.send('ready')
                continue
            elif fidx is None:
                # exit flag
                break
            fidx, t1 = fidx
            img = frames[fidx]
            t2 = perf_counter() 
            t_send_frame = t2 - t1

            result = inference_detector(model, img, gpu_pre=not opts.cpu_pre)
            torch.cuda.synchronize()

            t3 = perf_counter()
            det_res_send.send([result, t_send_frame, t3])

    except Exception:
        # report all errors from the child process to the parent
        # forward traceback info as well
        det_res_send.send(Exception("".join(traceback.format_exception(*sys.exc_info()))))


def main():
    assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing

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

    img = db.imgs[0]
    w_img, h_img = img['width'], img['height']

    mp.set_start_method('spawn')
    frame_recv, frame_send = mp.Pipe(False)
    det_res_recv, det_res_send = mp.Pipe(False)
    det_proc = mp.Process(target=det_process, args=(opts, frame_recv, det_res_send, w_img, h_img))
    det_proc.start()

    if opts.dynamic_schedule:
        runtime = pickle.load(open(opts.runtime, 'rb'))
        runtime_dist = dist_from_dict(runtime, opts.perf_factor)
        mean_rtf = runtime_dist.mean()*opts.fps

    n_total = 0
    t_det_all = []
    t_send_frame_all = []
    t_recv_res_all = []
    t_assoc_all = []
    t_forecast_all = []

    with torch.no_grad():
        kf_F = torch.eye(8)
        kf_Q = torch.eye(8)
        kf_R = 10*torch.eye(4)
        kf_P_init = 100*torch.eye(8).unsqueeze(0)

        for sid, seq in enumerate(tqdm(seqs)):
            frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
            frame_list = [join(opts.data_root, seq_dirs[sid], img['name']) for img in frame_list]
            n_frame = len(frame_list)
            n_total += n_frame
            
            timestamps = []
            results_parsed = []
            input_fidx = []
            
            processing = False
            fidx_t2 = None            # detection input index at t2
            fidx_latest = None
            tkidx = 0                 # track starting index
            kf_x = torch.empty((0, 8, 1))
            kf_P = torch.empty((0, 8, 8))
            n_matched12 = 0

            # let detector process to read all the frames
            frame_send.send(frame_list)
            # it is possible that unfetched results remain in the pipe
            while 1:
                msg = det_res_recv.recv() # wait till the detector is ready
                if msg == 'ready':
                    break
                elif isinstance(msg, Exception):
                    raise msg

            t_total = n_frame/opts.fps
            t_unit = 1/opts.fps
            t_start = perf_counter()
            while 1:
                t1 = perf_counter()
                t_elapsed = t1 - t_start
                if t_elapsed >= t_total:
                    break

                # identify latest available frame
                fidx_continous = t_elapsed*opts.fps
                fidx = int(np.floor(fidx_continous))
                if fidx == fidx_latest:
                    # algorithm is fast and has some idle time
                    wait_for_next = True
                else:
                    wait_for_next = False
                    if opts.dynamic_schedule:
                        if mean_rtf >= 1:
                            # when runtime < 1, it should always process every frame
                            fidx_remainder = fidx_continous - fidx
                            if mean_rtf < np.floor(fidx_remainder + mean_rtf):
                                # wait till next frame
                                wait_for_next = True

                if wait_for_next:
                    # sleep
                    continue

                if not processing:
                    t_start_frame = perf_counter()
                    frame_send.send((fidx, t_start_frame))
                    fidx_latest = fidx
                    processing = True
  
                # wait till query - forecast-rt-ub
                wait_time = t_unit - opts.forecast_rt_ub
                if det_res_recv.poll(wait_time): # wait
                    # new result
                    result = det_res_recv.recv() 
                    if isinstance(result, Exception):
                        raise result
                    result, t_send_frame, t_start_res = result
                    bboxes_t2, scores_t2, labels_t2, _ = \
                        parse_det_result(result, coco_mapping, n_class)
                    processing = False
                    t_det_end = perf_counter()
                    t_det_all.append(t_det_end - t_start_frame)
                    t_send_frame_all.append(t_send_frame)
                    t_recv_res_all.append(t_det_end - t_start_res)

                    # associate across frames
                    t_assoc_start = perf_counter()
                    if len(kf_x):
                        dt = fidx_latest - fidx_t2

                        kf_F = make_F(kf_F, dt)
                        kf_Q = make_Q(kf_Q, dt)
                        kf_x, kf_P = batch_kf_predict(kf_F, kf_x, kf_P, kf_Q)
                        bboxes_f = x2bbox(kf_x)
                                        
                    fidx_t2 = fidx_latest

                    n = len(bboxes_t2)
                    if n:
                        # put high scores det first for better iou matching
                        score_argsort = np.argsort(scores_t2)[::-1]
                        bboxes_t2 = bboxes_t2[score_argsort]
                        scores_t2 = scores_t2[score_argsort]
                        labels_t2 = labels_t2[score_argsort]

                        ltrb2ltwh_(bboxes_t2)

                    updated = False
                    if len(kf_x):
                        order1, order2, n_matched12, tracks, tkidx = iou_assoc(
                            bboxes_f, labels, tracks, tkidx,
                            bboxes_t2, labels_t2, opts.match_iou_th,
                            no_unmatched1=True,
                        )

                        if n_matched12:
                            kf_x = kf_x[order1]
                            kf_P = kf_P[order1]
                            kf_x, kf_P = batch_kf_update(
                                bbox2z(bboxes_t2[order2[:n_matched12]]),
                                kf_x,
                                kf_P,
                                kf_R,
                            )
                    
                            kf_x_new = bbox2x(bboxes_t2[order2[n_matched12:]])
                            n_unmatched2 = len(bboxes_t2) - n_matched12
                            kf_P_new = kf_P_init.expand(n_unmatched2, -1, -1)
                            kf_x = torch.cat((kf_x, kf_x_new))
                            kf_P = torch.cat((kf_P, kf_P_new))
                            labels = labels_t2[order2]
                            scores = scores_t2[order2]
                            updated = True

                    if not updated:
                        # start from scratch
                        kf_x = bbox2x(bboxes_t2)
                        kf_P = kf_P_init.expand(len(bboxes_t2), -1, -1)
                        labels = labels_t2
                        scores = scores_t2
                        tracks = np.arange(tkidx, tkidx + n, dtype=np.uint32)
                        tkidx += n

                    t_assoc_end = perf_counter()
                    t_assoc_all.append(t_assoc_end - t_assoc_start)

                # apply forecasting for the current query
                t_forecast_start = perf_counter()
                query_pointer = fidx + opts.eta + 1
                
                if len(kf_x):
                    dt = (query_pointer - fidx_t2)

                    kf_x_np = kf_x[:, :, 0].numpy()
                    bboxes_t3 = kf_x_np[:n_matched12, :4] + dt*kf_x_np[:n_matched12, 4:]
                    if n_matched12 < len(kf_x):
                        bboxes_t3 = np.concatenate((bboxes_t3, kf_x_np[n_matched12:, :4]))
                        
                    bboxes_t3, keep = extrap_clean_up(bboxes_t3, w_img, h_img, lt=True)
                    labels_t3 = labels[keep]
                    scores_t3 = scores[keep]
                    tracks_t3 = tracks[keep]

                else:
                    bboxes_t3 = np.empty((0, 4), dtype=np.float32)
                    scores_t3 = np.empty((0,), dtype=np.float32)
                    labels_t3 = np.empty((0,), dtype=np.int32)
                    tracks_t3 = np.empty((0,), dtype=np.int32)

                t_forecast_end = perf_counter()
                t_forecast_all.append(t_forecast_end - t_forecast_start)
                
                t3 = perf_counter()
                t_elapsed = t3 - t_start
                if t_elapsed >= t_total:
                    break

                if len(bboxes_t3):
                    ltwh2ltrb_(bboxes_t3)
                if fidx_t2 is not None:
                    timestamps.append(t_elapsed)
                    results_parsed.append((bboxes_t3, scores_t3, labels_t3, None, tracks_t3))
                    input_fidx.append(fidx_t2)

            out_path = join(opts.out_dir, seq + '.pkl')
            if opts.overwrite or not isfile(out_path):
                pickle.dump({
                    'results_parsed': results_parsed,
                    'timestamps': timestamps,
                    'input_fidx': input_fidx,
                }, open(out_path, 'wb'))

    # terminates the child process
    frame_send.send(None)

    out_path = join(opts.out_dir, 'time_info.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump({
            'n_total': n_total,
            't_det': t_det_all,
            't_send_frame': t_send_frame_all,
            't_recv_res': t_recv_res_all,
            't_assoc': t_assoc_all,
            't_forecast': t_forecast_all,
        }, open(out_path, 'wb'))
 
    # convert to ms for display
    s2ms = lambda x: 1e3*x
    print_stats(t_det_all, 'Runtime detection (ms)', cvt=s2ms)
    print_stats(t_send_frame_all, 'Runtime sending the frame (ms)', cvt=s2ms)
    print_stats(t_recv_res_all, 'Runtime receiving the result (ms)', cvt=s2ms)
    print_stats(t_assoc_all, "Runtime association (ms)", cvt=s2ms)
    print_stats(t_forecast_all, "Runtime forecasting (ms)", cvt=s2ms)

if __name__ == '__main__':
    main()