''' 
IoU-based greedy association + batched Kalman Filter
implemented as post-processing (zero runtime assumption)
batching is based on pytorch's batched matrix operations
using notations from Wikipedia
'''


import argparse, json, pickle
from os.path import join, isfile
from time import perf_counter

from tqdm import tqdm
import numpy as np

import torch

from pycocotools.coco import COCO

# the line below is for running in both the current directory 
# and the repo's root directory
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2, print_stats
from util.bbox import ltrb2ltwh_, ltwh2ltrb_
from det import imread, parse_det_result, eval_ccf
from track import vis_track, track_based_shuffle
from track import iou_assoc
# from track.iou_assoc_cp import iou_assoc
from forecast import extrap_clean_up


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--eta', type=float, default=0, help='eta >= -1')
    parser.add_argument('--assoc', type=str, default='iou')
    parser.add_argument('--match-iou-th', type=float, default=0.3)
    parser.add_argument('--forecast-rt-ub', type=float, default=0)
    parser.add_argument('--forecast-before-assoc', action='store_true', default=False)
    parser.add_argument('--in-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--vis-dir', type=str, default=None)
    parser.add_argument('--vis-scale', type=float, default=1)
    parser.add_argument('--no-eval', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts


def bbox2z(bboxes):
    return torch.from_numpy(bboxes).unsqueeze_(2)

def bbox2x(bboxes):
    x = torch.cat((torch.from_numpy(bboxes), torch.zeros(bboxes.shape)), dim=1)
    return x.unsqueeze_(2)

def x2bbox(x):
    return x[:, :4, 0].numpy()

def make_F(F, dt):
    F[[0, 1, 2, 3], [4, 5, 6, 7]] = dt
    return F

def make_Q(Q, dt):
    # assume the base Q is identity
    Q[[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]] = dt*dt
    return Q

def batch_kf_predict_only(F, x):
    return F @ x

def batch_kf_predict(F, x, P, Q):
    x = F @ x
    P = F @ P @ F.t() + Q
    return x, P

def batch_kf_update(z, x, P, R):
    # assume H is just slicing operation
    # y = z - Hx
    y = z - x[:, :4]

    # S = HPH' + R
    S = P[:, :4, :4] + R

    # K = PH'S^(-1)
    K = P[:, :, :4] @ S.inverse()

    # x = x + Ky
    x += K @ y

    # P = (I - KH)P
    P -= K @ P[:, :4]
    return x, P

def main():
    opts = parse_args()
    assert opts.forecast_before_assoc, "Not implemented"   # True

    mkdir2(opts.out_dir)
    vis_out = bool(opts.vis_dir)
    if vis_out:
        mkdir2(opts.vis_dir)

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    coco_mapping = db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    results_ccf = []
    in_time = 0
    miss = 0
    shifts = 0

    given_tracks = opts.assoc == 'given'
    assert not given_tracks, "Not implemented"
            
    t_assoc = []
    t_forecast = []

    with torch.no_grad():
        kf_F = torch.eye(8)   #状态转移矩阵 [8,8]
        kf_Q = torch.eye(8)   #过程噪声    [8,8]
        kf_R = 10*torch.eye(4)    #测量噪声  [8,8]
        kf_P_init = 100*torch.eye(8).unsqueeze(0)    #初始误差协方差 [1,8,8]

        for sid, seq in enumerate(tqdm(seqs)):
            frame_list = [img for img in db.imgs.values() if img['sid'] == sid]

            results = pickle.load(open(join(opts.in_dir, seq + '.pkl'), 'rb'))
            # use raw results when possible in case we change class subset during evaluation
            results_raw = results.get('results_raw', None)
            if results_raw is None:
                results_parsed = results['results_parsed']
            timestamps = results['timestamps']
            input_fidx = results['input_fidx']

            # t1 -> det1, t2 -> det2, interpolate at t3 (t3 is the current time)
            det_latest_p1 = 0           # latest detection index + 1
            det_t2 = None               # detection index at t2
            kf_x = torch.empty((0, 8, 1))
            kf_P = torch.empty((0, 8, 8))
            n_matched12 = 0

            if not given_tracks:
                tkidx = 0               # track starting index   true

            for ii, img in enumerate(frame_list):
                # pred, gt association by time
                t = (ii - opts.eta)/opts.fps     # 第几帧的时间
                while det_latest_p1 < len(timestamps) and timestamps[det_latest_p1] <= t:
                    det_latest_p1 += 1
                if det_latest_p1 == 0:
                    # no detection output
                    miss += 1
                    bboxes_t3, scores, labels, tracks = [], [], [], []
                else:
                    det_latest = det_latest_p1 - 1
                    ifidx = input_fidx[det_latest]
                    in_time += int(ii == ifidx)
                    shifts += ii - ifidx

                    if det_latest != det_t2:
                        # new detection
                        # we can now throw away old result (t1)
                        # the old one is kept for forecasting purpose

                        if len(kf_x) and opts.forecast_before_assoc:
                            dt = ifidx - input_fidx[det_t2]
                            dt = int(dt) # convert from numpy to basic python format
                            w_img, h_img = img['width'], img['height']

                            kf_F = make_F(kf_F, dt)
                            kf_Q = make_Q(kf_Q, dt)

                            kf_x, kf_P = batch_kf_predict(kf_F, kf_x, kf_P, kf_Q)
                            bboxes_f = x2bbox(kf_x)
                            
                        det_t2 = det_latest
                        if results_raw is None:
                            result_t2 = results_parsed[det_t2]
                            bboxes_t2, scores_t2, labels_t2, _ = result_t2[:4]
                            if not given_tracks and len(result_t2) > 4:
                                tracks_t2 = np.asarray(result_t2[4])
                            else:
                                tracks_t2 = np.empty((0,), np.uint32)
                        else:
                            result_t2 = results_raw[det_t2]
                            bboxes_t2, scores_t2, labels_t2, _ = \
                                parse_det_result(result_t2, coco_mapping, n_class)

                        t1 = perf_counter()
                        n = len(bboxes_t2)
                        if n:
                            if not given_tracks:
                                # put high scores det first for better iou matching
                                score_argsort = np.argsort(scores_t2)[::-1]
                                bboxes_t2 = bboxes_t2[score_argsort]
                                scores_t2 = scores_t2[score_argsort]
                                labels_t2 = labels_t2[score_argsort]

                            ltrb2ltwh_(bboxes_t2)
                            
                            if given_tracks:
                                raise NotImplementedError
                                order1, order2, n_matched12 = track_based_shuffle(
                                    tracks, tracks_t2, no_unmatched1=True
                                )
                                tracks = tracks[order2]
                            else:
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
                                    if not given_tracks:
                                        tracks = np.arange(tkidx, tkidx + n, dtype=np.uint32)
                                        tkidx += n

                            t2 = perf_counter()
                            t_assoc.append(t2 - t1)

                    t3 = perf_counter()
                    if len(kf_x):
                        dt = ii - ifidx
                        w_img, h_img = img['width'], img['height']

                        # PyTorch small matrix multiplication is slow
                        # use numpy instead
                        kf_x_np = kf_x[:, :, 0].numpy()
                        bboxes_t3 = kf_x_np[:n_matched12, :4] + dt*kf_x_np[:n_matched12, 4:]
                        if n_matched12 < len(kf_x):
                            bboxes_t3 = np.concatenate((bboxes_t3, kf_x_np[n_matched12:, :4]))
                        
                        bboxes_t3, keep = extrap_clean_up(bboxes_t3, w_img, h_img, lt=True)
                        labels_t3 = labels[keep]
                        scores_t3 = scores[keep]
                        tracks_t3 = tracks[keep]


                    t4 = perf_counter()
                    t_forecast.append(t4 - t3)

                n = len(bboxes_t3)
                for i in range(n):
                    result_dict = {
                        'image_id': img['id'],
                        'bbox': bboxes_t3[i],
                        'score': scores_t3[i],
                        'category_id': labels_t3[i],
                    }
                    results_ccf.append(result_dict)

                if vis_out:
                    img_path = join(opts.data_root, seq_dirs[sid], img['name'])
                    I = imread(img_path)
                    vis_path = join(opts.vis_dir, seq, img['name'][:-3] + 'jpg')

                    bboxes = bboxes_t3.copy()
                    if n:
                        ltwh2ltrb_(bboxes)
                    if opts.overwrite or not isfile(vis_path):
                        vis_track(
                            I, bboxes, tracks, labels,
                            class_names, None, scores,
                            out_scale=opts.vis_scale,
                            out_file=vis_path,
                        )

    s2ms = lambda x: 1e3*x
    if len(t_assoc):
        print_stats(t_assoc, "RT association (ms)", cvt=s2ms)
    if len(t_forecast):
        print_stats(t_forecast, "RT forecasting (ms)", cvt=s2ms)    

    out_path = join(opts.out_dir, 'results_ccf.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump(results_ccf, open(out_path, 'wb'))

    if not opts.no_eval:
        eval_summary = eval_ccf(db, results_ccf)
        out_path = join(opts.out_dir, 'eval_summary.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump(eval_summary, open(out_path, 'wb'))

    if vis_out:
        print(f'python vis/make_videos.py "{opts.vis_dir}" --fps {opts.fps}')

if __name__ == '__main__':
    main()