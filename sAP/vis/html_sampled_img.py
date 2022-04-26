'''
Generate a webpage containing a table of sampled results
The first column is ground truth, and the rest of columns
are different methods
'''

import json, socket, pickle, random
from os.path import join, basename, isfile
from glob import glob

import numpy as np

from html4vision import Col, imagetable

# the line below is for running both in the current directory 
# and the repo's root directory
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2

data_dir = '/data2/mengtial'
split = 'val'
annot_file = join(data_dir, 'Argoverse-HD/annotations', split + '.json')

vis_cfg = 'vis-th0.5'
out_dir = mkdir2(join(data_dir, 'Exp', 'Argoverse-HD', 'vis'))
out_name = 'single-vs-inf-gpus.html'
title = 'Single vs Infinite GPUs'
metric = 'AP'
link_video = True
n_show = 10
n_consec = None
align = True        # align to the stride in each sequence
stride = 30
random.seed(0)


names = [
    'Annotation',
    'Single GPU',
    'Infinite GPUs',
]

dirs = [
    join(data_dir, 'Argoverse-HD', 'vis', split),
    join(data_dir, 'Exp', 'Argoverse-HD', vis_cfg, 'rt_mrcnn50_nm_s0.5', split),
    join(data_dir, 'Exp', 'Argoverse-HD', vis_cfg, 'srt_mrcnn50_nm_inf_s0.5', split),
]

for d in dirs:
    print(f'python vis/make_videos_numbered.py "{d}" --fps 30')

srv_dir = data_dir
srv_port = 1234
host_name = socket.gethostname()

##

db = json.load(open(annot_file))
imgs = db['images']
seqs = db['sequences']


n_img = len(imgs)

if n_consec is None:
    # naive random sampling
    sel = random.choices(list(range(n_img)), k=n_show)
elif align:
    # sample multiple sets of consecutive frames
    start_idx = []
    last_sid = None
    for i, img in enumerate(imgs):
        if img['sid'] != last_sid:
            start_idx.append(i)
            last_sid = img['sid']
    start_idx = np.array(start_idx)

    sel = random.choices(list(range(n_img//n_consec)), k=n_show//n_consec)
    sel = np.array(sel)
    sel *= n_consec
    for i in range(len(sel)):
        diff = sel[i] - start_idx
        diff[diff < 0] = n_img
        nearest = np.argmin(diff)
        sel[i] -= (sel[i] - start_idx[nearest]) % stride
    # it is possible to have duplicated sel, but ignore for now
    consecs = np.arange(n_consec)
    sel = [i + consecs for i in sel]
    sel = np.array(sel).flatten().tolist()
else:
    sel = random.choices(list(range(n_img//n_consec)), k=n_show//n_consec)
    consecs = np.arange(n_consec)
    sel = [n_consec*i + consecs for i in sel]
    sel = np.array(sel).flatten().tolist()

img_paths = []
vid_paths = []

for idx in sel:
    img = imgs[idx]
    seq = seqs[img['sid']]
    img_paths.append(join(seq, '%06d.jpg' % (img['fid'] + 1)))
    vid_paths.append(seq + '.mp4')

cols = [Col('id1', 'ID')]
summary_row = [metric]

for i, name in enumerate(names):
    paths = [join(dirs[i], p) for p in img_paths]
    if link_video:
        hrefs = [join(dirs[i], p) for p in vid_paths]
    else:
        hrefs = paths
    cols.append(
        Col('img', name, paths, href=hrefs)
    )

    if 'Exp' in dirs[i]:
        eval_path = join(dirs[i].replace(vis_cfg, 'output'), 'eval_summary.pkl')
        if isfile(eval_path):
            eval_summary = pickle.load(open(eval_path, 'rb'))
            summary_row.append('%.1f' % (100*eval_summary['stats'][0]))
        else:
            summary_row.append('')
    else:
        summary_row.append('')

imagetable(
    cols,
    join(out_dir, out_name),
    title,
    summary_row=summary_row,
    imsize=(288, 180),
    sortable=True,
    sticky_header=True,
    sort_style='materialize',
    style='body {margin: 0}\n.html4vision td img {display: block; margin-left: auto; margin-right: auto;}',
    pathrep=srv_dir,
)

url = f'http://{host_name}:{srv_port}{out_dir.replace(srv_dir, "")}/{out_name}'
print(url)