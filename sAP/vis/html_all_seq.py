'''
Generate a webpage containing a grid of thumbnails
for all sequences (videos) in a dataset.
Each thunbnail link to the corresponding video sequence
'''

import json, socket, pickle
from os.path import join, isfile, dirname, basename
from os import scandir

import numpy as np

from html4vision import Col, imagetile


# the line below is for running both in the current directory 
# and the repo's root directory
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2

data_dir = '/data2/mengtial'
split = 'val'
folder = join(data_dir, 'Argoverse-HD', 'vis', split)
print(f'python vis/make_videos.py "{folder}" --fps 30')

out_dir = mkdir2(join(data_dir, 'Exp', 'Argoverse-HD', 'visf'))
out_name = 'gt_all_seq.html'
title = 'Ground Truth (All Sequences)'
link_video = True
n_show = 100
np.random.seed(0)

srv_dir = data_dir
srv_port = 1234
host_name = socket.gethostname()

##

seqs = sorted([item.name for item in scandir(folder) if item.is_dir()])
n_seq = len(seqs)

img_paths = []
vid_paths = []

for i, seq in enumerate(seqs):
    frames = [item.name for item in scandir(join(folder, seq)) if item.is_file() and item.name.endswith('.jpg')]
    frames = sorted(frames)
    frame = np.random.choice(frames)
    # fetch a random image for thumbnail
    img_paths.append(join(folder, seq, frame))
    vid_paths.append(join(folder, seq + '.mp4'))

hrefs = vid_paths if link_video else img_paths
captions = [f'{i+1}. {seq}' for i, seq in enumerate(seqs)]

imagetile(
    img_paths, 3,
    join(out_dir, out_name),
    title,
    caption=captions,
    href=hrefs,
    subset=n_show,
    imscale=0.3,
    pathrep=srv_dir,
    copyright=False,
)

url = f'http://{host_name}:{srv_port}{out_dir.replace(srv_dir, "")}/{out_name}'
print(url)