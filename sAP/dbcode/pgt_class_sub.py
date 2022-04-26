'''
Set up pseudo ground truth from detection results
''' 

from os import scandir
from os.path import join, isfile

import json, pickle
from tqdm import tqdm

import pycocotools.mask as maskUtils

# the line below is for running in both the current directory 
# and the repo's root directory
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
from util.bbox import ltrb2ltwh_
from dbcode.dbinfo import *

data_dir = None # Put your data path here
if data_dir is None:
    raise Exception("Please modify the path before running this code")
av_dir = join(data_dir, 'Argoverse-1.1', 'tracking')
class_subset = avhd_subset

view = 'ring_front_center'
img_width, img_height = 1920, 1200
split = 'test'

det_dir = join(data_dir, 'Exp', 'Argoverse-1.1', 'htc_dconv2_ms', split)
det_th = 0.3

out_dir = mkdir2(join(data_dir, 'Argoverse-HD', 'annotations'))
out_name = 'htc_dconv2_ms_' + split + '.json'

seqs_dir = join(av_dir, split)
seqs = sorted([item.name for item in scandir(seqs_dir) if item.is_dir()])
seq_dirs = [split + '/' + seq + '/' + view for seq in seqs]

iid = 0
aid = 0
imgs = []
annots = []

cats = []
for i, c in enumerate(class_subset):
    cats.append({'id': i, 'name': coco_classes[c]})

n_class = len(cats)

for sid, seq in enumerate(tqdm(seqs)):
    fid = 0 # frame id
    seq_imgs = sorted(
        [item.name for item in scandir(join(av_dir, seq_dirs[sid]))
            if item.is_file()])
    for fid, name in enumerate(seq_imgs):
        det = pickle.load(open(join(det_dir, seq, name[:-3] + 'pkl'), 'rb'))
        bboxes, masks = det

        bboxes = [bboxes[c] for c in class_subset]
        masks = [masks[c] for c in class_subset]

        # convert to coco fmt
        for c, row in enumerate(bboxes):
            for i in range(row.shape[0]):
                bbox = row[i]
                score = bbox[4]
                if score < det_th:
                    continue
                bbox = bbox[:4]
                ltrb2ltwh_(bbox)

                mask = masks[c][i]
                mask['counts'] = mask['counts'].decode(encoding='UTF-8')

                annots.append({
                    'id': aid,
                    'image_id': iid,
                    'bbox': bbox.tolist(),
                    'category_id': c,
                    'segmentation': mask,
                    'area': float(maskUtils.area(mask)),
                    'iscrowd': False,
                    'ignore': False,
                })

                aid += 1

        imgs.append({
            'id': iid,
            'sid': sid,
            'fid': fid,
            'name': name,
            'width': img_width,
            'height': img_height,
        })

        iid += 1

n_coco = len(coco_classes)
coco_mapping = n_coco*[n_coco]
for i, c in enumerate(class_subset):
    coco_mapping[c] = i

dataset = {
    'categories': cats,
    'images': imgs,
    'annotations': annots,
    'sequences': seqs,
    'seq_dirs': seq_dirs,
    'coco_subset': class_subset,
    'coco_mapping': coco_mapping,
}

json.dump(dataset, open(join(out_dir, out_name), 'w'))
