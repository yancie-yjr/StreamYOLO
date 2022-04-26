'''
Create a dummy dataset from a folder of images
Note that all images in the folder should be of the same size
'''

from os import scandir
from os.path import join, isfile, dirname

import argparse, json
from PIL import Image

# the line below is for running in both the current directory 
# and the repo's root directory
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
from dbcode.dbinfo import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-folder', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def main():
    opts = parse_args()
    if not opts.overwrite and isfile(opts.out_path):
        return
    mkdir2(dirname(opts.out_path))

    class_subset = avhd_subset
    cats = []
    class_names = []
    for i, c in enumerate(class_subset):
        cats.append({'id': i, 'name': coco_classes[c]})
        class_names.append(coco_classes[c])

    n_coco = len(coco_classes)
    coco_mapping = n_coco*[n_coco]
    for i, c in enumerate(class_subset):
        coco_mapping[c] = i

    frames = sorted([item.name for item in scandir(opts.img_folder) if item.is_file()])
    if opts.start is not None:
        frames = frames[opts.start:]
    img = Image.open(join(opts.img_folder, frames[0]))
    img_width, img_height = img.size

    seqs = [opts.prefix]
    seq_dirs = [opts.prefix]
    
    imgs = []
    for i, name in enumerate(frames):
        imgs.append({
            'id': i,
            'sid': 0,
            'fid': i,
            'name': name,
            'width': img_width,
            'height': img_height,
        })

    dataset = {
        'categories': cats,
        'images': imgs,
        'sequences': seqs,
        'seq_dirs': seq_dirs,
        'coco_subset': class_subset,
        'coco_mapping': coco_mapping,
    }

    json.dump(dataset, open(opts.out_path, 'w'))

if __name__ == '__main__':
    main()