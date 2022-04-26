'''
Make videos given a folder of folders of image frames
Images are selected by a format string
'''

import argparse
from sys import argv
from os import scandir, walk
from os.path import join
from glob import glob
from subprocess import run

from tqdm import tqdm
from multiprocessing import Pool

opts = None

def contain_img(d):
    return len(glob(join(d, '*.jpg'))) > 0

def worker_func(args):
    d, opts = args
    run(
        [
            'ffmpeg',
            '-loglevel', 'panic',
            '-y',                       # overwrite existing files
            '-framerate', str(opts.fps),
            '-i', join(d, '%06d.jpg'),
            '-c:v', 'libx264',          # H.264 encoding
            '-pix_fmt', 'yuv420p',      # to be compatible with browser viewing
            '-vf', 'pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2', 
            # pad so that w & h are even numbers
            join(d + '.mp4'),
        ],
        check=True,
    )

def main():
    parser = argparse.ArgumentParser(description='Make videos given folders of images')
    parser.add_argument('dir', help='working directory')
    parser.add_argument('-w', '--n-worker', type=int, default=4, help='number of parallel workers')
    parser.add_argument('-r', '--recursive', action='store_true', help='parse all subdirectories recursively')
    parser.add_argument('--fps', type=float, default=30, help='frames per second')
    
    opts = parser.parse_args()
    
    img_dirs = []
    if opts.recursive:
        for d, _, fs in walk(opts.dir):
            if not fs:
                continue
            if contain_img(d):
                img_dirs.append(join(opts.dir, d))
    else:
        for item in scandir(opts.dir):
            if not item.is_dir():
                pass
            d = join(opts.dir, item.name)
            if contain_img(d):
                img_dirs.append(d)
   
    args = [(d, opts) for d in img_dirs]

    if opts.n_worker == 0:
        for arg in tqdm(args):
            worker_func(arg)
    else:
        pool = Pool(opts.n_worker)
        try:
            list(
                tqdm(
                    pool.imap_unordered(worker_func, args),
                    total=len(args),
                )
            )
        except KeyboardInterrupt:
            print('Cancelled')
            pool.terminate()

if __name__ == "__main__":
    main()
