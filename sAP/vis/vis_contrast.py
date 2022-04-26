'''
Visualize detection output given a confidence threshold
'''

import argparse
from os import scandir
from os.path import join, isfile

from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# the line below is for running both in the current directory 
# and the repo's root directory
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
from vis.make_videos_numbered import worker_func as make_video



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-A', type=str, default=None)
    parser.add_argument('--dir-B', type=str, default=None)
    parser.add_argument('--horizontal', action='store_true', default=False)
    parser.add_argument('--split-pos', type=float, default=0.5)
    parser.add_argument('--split-animation', type=str, default=None)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--vis-scale', type=float, default=1)
    parser.add_argument('--seq', type=str, default=None)
    parser.add_argument('--make-video', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts


# Smoothing functions
# map time from 0-1 to progress from 0-1
def ease_in_out(t):
    return -np.cos(np.pi*t)/2 + 0.5

# animations
def split_anime_swing(t, split_pos, l, line_width):
    # timing information in seconds
    durations = [4, 1, 3, 2, 3, 1]

    small_end = -line_width//2 - 1
    big_end = l + line_width//2

    k = 0
    last_key = 0

    if t < durations[k]:
        return split_pos
    last_key += durations[k]
    k += 1

    if t < last_key + durations[k]:
        start_pos = split_pos
        end_pos = big_end
        p = ease_in_out((t - last_key)/durations[k])
        return start_pos + p*(end_pos - start_pos)
    last_key += durations[k]
    k += 1

    if t < last_key + durations[k]:
        return big_end
    last_key += durations[k]
    k += 1

    if t < last_key + durations[k]:
        start_pos = big_end
        end_pos = small_end
        p = ease_in_out((t - last_key)/durations[k])
        return start_pos + p*(end_pos - start_pos)
    last_key += durations[k]
    k += 1

    if t < last_key + durations[k]:
        return small_end
    last_key += durations[k]
    k += 1

    if t < last_key + durations[k]:
        start_pos = small_end
        end_pos = split_pos
        p = ease_in_out((t - last_key)/durations[k])
        return start_pos + p*(end_pos - start_pos)

    return split_pos

def main():
    opts = parse_args()

    seqs = sorted([item.name for item in scandir(opts.dir_A) if item.is_dir()])
    if opts.seq is not None:
        if opts.seq.isdigit():
            idx = int(opts.seq)
        else:
            idx = seqs.index(opts.seq)
        seqs = [seqs[idx]]

    line_width = 15
    line_color = [241, 159, 93]
    line_color = np.array(line_color, dtype=np.uint8).reshape((1, 1, 3))
    # font_path = r'C:\Windows\Fonts\Rock.otf'
    # font_path = r'C:\Windows\Fonts\AdobeGurmukhi-Regular.otf'
    # font = ImageFont.truetype(font_path, size=40)

    for s, seq in enumerate(seqs):
        print(f'Processing {s + 1}/{len(seqs)}: {seq}')
        seq_dir_A = join(opts.dir_A, seq)
        seq_dir_B = join(opts.dir_B, seq)
        seq_dir_out = mkdir2(join(opts.out_dir, seq))

        frame_list = [item.name for item in scandir(seq_dir_A) if item.is_file() and item.name.endswith('.jpg')]
        frame_list = sorted(frame_list)
        for ii, frame in enumerate(tqdm(frame_list)):
            out_path = join(seq_dir_out, frame)
            if not opts.overwrite and isfile(out_path):
                continue

            img_A = Image.open(join(seq_dir_A, frame))
            img_B = Image.open(join(seq_dir_B, frame))
            w, h = img_A.size
            l = h if opts.horizontal else w

            split_pos = opts.split_pos if opts.split_pos > 1 else l*opts.split_pos
            if opts.split_animation:
                split_pos = globals()['split_anime_' + opts.split_animation](
                    ii/opts.fps, split_pos, l, line_width,
                )
            split_pos = int(round(split_pos))

            line_start = split_pos - (line_width - 1)//2
            line_end = split_pos + line_width//2            # inclusive

            # using TrueType supported in PIL
            # draw = ImageDraw.Draw(img)
            # draw.text(
            #     (lt[0], lt[1] - font.size),
            #     text, (*color, 1), # RGBA
            #     font=font,
            # )

            if split_pos <= 0:
                img = np.array(img_B)
            else:
                img = np.array(img_A)
                img_B = np.asarray(img_B)
                if opts.horizontal:
                    img[split_pos:] = img_B[split_pos:]
                else:
                    img[:, split_pos:] = img_B[:, split_pos:]
            
            if line_start < l and line_end >= 0:
                # line is visible
                line_start = max(0, line_start)
                line_end = min(l, line_end)
                if opts.horizontal:
                    img[line_start:line_end, :] = line_color
                else:
                    img[:, line_start:line_end] = line_color

            Image.fromarray(img).save(out_path)

    if opts.make_video:
        out_path = seq_dir_out + '.mp4'
        if opts.overwrite or not isfile(out_path):
            print('Making the video')
            make_video((seq_dir_out, opts))
    else:
        print(f'python vis/make_videos_numbered.py "{opts.out_dir}" --fps {opts.fps}')

if __name__ == '__main__':
    main()