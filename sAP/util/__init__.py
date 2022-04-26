'''
Utility function module
'''

import os
import numpy as np

def mkdir2(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def print_stats(var, name='', fmt='%.3g', cvt=lambda x: x):
    var = np.asarray(var)
    
    if name:
        prefix = name + ': '
    else:
        prefix = ''

    if len(var) == 1:
        print(('%sscalar: ' + fmt) % (
            prefix,
            cvt(var[0]),
        ))
    else:
        fmt_str = 'mean: %s; std: %s; min: %s; max: %s' % (
            fmt, fmt, fmt, fmt
        )
        print(('%s' + fmt_str) % (
            prefix,
            cvt(var.mean()),
            cvt(var.std(ddof=1)),
            cvt(var.min()),
            cvt(var.max()),
        ))