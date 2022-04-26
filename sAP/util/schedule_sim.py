
''' 
Scheduling Simulator
It tests out the empirical temporal mismatch
of different scheduling policies

For more information, check out Section B.1 in the appendix
'''

import math
from fractions import Fraction

# r = 1.51108
r = 1 + Fraction(1, 2)  # runtime in rational number form, exact computation!
T = 13                  # number of frames
eta = 0                 # observation pointer - query pointer

##
def sim(policy, r, T, eta):
    assert eta >= -1        # not implemented for eta < -1
    cmismatch = 0           # cumulative mismatch
    result_idx = None       # latest result input index
    process_idx = 0         # data index under processing
    t_finish = r            # always start with no idle time
    for t in range(T - eta):
        if t_finish < t:
            result_idx = process_idx
            if policy(t_finish, r):
                # wait
                t_finish = t + r
                process_idx = t
            else:
                process_idx = t if t_finish == t or result_idx == t - 1 else t - 1
                # result == t - 1 means r <= 1, the algorithm is already waiting
                t_finish += r
        # report latest result (excluding empty output cases)
        if t + eta >= 0 and result_idx is not None:
            cmismatch += t + eta - result_idx
    return cmismatch

##
tail = lambda x: x - math.floor(x)

def p_idle_free(t_finish, r):
    return False

def p_idle_next(t_finish, r):
    return True

def p_shrinking_tail(t_finish, r):
    cur_tail = tail(t_finish)
    next_tail = tail(t_finish + r)
    return cur_tail > next_tail

def p_half_tail(t_finish, r):
    cur_tail = tail(t_finish)
    return cur_tail >= 0.5

def p_half_next_tail(t_finish, r):
    next_tail = tail(t_finish + r)
    return next_tail < 0.5

##
all_vars = list(globals().keys())
for name in all_vars:
    if not name.startswith('p_'):
        continue
    p = globals()[name]
    cmismatch = sim(p, r, T, eta)
    print(f'{name[2:]}: {cmismatch}, {cmismatch/T:.6g}')

