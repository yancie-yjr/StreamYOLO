'''
Runtime distributions
'''


import numpy as np

class Empirical():
    def __init__(self, samples, perf_factor=1):
        self.samples = np.array(samples)
        assert perf_factor > 0, perf_factor
        if perf_factor != 1:
            self.samples /= perf_factor
        self.sidx = 0

    def draw(self):
        return np.random.choice(self.samples)

    def draw_sequential(self):
        sample = self.samples[self.sidx]
        self.sidx = (self.sidx + 1) % len(self.samples)
        return sample

    def mean(self):
        return self.samples.mean()

    def std(self):
        return self.samples.std(ddof=1)
    
    def min(self):
        return self.samples.min()

    def max(self):
        return self.samples.max()

def dist_from_dict(dist_dict, perf_factor=1):
    if dist_dict['type'] == 'empirical':
        return Empirical(dist_dict['samples'], perf_factor)
    else:
        raise ValueError(f'Unknown distribution type "{dist_dict["type"]}"')

    