import datetime
import numpy as np
import sys
import time

class ProgressBar(object):
    """
    Print pretty progress bar
    """
    def __init__(self, njobs, start=None, message=""):
        self.njobs = njobs
        self.start = (start if start else time.time())
        self.message = message
        self.finished = 0

    @property
    def progress(self):
        return 100 * (self.finished / float(self.njobs))

    @property
    def elapsed(self):
        return datetime.timedelta(seconds=int(time.time() - self.start))

    def update(self):
        # build the bar
        hashes = '#' * int(self.progress / 5.)
        nohash = ' ' * int(20 - len(hashes))

        # print to stderr
        print("\r[{}] {:>3}% {} | {:<12} ".format(*[
            hashes + nohash,
            int(self.progress),
            self.elapsed,
            self.message,
        ]), end="")
        sys.stdout.flush()

# used in vcf_to_hdf5
## with N and - masked to 9
GETCONS = np.array([
    [82, 71, 65],
    [75, 71, 84],
    [83, 71, 67],
    [89, 84, 67],
    [87, 84, 65],
    [77, 67, 65], 
    [78, 9, 9],
    [45, 9, 9],
    ], dtype=np.uint8)

# # used in write_outfiles.write_geno
TRANSFULL = {
    ('G', 'A'): "R",
    ('G', 'T'): "K",
    ('G', 'C'): "S",
    ('T', 'C'): "Y",
    ('T', 'A'): "W",
    ('C', 'A'): "M",
    ('A', 'C'): "M",
    ('A', 'T'): "W",
    ('C', 'T'): "Y",
    ('C', 'G'): "S",
    ('T', 'G'): "K",
    ('A', 'G'): "R",
    }
