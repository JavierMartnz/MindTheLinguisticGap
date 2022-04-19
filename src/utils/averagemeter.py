class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from helpers import *

class F1Meter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.F1 = 0

    def update(self, TP, TN, FP, FN):
        self.TP += TP
        self.TN += TN
        self.FP += FP
        self.FN += FN
        _, self.F1B, _, _ = f1_score_numpy(self.TP, self.TN, self.FP, self.FN)