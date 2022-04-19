import torch
from torch import nn, Tensor, sigmoid
from torch.autograd import Variable
import numpy as np

from torch.nn.modules.distance import PairwiseDistance
from scipy import interpolate
import math
from draw import show_ROC
class TripletLoss(nn.Module):

    def __init__(self, alpha):
        super(TripletLoss, self).__init__()
        self.alpha = alpha
        self.pdist = PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.alpha + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)

        return loss


class CE_Loss(nn.Module):
    def __init__(self):
        super(CE_Loss, self).__init__()
        self.skeleton_loss_fun = nn.MSELoss()
        self.cls_loss_fun = nn.CrossEntropyLoss(reduction="mean")
        self.cls_loss_fun = self.cls_loss_fun.to("cuda")


    def forward(self, cls_results, cls_targets):



        cls_results_clone = cls_results.clone()
        cls_targets_clone = cls_targets.clone()

        # cls_results_valid = cls_results_clone[valid, :]
        # cls_targets_valid = cls_targets_clone[valid]

        # skeleton_loss = self.skeleton_loss_fun(skeleton_results_valid.view(-1), skeleton_targets_valid.view(-1)) * 144
        cls_loss = self.cls_loss_fun(cls_results_clone, cls_targets_clone)
        loss = cls_loss

        # tj :  computing F1 - score
        pre_cls = np.argmax(cls_results_clone.detach().cpu().numpy(), axis=1) > 0
        gt_cls = cls_targets_clone.detach().cpu().numpy() > 0



        TP = (pre_cls & gt_cls).sum()
        TN = ((~pre_cls) & (~gt_cls)).sum()
        FP = (pre_cls & (~gt_cls)).sum()
        FN = ((~pre_cls) & gt_cls).sum()

        return loss, TP, TN, FP, FN


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, distance_metric=0,
                  subtract_mean=False):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    pos_idx = actual_issame == 1
    neg_idx = actual_issame == 0

    num_pairs = min(len(actual_issame), embeddings1.shape[0])
    num_thresholds = len(thresholds)

    tpr_thresholds = np.zeros(num_thresholds)
    fpr_thresholds = np.zeros(num_thresholds)

    if subtract_mean:
        mean = np.mean(np.concatenate([embeddings1, embeddings2]), axis=0)
    else:
        mean = 0.0
    dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

    # Find the best threshold for the fold
    acc_thresholds = np.zeros(num_thresholds)
    for threshold_idx, threshold in enumerate(thresholds):
        tpr_thresholds[threshold_idx], fpr_thresholds[threshold_idx], acc_thresholds[threshold_idx] = calculate_accuracy(threshold, dist,
                                                                                                     actual_issame)

    max_accuracy = max(acc_thresholds)

    thresh_for_max_acc = thresholds[np.argmax(acc_thresholds)]

    _,_,pos_acc = calculate_accuracy(thresh_for_max_acc, dist[pos_idx], actual_issame[pos_idx])
    _, _, neg_acc = calculate_accuracy(thresh_for_max_acc, dist[neg_idx], actual_issame[neg_idx])

    return tpr_thresholds, fpr_thresholds, max_accuracy, thresh_for_max_acc, pos_acc, neg_acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, distance_metric=0,
                  subtract_mean=False):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    num_pairs = min(len(actual_issame), embeddings1.shape[0])
    num_thresholds = len(thresholds)

    if subtract_mean:
        mean = np.mean(np.concatenate([embeddings1, embeddings2]), axis=0)
    else:
        mean = 0.0

    dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

    # Find the threshold that gives FAR = far_target
    far_theshholds = np.zeros(num_thresholds)
    val_theshholds = np.zeros(num_thresholds)

    for threshold_idx, threshold in enumerate(thresholds):
        val_theshholds[threshold_idx], far_theshholds[threshold_idx] = calculate_val_far(threshold, dist, actual_issame)

    if np.max(far_theshholds) >= far_target:
        f = interpolate.interp1d(far_theshholds, thresholds, kind='slinear')
        threshold = f(far_target)
    else:
        threshold = 0.0
    # tj : show_ROC(far_theshholds, val_theshholds)
    # val, far = calculate_val_far(f(0.1), dist, actual_issame)
    val, far = calculate_val_far(threshold, dist, actual_issame)

    return val, far, val_theshholds, far_theshholds


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_same != 0:
        val = float(true_accept) / float(n_same)
    else:
        val = -1

    if n_diff != 0:
        far = float(false_accept) / float(n_diff)
    else:
        far = -1
    return val, far