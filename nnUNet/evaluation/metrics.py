#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
from medpy import metric
import sklearn.metrics as metrics
import pymia.evaluation.metric as m
import torch


def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)


class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))


def jaccard(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp + fn))


def precision(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp))


def sensitivity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fn))


def recall(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    return sensitivity(test, reference, confusion_matrix, nan_for_nonexisting, **kwargs)


def specificity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tn / (tn + fp))


def accuracy(test=None, reference=None, confusion_matrix=None, **kwargs):
    """(TP + TN) / (TP + FP + FN + TN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float((tp + tn) / (tp + fp + tn + fn))


def fscore(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, beta=1., **kwargs):
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""

    precision_ = precision(test, reference, confusion_matrix, nan_for_nonexisting)
    recall_ = recall(test, reference, confusion_matrix, nan_for_nonexisting)

    return (1 + beta*beta) * precision_ * recall_ /\
        ((beta*beta * precision_) + recall_)


def false_positive_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FP / (FP + TN)"""

    return 1 - specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_omission_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TN + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(fn / (fn + tn))


def false_negative_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TP + FN)"""

    return 1 - sensitivity(test, reference, confusion_matrix, nan_for_nonexisting)


def true_negative_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    return specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_discovery_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FP / (TP + FP)"""

    return 1 - precision(test, reference, confusion_matrix, nan_for_nonexisting)


def negative_predictive_value(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FN)"""

    return 1 - false_omission_rate(test, reference, confusion_matrix, nan_for_nonexisting)


def total_positives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fp


def total_negatives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fn


def total_positives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fn


def total_negatives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fp


def hausdorff_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd(test, reference, voxel_spacing, connectivity)


def hausdorff_distance_95(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd95(test, reference, voxel_spacing, connectivity)


def avg_surface_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.asd(test, reference, voxel_spacing, connectivity)


def avg_surface_distance_symmetric(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.assd(test, reference, voxel_spacing, connectivity)

"""

Simple test
ece = np_fn.ece_binary(probabilities, target, self.n_bins, self.threshold_range, mask, out_bins,
                               self.bin_weighting)
"""

def ece_binary(probabilities, target, n_bins=10, threshold_range: tuple = None, mask=None, out_bins: dict = None,
               bin_weighting='proportion'):
    '''
    probabilities(155, 240, 240, 2)
    target(155, 240, 240)
    '''
    target = target.cpu().numpy()
    probabilities = probabilities.cpu().detach().numpy()
    n_dim = target.ndim

    pos_frac, mean_confidence, bin_count, non_zero_bins = \
        binary_calibration(probabilities, target, n_bins, threshold_range, mask)

    bin_proportions = _get_proportion(bin_weighting, bin_count, non_zero_bins, n_dim)

    if out_bins is not None:
        out_bins['bins_count'] = bin_count
        out_bins['bins_avg_confidence'] = mean_confidence
        out_bins['bins_positive_fraction'] = pos_frac
        out_bins['bins_non_zero'] = non_zero_bins

    ece = (np.abs(mean_confidence - pos_frac) * bin_proportions).sum()
    return ece,mean_confidence.sum(),pos_frac.sum()


def binary_calibration(probabilities, target, n_bins=10, threshold_range: tuple = None, mask=None):
    if probabilities.ndim > target.ndim:
        if probabilities.shape[-1] > 2:
            raise ValueError('can only evaluate the calibration for binary classification')
        elif probabilities.shape[-1] == 2:
            probabilities = probabilities[..., 1]
        else:
            probabilities = np.squeeze(probabilities, axis=-1)

    if mask is not None:
        probabilities = probabilities[mask]
        target = target[mask]

    if threshold_range is not None:
        low_thres, up_thres = threshold_range
        mask = np.logical_and(probabilities < up_thres, probabilities > low_thres)
        probabilities = probabilities[mask]
        target = target[mask]

    pos_frac, mean_confidence, bin_count, non_zero_bins = \
        _binary_calibration(target.flatten(), probabilities.flatten(), n_bins)

    return pos_frac, mean_confidence, bin_count, non_zero_bins


def _binary_calibration(target, probs_positive_cls, n_bins=10):
    # same as sklearn.calibration calibration_curve but with the bin_count returned
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(probs_positive_cls, bins) - 1

    # # note: this is the original formulation which has always n_bins + 1 as length
    # bin_sums = np.bincount(binids, weights=probs_positive_cls, minlength=len(bins))
    # bin_true = np.bincount(binids, weights=target, minlength=len(bins))
    # bin_total = np.bincount(binids, minlength=len(bins))

    bin_sums = np.bincount(binids, weights=probs_positive_cls, minlength=n_bins)
    bin_true = np.bincount(binids, weights=target, minlength=n_bins)
    bin_total = np.bincount(binids, minlength=n_bins)

    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])

    return prob_true, prob_pred, bin_total[nonzero], nonzero


def _get_proportion(bin_weighting: str, bin_count: np.ndarray, non_zero_bins: np.ndarray, n_dim: int):
    if bin_weighting == 'proportion':
        bin_proportions = bin_count / bin_count.sum()
    elif bin_weighting == 'log_proportion':
        bin_proportions = np.log(bin_count) / np.log(bin_count).sum()
    elif bin_weighting == 'power_proportion':
        bin_proportions = bin_count**(1/n_dim) / (bin_count**(1/n_dim)).sum()
    elif bin_weighting == 'mean_proportion':
        bin_proportions = 1 / non_zero_bins.sum()
    else:
        raise ValueError('unknown bin weighting "{}"'.format(bin_weighting))
    return bin_proportions


def uncertainty(prediction, target, thresholded_uncertainty, mask=None):
    if mask is not None:
        prediction = prediction[mask]
        target = target[mask]
        thresholded_uncertainty = thresholded_uncertainty[mask]

    tps = np.logical_and(target, prediction)
    tns = np.logical_and(~target, ~prediction)
    fps = np.logical_and(~target, prediction)
    fns = np.logical_and(target, ~prediction)

    tpu = np.logical_and(tps, thresholded_uncertainty).sum()
    tnu = np.logical_and(tns, thresholded_uncertainty).sum()
    fpu = np.logical_and(fps, thresholded_uncertainty).sum()
    fnu = np.logical_and(fns, thresholded_uncertainty).sum()

    tp = tps.sum()
    tn = tns.sum()
    fp = fps.sum()
    fn = fns.sum()

    return tp, tn, fp, fn, tpu, tnu, fpu, fnu


def error_dice(fp, fn, tpu, tnu, fpu, fnu):
    if ((fnu + fpu) == 0) and ((fn + fp + fnu + fpu + tnu + tpu) == 0):
        return 1.
    return (2 * (fnu + fpu)) / (fn + fp + fnu + fpu + tnu + tpu)


def error_recall(fp, fn, fpu, fnu):
    if ((fnu + fpu) == 0) and ((fn + fp) == 0):
        return 1.
    return (fnu + fpu) / (fn + fp)


def error_precision(tpu, tnu, fpu, fnu):
    if ((fnu + fpu) == 0) and ((fnu + fpu + tpu + tnu) == 0):
        return 1.
    return (fnu + fpu) / (fnu + fpu + tpu + tnu)

def confusion_matrx(prediction, target):
    _check_ndarray(prediction)
    _check_ndarray(target)

    cm = m.ConfusionMatrix(prediction, target)
    return cm.tp, cm.tn, cm.fp, cm.fn, cm.n

def log_loss_sklearn(probabilities, target, labels=None):
    _check_ndarray(probabilities)
    _check_ndarray(target)

    if probabilities.shape[-1] != target.shape[-1]:
        probabilities = probabilities.reshape(-1, probabilities.shape[-1])
    else:
        probabilities = probabilities.reshape(-1)
    target = target.reshape(-1)
    return 0
    # return metrics.log_loss(target, probabilities, labels=labels)


def entropy(p, dim=-1, keepdims=False):
    # exactly the same as scipy.stats.entropy()
    return -np.where(p > 0, p * np.log(p), [0.0]).sum(axis=dim, keepdims=keepdims)


def _check_ndarray(obj):
    if not isinstance(obj, np.ndarray):
        raise ValueError("object of type '{}' must be '{}'".format(type(obj).__name__, np.ndarray.__name__))



ALL_METRICS = {
    "False Positive Rate": false_positive_rate,
    "Dice": dice,
    "Jaccard": jaccard,
    "Hausdorff Distance": hausdorff_distance,
    "Hausdorff Distance 95": hausdorff_distance_95,
    "Precision": precision,
    "Recall": recall,
    "Avg. Symmetric Surface Distance": avg_surface_distance_symmetric,
    "Avg. Surface Distance": avg_surface_distance,
    "Accuracy": accuracy,
    "False Omission Rate": false_omission_rate,
    "Negative Predictive Value": negative_predictive_value,
    "False Negative Rate": false_negative_rate,
    "True Negative Rate": true_negative_rate,
    "False Discovery Rate": false_discovery_rate,
    "Total Positives Test": total_positives_test,
    "Total Negatives Test": total_negatives_test,
    "Total Positives Reference": total_positives_reference,
    "total Negatives Reference": total_negatives_reference
}
