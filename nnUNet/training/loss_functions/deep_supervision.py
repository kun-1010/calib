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


from torch import nn
import torch
import numpy as np
from nnunet.utilities.nd_softmax import softmax_helper
from datetime import datetime
from time import time, sleep
import sys
from batchgenerators.utilities.file_and_folder_operations import *
import torch.nn.functional as F

class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l


class BratsDiceLoss(nn.Module):
    """Dice loss of Brats dataset
    Args:
        outputs: A tensor of shape [N, *]
        labels: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, nonSquared=False, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(BratsDiceLoss, self).__init__()
        self.nonSquared = nonSquared

    def forward(self, outputs, labels):
        # bring outputs into correct shape
        #4 label

        #kits
        # bg, kidney, tumor = outputs.chunk(3, dim=1)
        # s = bg.shape
        # bg = bg.view(s[0], s[2], s[3], s[4])
        # kidney = kidney.view(s[0], s[2], s[3], s[4])
        # tumor = tumor.view(s[0], s[2], s[3], s[4])
        #
        # # bring masks into correct shape
        # bgMask, kidneyMask, tumorMask = labels.chunk(3, dim=1)
        # s = bgMask.shape
        # bgMask = bgMask.view(s[0], s[2], s[3], s[4])
        # kidneyMask = kidneyMask.view(s[0], s[2], s[3], s[4])
        # tumorMask = tumorMask.view(s[0], s[2], s[3], s[4])
        #
        # # calculate losses
        # bgLoss = self.weightedDiceLoss(bg, bgMask, mean=0.03)
        # kidneyLoss = self.weightedDiceLoss(kidney, kidneyMask, mean=0.01)
        # tumorLoss = self.weightedDiceLoss(tumor, tumorMask, mean=0.01)
        #
        # return (bgLoss + kidneyLoss + tumorLoss) / 5
        #brats

        bg, nd, nt, et = outputs.chunk(4, dim=1)
        s = bg.shape
        bg = bg.view(s[0], s[2], s[3], s[4])
        nd = nd.view(s[0], s[2], s[3], s[4])
        nt = nt.view(s[0], s[2], s[3], s[4])
        et = et.view(s[0], s[2], s[3], s[4])
        #label = self.to_binary(labels[0])
        # bring masks into correct shape
        bgMask, ndMask, ntMask, etMask = labels.chunk(4, dim=1)
        s = bgMask.shape
        bgMask = bgMask.view(s[0], s[2], s[3], s[4])
        ndMask = ndMask.view(s[0], s[2], s[3], s[4])
        ntMask = ntMask.view(s[0], s[2], s[3], s[4])
        etMask = etMask.view(s[0], s[2], s[3], s[4])

        # calculate losses
        bgLoss = self.weightedDiceLoss(bg, bgMask, mean=0.03)
        ndLoss = self.weightedDiceLoss(nd, ndMask, mean=0.02)
        ntLoss = self.weightedDiceLoss(nt, ntMask, mean=0.01)
        etLoss = self.weightedDiceLoss(et, etMask, mean=0.01)

        return (bgLoss + ndLoss + etLoss + ntLoss) / 5

    def diceLoss(self, pred, target, nonSquared=False):
        return 1 - self.softDice(pred, target, nonSquared=nonSquared)


    def to_binary(self, y):
        shape = y.shape
        c = int(torch.max(y).item() + 1)
        out = torch.zeros([shape[0], c, shape[2], shape[3], shape[4]]).to(y.device.index)
        for i in range(c):
            out[:, i:i+1, :, :, :] = (y == i)
        return out


    def weightedDiceLoss(self, pred, target, smoothing=1, mean=0.01):

        mean = mean
        w_1 = 1 / mean ** 2
        w_0 = 1 / (1 - mean) ** 2

        pred_1 = pred
        target_1 = target
        pred_0 = 1 - pred
        target_0 = 1 - target

        intersection_1 = (pred_1 * target_1).sum(dim=(1, 2, 3))
        intersection_0 = (pred_0 * target_0).sum(dim=(1, 2, 3))
        intersection = w_0 * intersection_0 + w_1 * intersection_1

        union_1 = (pred_1).sum() + (target_1).sum()
        union_0 = (pred_0).sum() + (target_0).sum()
        union = w_0 * union_0 + w_1 * union_1

        dice = (2 * intersection + smoothing) / (union + smoothing)
        # fix nans
        dice[dice != dice] = dice.new_tensor([1.0])
        return 1 - dice.mean()

    def softDice(self, pred, target, smoothing=1, nonSquared=False):
        intersection = (pred * target).sum(dim=(1, 2, 3))
        if nonSquared:
            union = (pred).sum() + (target).sum()
        else:
            union = (pred * pred).sum(dim=(1, 2, 3)) + (target * target).sum(dim=(1, 2, 3))
        dice = (2 * intersection + smoothing) / (union + smoothing)

        # fix nans
        dice[dice != dice] = dice.new_tensor([1.0])

        return dice.mean()


class PromiseDiceLoss(nn.Module):
    """Dice loss of Brats dataset
    Args:
        outputs: A tensor of shape [N, *]
        labels: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, nonSquared=False, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(PromiseDiceLoss, self).__init__()
        self.nonSquared = nonSquared

    def forward(self, outputs, labels):
        # bring outputs into correct shape
        bg, nd = outputs.chunk(2, dim=1)
        s = bg.shape
        bg = bg.view(s[0], s[2], s[3], s[4])
        nd = nd.view(s[0], s[2], s[3], s[4])

        # bring masks into correct shape
        bgMask, ndMask = labels.chunk(2, dim=1)
        s = bgMask.shape
        bgMask = bgMask.view(s[0], s[2], s[3], s[4])
        ndMask = ndMask.view(s[0], s[2], s[3], s[4])

        # calculate losses
        bgLoss = self.weightedDiceLoss(bg, bgMask, mean=0.1)
        ndLoss = self.weightedDiceLoss(nd, ndMask, mean=0.02)

        return (bgLoss + ndLoss) / 5

    def diceLoss(self, pred, target, nonSquared=False):
        return 1 - self.softDice(pred, target, nonSquared=nonSquared)

    def weightedDiceLoss(self, pred, target, smoothing=1, mean=0.01):

        mean = mean
        w_1 = 1 / mean ** 2
        w_0 = 1 / (1 - mean) ** 2

        pred_1 = pred
        target_1 = target
        pred_0 = 1 - pred
        target_0 = 1 - target

        intersection_1 = (pred_1 * target_1).sum(dim=(1, 2, 3))
        intersection_0 = (pred_0 * target_0).sum(dim=(1, 2, 3))
        intersection = w_0 * intersection_0 + w_1 * intersection_1

        union_1 = (pred_1).sum() + (target_1).sum()
        union_0 = (pred_0).sum() + (target_0).sum()
        union = w_0 * union_0 + w_1 * union_1

        dice = (2 * intersection + smoothing) / (union + smoothing)
        # fix nans
        dice[dice != dice] = dice.new_tensor([1.0])
        return 1 - dice.mean()

    def softDice(self, pred, target, smoothing=1, nonSquared=False):
        intersection = (pred * target).sum(dim=(1, 2, 3))
        if nonSquared:
            union = (pred).sum() + (target).sum()
        else:
            union = (pred * pred).sum(dim=(1, 2, 3)) + (target * target).sum(dim=(1, 2, 3))
        dice = (2 * intersection + smoothing) / (union + smoothing)

        # fix nans
        dice[dice != dice] = dice.new_tensor([1.0])

        return dice.mean()

class ISBIDiceLoss(nn.Module):
    """Dice loss of Brats dataset
    Args:
        outputs: A tensor of shape [N, *]
        labels: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, nonSquared=False, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(ISBIDiceLoss, self).__init__()
        self.nonSquared = nonSquared

    def forward(self, outputs, labels):
        # bring outputs into correct shape
        bg, nd = outputs.chunk(2, dim=1)
        s = bg.shape
        bg = bg.view(s[0], s[2], s[3])
        nd = nd.view(s[0], s[2], s[3])

        # bring masks into correct shape
        bgMask, ndMask = labels.chunk(2, dim=1)
        s = bgMask.shape
        bgMask = bgMask.view(s[0], s[2], s[3])
        ndMask = ndMask.view(s[0], s[2], s[3])

        # calculate losses
        bgLoss = self.weightedDiceLoss(bg, bgMask, mean=0.1)
        ndLoss = self.weightedDiceLoss(nd, ndMask, mean=0.02)

        return (bgLoss + ndLoss) / 5

    def diceLoss(self, pred, target, nonSquared=False):
        return 1 - self.softDice(pred, target, nonSquared=nonSquared)

    def weightedDiceLoss(self, pred, target, smoothing=1, mean=0.01):

        mean = mean
        w_1 = 1 / mean ** 2
        w_0 = 1 / (1 - mean) ** 2

        pred_1 = pred
        target_1 = target
        pred_0 = 1 - pred
        target_0 = 1 - target

        intersection_1 = (pred_1 * target_1).sum(dim=(1, 2))
        intersection_0 = (pred_0 * target_0).sum(dim=(1, 2))
        intersection = w_0 * intersection_0 + w_1 * intersection_1

        union_1 = (pred_1).sum() + (target_1).sum()
        union_0 = (pred_0).sum() + (target_0).sum()
        union = w_0 * union_0 + w_1 * union_1

        dice = (2 * intersection + smoothing) / (union + smoothing)
        # fix nans
        dice[dice != dice] = dice.new_tensor([1.0])
        return 1 - dice.mean()

    def softDice(self, pred, target, smoothing=1, nonSquared=False):
        intersection = (pred * target).sum(dim=(1, 2, 3))
        if nonSquared:
            union = (pred).sum() + (target).sum()
        else:
            union = (pred * pred).sum(dim=(1, 2, 3)) + (target * target).sum(dim=(1, 2, 3))
        dice = (2 * intersection + smoothing) / (union + smoothing)

        # fix nans
        dice[dice != dice] = dice.new_tensor([1.0])

        return dice.mean()


class KickFlipLoss(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(KickFlipLoss, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        self.mse = nn.MSELoss()
        #self.dice_loss = ISBIDiceLoss()
        self.dice_loss = BratsDiceLoss()

    def forward(self, x, y, flip):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors
        #label = self.to_binary_2d(y[0])
        label = self.to_binary(y[0])
        label2 = softmax_helper(flip[0] + label)

        flip_loss = weights[0] * self.dice_loss(x[0], label2) + self.mse(label2, label)
        l = self.loss(x[0], y[0]) + flip_loss
        return l

    def to_binary(self, y):
        shape = y.shape
        c = int(torch.max(y).item() + 1)
        out = torch.zeros([shape[0], c, shape[2], shape[3], shape[4]]).to(y.device.index)
        for i in range(c):
            out[:, i:i+1, :, :, :] = (y == i)
        return out

    def to_binary_2d(self, y):
        shape = y.shape
        c = int(torch.max(y).item() + 1)
        out = torch.zeros([shape[0], c, shape[2], shape[3]]).to(y.device.index)
        for i in range(c):
            out[:, i:i + 1, :, :] = (y == i)
        return out

    def to_one(self, y):
        bg, ed, nt, et = y.chunk(4, dim=1)
        s = y.shape
        bg = (bg > 0.5).view(s[0], 1, s[2], s[3], s[4])
        ed = (ed > 0.5).view(s[0], 1, s[2], s[3], s[4])
        nt = (nt > 0.5).view(s[0], 1, s[2], s[3], s[4])
        et = (et > 0.5).view(s[0], 1, s[2], s[3], s[4])

        result = y.new_zeros((s[0], 1, s[2], s[3], s[4]))
        result[bg] = 0
        result[ed] = 1
        result[nt] = 2
        result[et] = 3
        return result

class HybridLoss(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(HybridLoss, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        self.mse = nn.MSELoss()
        self.dice_loss = BratsDiceLoss()
        self.log_file = None
        #self.dice_loss = ISBIDiceLoss()

    def forward(self, model, y, weight):
        x = model["seg_out"][0]
        calib = model["calib"]

        label = self.to_binary(y[0])
        label2 = torch.sigmoid(model["logits"])
        flip_loss = weight * self.dice_loss(x, label)+(1-weight) * self.dice_loss(x, label2)
        calib_loss = self._convert_prediction(y[0], calib[0], x)
        l = self.loss(x, y[0]) + flip_loss + calib_loss+self.hybrid_loss(model)
        return l

    def _binary_calibration(self, target, calibration, probability):
        # same as sklearn.calibration calibration_curve but with the bin_count returned
        n_bins = calibration.size()[0]
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
        #binids表示prob4194304在bins的哪一段8192
        binids = np.digitize(probability.cpu().detach().numpy(), bins) - 1

        bin_true = np.bincount(binids, weights=target.cpu().detach(), minlength=n_bins)
        bin_total = np.bincount(binids, minlength=n_bins)
        nonzero = bin_total != 0
        bin_true = torch.sigmoid(torch.from_numpy(bin_true).float().to(target.device.index))
        bin_total = torch.sigmoid(torch.from_numpy(bin_total).float().to(target.device.index))
        # loss = torch.mean(bin_true - calibration)
        cit = nn.MSELoss()
        loss = cit(calibration, bin_true/bin_total)
        # nonzero = bin_total != 0
        # prob_true = (bin_true[nonzero] / bin_total[nonzero])
        # prob_pred = (bin_sums[nonzero] / bin_total[nonzero])
        return loss

    def _convert_prediction(self, target, calibration, output):
        num_classes = output.shape[1]
        output_softmax = softmax_helper(output)
        target = target[:, 0]
        #[2,1,128,128,128] --> [2,128,128,128]
        probabilities = torch.zeros_like(target).unsqueeze(-1)
        #[2,128,128,128,1]
        ece = 0
        for c in range(1, num_classes-1):
            probabilities[..., 0] = output_softmax[:, c]
            sub_calibration = torch.sigmoid(calibration[:, c])
            ece_target = (target == c)
            ece += self._binary_calibration(ece_target.flatten(), sub_calibration.flatten(), probabilities.flatten())
        return ece

    def to_binary(self, y):
        shape = y.shape
        out = torch.zeros([shape[0], 4, shape[2], shape[3], shape[4]]).to(y.device.index)
        #out = torch.zeros([shape[0], 3, shape[2], shape[3], shape[4]]).to(y.device.index)
        out[:, 0:1, :, :, :] = (y == 0)
        out[:, 1:2, :, :, :] = (y == 1)
        out[:, 2:3, :, :, :] = (y == 2)
        out[:, 3:4, :, :, :] = (y == 3)
        # c = int(torch.max(y).item() + 1)
        # out = torch.zeros([shape[0], c, shape[2], shape[3], shape[4]]).to(y.device.index)
        # for i in range(c):
        #     out[:, i:i + 1, :, :, :] = (y == i)
        return out

    def to_binary_2d(self, y):
        shape = y.shape
        c = int(torch.max(y).item() + 1)
        out = torch.zeros([shape[0], c, shape[2], shape[3]]).to(y.device.index)
        for i in range(c):
            out[:, i:i + 1, :, :] = (y == i)
        return out

    def to_one(self, y):
        bg, ed, nt, et = y.chunk(4, dim=1)
        s = y.shape
        bg = (bg > 0.5).view(s[0], 1, s[2], s[3], s[4])
        ed = (ed > 0.5).view(s[0], 1, s[2], s[3], s[4])
        nt = (nt > 0.5).view(s[0], 1, s[2], s[3], s[4])
        et = (et > 0.5).view(s[0], 1, s[2], s[3], s[4])

        result = y.new_zeros((s[0], 1, s[2], s[3], s[4]))
        result[bg] = 0
        result[ed] = 1
        result[nt] = 2
        result[et] = 3

        return result
    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):

        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:

            timestamp = datetime.now()
            self.log_file = join('/home/dell/data/Dataset/Brats21/DATASET/nnUNet_trained_models', "dice_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)
    def hybrid_loss(self, model_out):

        x_rec = torch.sigmoid(model_out["logits"])
        labels = model_out['images']
        mask_label = model_out["label_mask"]
        pred_mask_region = model_out["pred_mask_region"]
        mask_labels = model_out["mask_labels"]
        contrast_pred_1 = model_out["contrast_pred_1"]
        contrast_pred_2 = model_out["contrast_pred_2"]
        pred_mask_region_position = model_out["pred_mask_region_position"]
        mask_region_position_label = model_out["mask_position_lables"]
        loss_rec = self.forward_loss_reconstruct_mask(x_rec, labels, mask_label, mask_value=0.0)
        # loss_rec = forward_loss_reconstruct(x_rec, labels)
        # loss_rec = mse_loss(x_rec, labels)
        loss_mask_region = self.forward_loss_mask(pred_mask_region, mask_labels)
        position_pred = (torch.sigmoid(pred_mask_region_position) > 0.5).float()
        position_pred_num_region = position_pred.sum(dim=-1)

        loss_consistency = (self.forward_loss_mask(pred_mask_region, position_pred_num_region.detach()) + nn.MSELoss()(position_pred_num_region, pred_mask_region.argmax(dim=-1).float().detach())) / 2

        loss_contrast = self.forward_constrast_loss(contrast_pred_1, contrast_pred_2)
        loss_position = self.forward_loss_mask_position(pred_mask_region_position, mask_region_position_label)

        loss = loss_rec + 0.1 * loss_mask_region + 0.1 * loss_position + 0.01 * loss_consistency + 0.1 * loss_contrast
        return loss

    def patchify(self, in_channels, imgs, patch_size):
        """
        imgs: (N, 4, D, H, W)
        x: (N, L, patch_size**3 *4)
        """
        p = patch_size[0]
        assert imgs.shape[3] == imgs.shape[4] and imgs.shape[2] % p == 0
        d = h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], in_channels, d, p, h, p, w, p))
        x = torch.einsum('ncdkhpwq->ndhwkpqc', x)
        x = x.reshape(shape=(imgs.shape[0], d * h * w, p**3 * in_channels))
        return x


    def forward_constrast_loss(self, x_i, x_j, temp=0.5):
        device = x_i.device
        batch_size = x_i.shape[0]
        temp = torch.tensor(temp).to(device)
        neg_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float()
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1),
                                  z.unsqueeze(0),
                                  dim=2)
        sim_ij = torch.diag(sim, batch_size)
        sim_ji = torch.diag(sim, -batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / temp)
        denom = neg_mask * torch.exp(sim / temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * batch_size)

    def forward_loss_reconstruct_mask(self, pred, labels, mask_image, mask_value=0.0):
        # pred (b c d w h)
        # pred = torch.einsum("")
        mask = (mask_image == mask_value).float()

        loss = (pred - labels) ** 2

        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_reconstruct(self, pred, labels):
        loss = (pred - labels) ** 2
        loss = loss.mean()  # [N, L], mean loss per patch
        return loss


    def forward_loss_similarity(self, pred_1, pred_2):
        loss_fct = nn.CrossEntropyLoss()
        device = pred_1.device
        cos = nn.CosineSimilarity(dim=-1)
        sim = cos(pred_1.unsqueeze(dim=1), pred_2.unsqueeze(dim=0))
        labels = torch.arange(sim.shape[0], dtype=torch.long).to(device)

        loss = loss_fct(sim, labels)

        return loss

    def forward_loss_mask_region(self, pred_bottom_feature, mask_labels):

        # pred_bottom_feature = einops.rearrange(pred_bottom_feature, "b d w h c->b (d w h) c")
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(pred_bottom_feature.reshape(-1, pred_bottom_feature.shape[-1]), mask_labels.reshape(-1))
        return loss

    def forward_loss_mask(self, pred_bottom_feature, mask_labels):
        mask_labels = mask_labels.long()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(pred_bottom_feature.reshape(-1, pred_bottom_feature.shape[-1]), mask_labels.reshape(-1))
        return loss

    def forward_loss_mask_region_patch(self, pred_bottom_feature_patch, mask_labels_patch):

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(pred_bottom_feature_patch.reshape(-1, pred_bottom_feature_patch.shape[-1]), mask_labels_patch.reshape(-1))
        return loss


    def forward_loss_mask_region_multi_label(self, pred_bottom_feature, mask_labels):
        loss_fct = nn.BCEWithLogitsLoss()
        mask_labels = mask_labels.float()
        loss = loss_fct(pred_bottom_feature, mask_labels)
        return loss


    def forward_loss_mask_position(self, pred_bottom_feature, mask_labels):
        mask_labels = mask_labels.float()
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(pred_bottom_feature, mask_labels)
        return loss


class Contrast(torch.nn.Module):
    def __init__(self, args, batch_size, temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:{args.local_rank}")
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(torch.device(f"cuda:{args.local_rank}")))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)


class Loss(torch.nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss().cuda()
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = Contrast(args, batch_size).cuda()
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons):
        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        total_loss = rot_loss + contrast_loss + recon_loss

        return total_loss, (rot_loss, contrast_loss, recon_loss)



class CalibflipLoss(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(CalibflipLoss, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        self.mse = nn.MSELoss()
        self.dice_loss = BratsDiceLoss()
        self.log_file = None
        #self.dice_loss = ISBIDiceLoss()

    def forward(self, x, y, flip, calib, weight):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        #label = self.to_binary_2d(y[0])
        label = self.to_binary(y[0])
        #flip_norm = torch.norm(flip[0])
        #flip[0] = 3*(flip[0]/flip_norm.clamp(min=1e-12))
        # label2 = softmax_helper(torch.tanh(flip[0]) + label)
        label2 = flip[0]
        #label2 = softmax_helper(torch.clamp(flip[0],-3,3) + label)

        #flip_loss = self.dice_loss(x[0], label2) + self.mse(label2, label)
        # att_loss = self.dice_loss(att[0],flip[0])
        flip_loss =(1-weight) * self.dice_loss(x[0], label2) + weight * self.mse(label2, label)
        calib_loss = self._convert_prediction(y[0], calib[0], x[0])
        #l = weight * self.loss(x[0], y[0]) + (1 - weight) * flip_loss + calib_loss
        l = self.loss(x[0], y[0]) + flip_loss + calib_loss
        dice = self.dice_loss(label2,label)
        # self.print_to_log_file("dice = %.4f" % dice)
        # for i in range(1, len(x)):
        #     if weights[i] != 0:
        #         calib_loss = self._convert_prediction(y[i], x[i], x[i])
        #         l += weights[i] * self.loss(x[i], y[i]) + weights[i] * calib_loss
        return l

    def _binary_calibration(self, target, calibration, probability):
        # same as sklearn.calibration calibration_curve but with the bin_count returned
        n_bins = calibration.size()[0]
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
        #binids表示prob4194304在bins的哪一段8192
        binids = np.digitize(probability.cpu().detach().numpy(), bins) - 1

        bin_true = np.bincount(binids, weights=target.cpu().detach(), minlength=n_bins)
        bin_total = np.bincount(binids, minlength=n_bins)
        nonzero = bin_total != 0
        bin_true = torch.sigmoid(torch.from_numpy(bin_true).float().to(target.device.index))
        bin_total = torch.sigmoid(torch.from_numpy(bin_total).float().to(target.device.index))
        # loss = torch.mean(bin_true - calibration)
        cit = nn.MSELoss()
        loss = cit(calibration, bin_true/bin_total)
        # nonzero = bin_total != 0
        # prob_true = (bin_true[nonzero] / bin_total[nonzero])
        # prob_pred = (bin_sums[nonzero] / bin_total[nonzero])
        return loss

    def _convert_prediction(self, target, calibration, output):
        num_classes = output.shape[1]
        output_softmax = softmax_helper(output)
        target = target[:, 0]
        #[2,1,128,128,128] --> [2,128,128,128]
        probabilities = torch.zeros_like(target).unsqueeze(-1)
        #[2,128,128,128,1]
        ece = 0
        for c in range(1, num_classes-1):
            probabilities[..., 0] = output_softmax[:, c]
            sub_calibration = torch.sigmoid(calibration[:, c])
            ece_target = (target == c)
            ece += self._binary_calibration(ece_target.flatten(), sub_calibration.flatten(), probabilities.flatten())
        return ece

    def to_binary(self, y):
        shape = y.shape
        out = torch.zeros([shape[0], 4, shape[2], shape[3], shape[4]]).to(y.device.index)
        #out = torch.zeros([shape[0], 3, shape[2], shape[3], shape[4]]).to(y.device.index)
        out[:, 0:1, :, :, :] = (y == 0)
        out[:, 1:2, :, :, :] = (y == 1)
        out[:, 2:3, :, :, :] = (y == 2)
        out[:, 3:4, :, :, :] = (y == 3)
        # c = int(torch.max(y).item() + 1)
        # out = torch.zeros([shape[0], c, shape[2], shape[3], shape[4]]).to(y.device.index)
        # for i in range(c):
        #     out[:, i:i + 1, :, :, :] = (y == i)
        return out

    def to_binary_2d(self, y):
        shape = y.shape
        c = int(torch.max(y).item() + 1)
        out = torch.zeros([shape[0], c, shape[2], shape[3]]).to(y.device.index)
        for i in range(c):
            out[:, i:i + 1, :, :] = (y == i)
        return out

    def to_one(self, y):
        bg, ed, nt, et = y.chunk(4, dim=1)
        s = y.shape
        bg = (bg > 0.5).view(s[0], 1, s[2], s[3], s[4])
        ed = (ed > 0.5).view(s[0], 1, s[2], s[3], s[4])
        nt = (nt > 0.5).view(s[0], 1, s[2], s[3], s[4])
        et = (et > 0.5).view(s[0], 1, s[2], s[3], s[4])

        result = y.new_zeros((s[0], 1, s[2], s[3], s[4]))
        result[bg] = 0
        result[ed] = 1
        result[nt] = 2
        result[et] = 3

        return result
    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):

        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:

            timestamp = datetime.now()
            self.log_file = join('/home/dell/data/Dataset/Brats21/DATASET/nnUNet_trained_models', "dice_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

class CaliblabelLoss(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(CaliblabelLoss, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        self.mse = nn.MSELoss()
        self.dice_loss = BratsDiceLoss()

    def forward(self, x, y, calib_out, calib):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        label2 = softmax_helper(calib_out[0])

        flip_loss = weights[0] * self.dice_loss(x[0], label2)

        calib_loss = self._convert_prediction(y[0], calib[0], calib_out[0])
        l = self.loss(x[0], y[0]) + flip_loss + calib_loss

        for i in range(1, len(x)):
            if weights[i] != 0:
                calib_loss = self._convert_prediction(y[i], x[i], x[i])
                l += weights[i] * self.loss(x[i], y[i]) + weights[i] * calib_loss
        return l

    def _binary_calibration(self, target, calibration, probability):
        # same as sklearn.calibration calibration_curve but with the bin_count returned
        n_bins = calibration.size()[0]
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
        binids = np.digitize(probability.cpu().detach().numpy(), bins) - 1

        bin_true = np.bincount(binids, weights=target.cpu().detach(), minlength=n_bins)
        bin_total = np.bincount(binids, minlength=n_bins)
        nonzero = bin_total != 0
        bin_true = torch.sigmoid(torch.from_numpy(bin_true).float().to(target.device.index))
        bin_total = torch.sigmoid(torch.from_numpy(bin_total).float().to(target.device.index))
        # loss = torch.mean(bin_true - calibration)
        cit = nn.MSELoss()
        loss = cit(calibration, bin_true/bin_total)
        # nonzero = bin_total != 0
        # prob_true = (bin_true[nonzero] / bin_total[nonzero])
        # prob_pred = (bin_sums[nonzero] / bin_total[nonzero])
        return loss

    def _convert_prediction(self, target, calibration, output):
        num_classes = output.shape[1]
        output_softmax = softmax_helper(output)
        target = target[:, 0]
        probabilities = torch.zeros_like(target).unsqueeze(-1)
        ece = 0
        for c in range(1, num_classes-1):
            probabilities[..., 0] = output_softmax[:, c]
            sub_calibration = torch.sigmoid(calibration[:, c])
            ece_target = (target == c)
            ece += self._binary_calibration(ece_target.flatten(), sub_calibration.flatten(), probabilities.flatten())
        return ece


class CalibLoss(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(CalibLoss, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        self.mse = nn.MSELoss()

    def forward(self, x, y, calib):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        calib_loss = self._convert_prediction(y[0], calib[0], x[0])
        l = self.loss(x[0], y[0]) + calib_loss

        return l

    def _binary_calibration(self, target, calibration, probability):
        # same as sklearn.calibration calibration_curve but with the bin_count returned
        n_bins = calibration.size()[0]
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
        binids = np.digitize(probability.cpu().detach().numpy(), bins) - 1

        bin_true = np.bincount(binids, weights=target.cpu().detach(), minlength=n_bins)
        bin_total = np.bincount(binids, minlength=n_bins)
        nonzero = bin_total != 0
        bin_true = torch.sigmoid(torch.from_numpy(bin_true).float().to(target.device.index))
        bin_total = torch.sigmoid(torch.from_numpy(bin_total).float().to(target.device.index))
        # loss = torch.mean(bin_true - calibration)
        cit = nn.MSELoss()
        loss = cit(calibration, bin_true/bin_total)
        # nonzero = bin_total != 0
        # prob_true = (bin_true[nonzero] / bin_total[nonzero])
        # prob_pred = (bin_sums[nonzero] / bin_total[nonzero])
        return loss

    def _convert_prediction(self, target, calibration, output):
        num_classes = output.shape[1]
        output_softmax = softmax_helper(output)
        target = target[:, 0]
        probabilities = torch.zeros_like(target).unsqueeze(-1)
        ece = 0
        for c in range(1, num_classes-1):
            probabilities[..., 0] = output_softmax[:, c]
            sub_calibration = torch.sigmoid(calibration[:, c])
            ece_target = (target == c)
            ece += self._binary_calibration(ece_target.flatten(), sub_calibration.flatten(), probabilities.flatten())
        return ece
