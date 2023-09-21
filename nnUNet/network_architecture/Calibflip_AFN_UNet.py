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


from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.AFN.modules import GAU, Super_multaffSegdecoder
import torch.nn.functional as F
from nnunet.network_architecture.AFN.ASCLoss import ASCLoss


def make_gt_to_affinity(gt_mask, tmp_size):
    # h, w = gt_mask.shape[2], gt_mask.shape[3] NCHW
    # gt_pad = F.pad(gt_mask,(tmp_size,tmp_size,tmp_size,tmp_size,0,0,0,0),"constant")
    affinity_gt = torch.zeros([gt_mask.shape[0], 8, gt_mask.shape[2], gt_mask.shape[3]])
    gt_pad = F.pad(gt_mask, [tmp_size, tmp_size, tmp_size, tmp_size], mode='constant', value=0)
    # print("gt_pad_shape_value",gt_pad.shape)
    gt_pad_bool = gt_pad.bool()
    gt_array_bool = gt_mask.bool()
    right = torch.bitwise_xor(gt_pad_bool[:, :, 2 * tmp_size:, tmp_size:-tmp_size], gt_array_bool)
    left = torch.bitwise_xor(gt_pad_bool[:, :, :-2 * tmp_size, tmp_size:-tmp_size], gt_array_bool)
    up = torch.bitwise_xor(gt_pad_bool[:, :, tmp_size:-tmp_size, :-2 * tmp_size], gt_array_bool)
    down = torch.bitwise_xor(gt_pad_bool[:, :, tmp_size:-tmp_size, 2 * tmp_size:], gt_array_bool)
    diag1 = torch.bitwise_xor(gt_pad_bool[:, :, 2 * tmp_size:, :-2 * tmp_size], gt_array_bool)  # right_up
    diag2 = torch.bitwise_xor(gt_pad_bool[:, :, 2 * tmp_size:, 2 * tmp_size:], gt_array_bool)  # right_down
    diag3 = torch.bitwise_xor(gt_pad_bool[:, :, :-2 * tmp_size, :-2 * tmp_size], gt_array_bool)  # left_up
    diag4 = torch.bitwise_xor(gt_pad_bool[:, :, :-2 * tmp_size, 2 * tmp_size:], gt_array_bool)  # left_up
    affinity_gt[:, 0, :, :] = 1 + (-1) * right[:, 0, :, :].type(affinity_gt.type())
    affinity_gt[:, 1, :, :] = 1 + (-1) * left[:, 0, :, :].type(affinity_gt.type())
    affinity_gt[:, 2, :, :] = 1 + (-1) * up[:, 0, :, :].type(affinity_gt.type())
    affinity_gt[:, 3, :, :] = 1 + (-1) * down[:, 0, :, :].type(affinity_gt.type())
    affinity_gt[:, 4, :, :] = 1 + (-1) * diag1[:, 0, :, :].type(affinity_gt.type())
    affinity_gt[:, 5, :, :] = 1 + (-1) * diag2[:, 0, :, :].type(affinity_gt.type())
    affinity_gt[:, 6, :, :] = 1 + (-1) * diag3[:, 0, :, :].type(affinity_gt.type())
    affinity_gt[:, 7, :, :] = 1 + (-1) * diag4[:, 0, :, :].type(affinity_gt.type())

    return affinity_gt


def make_gt_to_affinity_mult(gt_mask, tmp_size=3):
    # h, w = gt_mask.shape[2], gt_mask.shape[3] NCHW
    # gt_pad = F.pad(gt_mask,(tmp_size,tmp_size,tmp_size,tmp_size,0,0,0,0),"constant")
    affinity_gt = torch.zeros([gt_mask.shape[0], 8 * tmp_size, gt_mask.shape[2], gt_mask.shape[3], gt_mask.shape[4]])
    # for i_size in range(1, tmp_size + 1):
    mult_size = [1, 4, 7]
    for i in range(tmp_size):
        i_size = mult_size[i]
        gt_pad = F.pad(gt_mask, [i_size, i_size, i_size, i_size, i_size, i_size], mode='constant', value=0)
        gt_pad_bool = gt_pad.bool()
        gt_array_bool = gt_mask.bool()

        right = torch.bitwise_xor(gt_pad_bool[:, :, 2 * i_size:, i_size:-i_size, i_size:-i_size], gt_array_bool)
        left = torch.bitwise_xor(gt_pad_bool[:, :, :-2 * i_size, i_size:-i_size, i_size:-i_size], gt_array_bool)
        up = torch.bitwise_xor(gt_pad_bool[:, :, i_size:-i_size, :-2 * i_size, i_size:-i_size], gt_array_bool)
        down = torch.bitwise_xor(gt_pad_bool[:, :, i_size:-i_size, 2 * i_size:, i_size:-i_size], gt_array_bool)
        diag1 = torch.bitwise_xor(gt_pad_bool[:, :, 2 * i_size:, :-2 * i_size, i_size:-i_size], gt_array_bool)  # right_up
        diag2 = torch.bitwise_xor(gt_pad_bool[:, :, 2 * i_size:, 2 * i_size:, i_size:-i_size], gt_array_bool)  # right_down
        diag3 = torch.bitwise_xor(gt_pad_bool[:, :, :-2 * i_size, :-2 * i_size, i_size:-i_size], gt_array_bool)  # left_up
        diag4 = torch.bitwise_xor(gt_pad_bool[:, :, :-2 * i_size, 2 * i_size:, i_size:-i_size], gt_array_bool)  # left_down

        affinity_gt[:, 0 + i * 8, :, :, :] = 1 + (-1) * right[:, 0, :, :, :].type(affinity_gt.type())
        affinity_gt[:, 1 + i * 8, :, :, :] = 1 + (-1) * left[:, 0, :, :, :].type(affinity_gt.type())
        affinity_gt[:, 2 + i * 8, :, :, :] = 1 + (-1) * up[:, 0, :, :, :].type(affinity_gt.type())
        affinity_gt[:, 3 + i * 8, :, :, :] = 1 + (-1) * down[:, 0, :, :, :].type(affinity_gt.type())
        affinity_gt[:, 4 + i * 8, :, :, :] = 1 + (-1) * diag1[:, 0, :, :, :].type(affinity_gt.type())
        affinity_gt[:, 5 + i * 8, :, :, :] = 1 + (-1) * diag2[:, 0, :, :, :].type(affinity_gt.type())
        affinity_gt[:, 6 + i * 8, :, :, :] = 1 + (-1) * diag3[:, 0, :, :, :].type(affinity_gt.type())
        affinity_gt[:, 7 + i * 8, :, :, :] = 1 + (-1) * diag4[:, 0, :, :, :].type(affinity_gt.type())

    return affinity_gt


class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class Calibflip_AFN_UNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Calibflip_AFN_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            calib_pool = nn.AvgPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            calib_pool = nn.AvgPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.calib_blocks_context = []
        self.conv_blocks_localization = []
        self.kick_blocks_localization = []
        self.td = []
        self.tu = []
        self.cb = []
        self.kf = []
        self.seg_outputs = []
        self.calib_outputs = []
        self.kick_outputs = []

        output_features = base_num_features
        input_features = input_channels
        input_calib = num_classes

        filters = [32, 64, 128, 256, 320]
        reduce_dim = False
        reduce_filters = [32, 64, 128, 256, 320]

        self.bce_loss = nn.BCELoss()
        self.asc_loss = ASCLoss()
        self.skip_blocks = []
        for i in range(5):
            self.skip_blocks.append(GAU(filters[i], True, reduce_dim, reduce_filters[i]))
        use_fim = [True, True, True, True]
        up = [True, True, True, True]
        self.affinity = [[3, 9, 15], 3, 3, 3]
        self.affinity_supervised = [True, False, False, False]
        self.decoder = []
        for d in range(4):
            if d == 3:
                self.decoder.append(
                    Super_multaffSegdecoder(reduce_filters[d + 1], reduce_filters[d], reduce_filters[d], use_fim[d],
                                            up[d], affinity=self.affinity[d], bottom=True,
                                            fuse_layer=self.affinity_supervised[d]))
            else:
                self.decoder.append(
                    Super_multaffSegdecoder(reduce_filters[d + 1], reduce_filters[d], reduce_filters[d], use_fim[d],
                                            up[d], affinity=self.affinity[d], fuse_layer=self.affinity_supervised[d]))

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            self.calib_blocks_context.append(StackedConvLayers(input_calib, output_features, num_conv_per_stage,
                                                               self.conv_op, self.conv_kwargs, self.norm_op,
                                                               self.norm_op_kwargs, self.dropout_op,
                                                               self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                               first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
                self.cb.append(calib_pool(pool_op_kernel_sizes[d]))
            input_features = output_features
            input_calib = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)
        self.skip_blocks = nn.ModuleList(self.skip_blocks)
        self.decoder = nn.ModuleList(self.decoder)
        self.filters = filters
        self.reduce_filters = reduce_filters
        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
                self.kf.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))
                self.kf.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))
            self.kick_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))
            self.kick_outputs.append(conv_op(self.kick_blocks_localization[ds][-1].output_channels, num_classes,
                                             1, 1, 0, 1, 1, seg_output_use_bias))
        for ds in range(len(self.calib_blocks_context)):
            self.calib_outputs.append(conv_op(self.calib_blocks_context[ds].output_channels, num_classes,
                                              1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.kick_blocks_localization = nn.ModuleList(self.kick_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.calib_blocks_context = nn.ModuleList(self.calib_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.kf = nn.ModuleList(self.kf)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        self.calib_outputs = nn.ModuleList(self.calib_outputs)
        self.kick_outputs = nn.ModuleList(self.kick_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, x):
        skips = []
        seg_outputs = []
        kick_flip_outputs = []
        calib_outputs = []
        output_dict = {}
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)
        k = x
        bs, c, h, w, l = x.shape
        y = torch.zeros([bs, 1, h, w, l], device=x.device)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        feature_dict = {}
        # for i in range(3):
        x1 = self.skip_blocks[0](skips[0], y)
        x2 = self.skip_blocks[1](skips[1], y)
        x3 = self.skip_blocks[2](skips[2], y)
        x4 = self.skip_blocks[3](skips[3], y)
        x5 = self.skip_blocks[4](skips[4], y)

        x5_s = x5
        x5_b = x5

        x4_s, x4_b, s4_cls, b4_cls, _, Tfeature4, Ffeature4 = self.decoder[-1](x5_s, x5_b, x4)
        x3_s, x3_b, s3_cls, b3_cls, _, Tfeature3, Ffeature3 = self.decoder[-2](x4_s, x4_b, x3)
        x2_s, x2_b, s2_cls, b2_cls, _, Tfeature2, Ffeature2 = self.decoder[-3](x3_s, x3_b, x2)
        x1_s, x1_b, s1_cls, b1_cls, weight, Tfeature1, Ffeature1 = self.decoder[-4](x2_s, x2_b, x1)
        # output_dict['step_' + str(i) + '_output_mask'] = [s1_cls, s2_cls, s3_cls, s4_cls]
        # output_dict['step_' + str(i) + '_output_affinity'] = [b1_cls, b2_cls, b3_cls, b4_cls]
        # output_dict['step_' + str(i) + "weight"] = weight
        output_dict['_output_mask'] = [s1_cls, s2_cls, s3_cls, s4_cls]
        output_dict['_output_affinity'] = [b1_cls, b2_cls, b3_cls, b4_cls]
        output_dict["weight"] = weight
        # if i == 2:
        #     feature_dict['step_' + str(i) + "Tfeature"] = [Tfeature1, Tfeature2, Tfeature3, Tfeature4]
        #     feature_dict['step_' + str(i) + "Ffeature"] = [Ffeature1, Ffeature2, Ffeature3, Ffeature4]
        #     feature_dict['step_' + str(i) + "weight"] = weight
        y = s1_cls

        output_dict['output'] = y

        calib = seg_outputs[-1]
        for d in range(len(self.calib_blocks_context) - 1):
            calib = self.calib_blocks_context[d](calib)
            if not self.convolutional_pooling:
                calib = self.cb[d](calib)
            calib_outputs.append(self.calib_outputs[d](calib))

        for u in range(len(self.kf)):
            k = self.kf[u](k)
            k = torch.cat((k, skips[-(u + 1)]), dim=1)
            k = self.kick_blocks_localization[u](k)
            kick_flip_outputs.append(self.kick_outputs[u](k))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]), \
                   tuple([kick_flip_outputs[-1]] + [i(j) for i, j in
                                                    zip(list(self.upscale_logits_ops)[::-1],
                                                        kick_flip_outputs[:-1][::-1])]), \
                   tuple([calib_outputs[-1]] + [i(j) for i, j in
                                                zip(list(self.upscale_logits_ops)[::-1], calib_outputs[:-1][::-1])]), \
                   output_dict
        else:
            return seg_outputs[-1], kick_flip_outputs[-1], calib_outputs[-1], output_dict

    def compute_objective(self, output_dict, batch_dict):
        loss_dict = {}

        gt_mask = batch_dict[0]
        # gt_boundary = batch_dict['anno_boundary']
        d, h, w = gt_mask.shape[2], gt_mask.shape[3], gt_mask.shape[4]
        # need change gt to affinity
        total_loss = None
        lamba_asc = 5
        # lamba_asc = 6
        #for i in range(self.steps):
        pred_mask = output_dict['_output_mask']  # list
        # pred_boundary = output_dict['step_' + str(i) + '_output_boundary'] # list
        pred_affinity = output_dict['_output_affinity']  # list
        weight = output_dict["weight"]
        # with torch.no_grad():
        #    print("weight",torch.unique(weight))
        step_mask_loss = None
        for k in range(len(pred_mask)):
            inner_pred = pred_mask[k]
            if inner_pred is None:
                continue
            inner_pred = F.interpolate(inner_pred, (d, h, w), mode='trilinear', align_corners=True)
            mask_loss = self.bce_loss(inner_pred, gt_mask.float())
            if step_mask_loss is None:
                step_mask_loss = torch.zeros_like(mask_loss).to(mask_loss.device)
            step_mask_loss = step_mask_loss + mask_loss
        step_mask_loss = step_mask_loss / len(pred_mask)

        step_topo_loss = None
        for k in range(len(pred_affinity)):
            if self.affinity_supervised[k] is True:

                inner_pred = pred_affinity[k]
                if inner_pred is None:
                    continue

                gt_affintiy_or = gt_mask.clone()

                if isinstance(self.affinity[k], list) and len(self.affinity[k]) > 1:
                    gt_affinity = make_gt_to_affinity_mult(gt_affintiy_or, tmp_size=3)
                else:
                    gt_affinity = make_gt_to_affinity(gt_affintiy_or, tmp_size=2)

                gt_affinity = gt_affinity.to(device='cuda')

                inner_pred = F.interpolate(inner_pred, (d, h, w), mode='trilinear', align_corners=True)

                bce_loss = self.bce_loss(inner_pred, gt_affinity.float())
                if k == 0:
                    asc_loss = self.asc_loss(inner_pred, gt_affinity.float())

                    topo_loss = bce_loss + lamba_asc * asc_loss

                else:
                    topo_loss = bce_loss
                if step_topo_loss is None:
                    step_topo_loss = torch.zeros_like(topo_loss).to(topo_loss.device)
                step_topo_loss = step_topo_loss + topo_loss
        if step_topo_loss is not None:
            step_topo_loss = step_topo_loss / len(pred_affinity)
        else:
            step_topo_loss = torch.zeros_like(step_mask_loss).to(step_mask_loss.device)

        if total_loss is None:
            total_loss = torch.zeros_like(step_mask_loss).to(step_mask_loss.device)
        total_loss = total_loss + step_mask_loss + step_topo_loss

        loss_dict['total_loss'] = step_mask_loss + step_topo_loss

        loss_dict['total_loss'] = total_loss
        return loss_dict

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (
                    npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
