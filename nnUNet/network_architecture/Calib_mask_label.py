# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from einops import rearrange

from copy import deepcopy
from multiprocessing import pool
from typing import Sequence, Union
import torch
import torch.nn as nn
import numpy as np
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from nnunet.network_architecture.initialization import InitWeights_He
import copy
from nnunet.network_architecture.neural_network import SegmentationNetwork


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



def patchify(in_channels, imgs, patch_size):
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

def unpatchify(in_channels, x, patch_size, image_size):
    """
    x: (N, L, patch_size**3 *4)
    imgs: (N, 4, D, H, W)
    """
    p = patch_size[0]
    d, h, w = image_size
    assert h * w * d == x.shape[1]

    x = x.reshape(shape=(x.shape[0], d, h, w, p, p, p, in_channels))
    x = torch.einsum('ndhwkpqc->ncdkhpwq', x)
    imgs = x.reshape(shape=(x.shape[0], in_channels, d * p, h * p, h * p))
    return imgs

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

def mask_func(x, in_channels, mask_ratio, patch_size, image_size, mask_value=0.0):
    batch = x.shape[0]
    x_patch = patchify(in_channels, x, patch_size)

    mask_patch, mask, id = random_masking(x_patch, mask_ratio)
    mask_tokens = torch.ones(1, 1, in_channels * patch_size[0] * patch_size[1] * patch_size[2]) * mask_value
    device = x.device
    mask_tokens = mask_tokens.repeat(batch,  id.shape[1] - mask_patch.shape[1], 1)
    mask_tokens = mask_tokens.to(device)

    x_ = torch.cat([mask_patch, mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(x_, dim=1, index=id.unsqueeze(-1).repeat(1, 1, mask_patch.shape[2]))  # unshuffle
    # mask the input
    x = unpatchify(in_channels, x_, patch_size=patch_size, image_size=image_size)
    return x, mask

def get_region_nums(mask_nums, patches_of_region):
    assert mask_nums % patches_of_region == 0
    return mask_nums // patches_of_region, patches_of_region

def get_mask_labels(batch_size, num_regions, mask, mask_region_patches, device):
    mask_labels = []
    for b in range(batch_size):
        mask_label_b = []
        for i in range(num_regions):
            mask_label_b.append(mask[b, i*mask_region_patches: (i+1)*mask_region_patches].sum().item())
        mask_labels.append(mask_label_b)
    mask_labels = torch.tensor(mask_labels, device=device).long()

    return mask_labels

def get_mask_labelsv2(batch_size, num_regions, mask, mask_region_patches, device):
    mask_labels = torch.zeros(batch_size, num_regions, mask_region_patches).to(device)
    for b in range(batch_size):
        for i in range(len(mask[b])):
            region_i = i // mask_region_patches
            patch_i = i % mask_region_patches
            mask_labels[b, region_i, patch_i] = mask[b, i]
    return mask_labels

def get_random_patch(img,
                     downsample_scale,
                     mask_labels,
                     patches_of_region):

    device = img.device
    batch_size = img.shape[0]
    in_channels = img.shape[1]
    d, w, h = img.shape[2], img.shape[3], img.shape[4]
    patch_scale = (d // downsample_scale[0], w // downsample_scale[1], h // downsample_scale[2])
    img = rearrange(img, "b c (p f) (q g) (o h) -> b (f g h) (c p q o)",
                    p=downsample_scale[0], q=downsample_scale[1], o=downsample_scale[2],
                    f=patch_scale[0], g=patch_scale[1], h=patch_scale[2])
    rec_patchs = torch.zeros(img.shape[0],
                             in_channels,
                             downsample_scale[0],
                             downsample_scale[1],
                             downsample_scale[2],
                             device=device)
    index = []
    mask_labels_cpu = mask_labels.cpu().numpy()

    for b in range(batch_size):
        no_all_mask_patches = np.argwhere(mask_labels_cpu[b] < patches_of_region).reshape(-1)
        # get the random patch index
        random_rec_patch_index = no_all_mask_patches[np.random.randint(0, len(no_all_mask_patches))]
        index.append(random_rec_patch_index)
        rec_patchs[b] = rearrange(img[b, random_rec_patch_index], "(c p q o) -> c p q o",
                                  c=in_channels,
                                  p=downsample_scale[0],
                                  q=downsample_scale[1],
                                  o=downsample_scale[2])

    return rec_patchs, index


def to_binary(y):
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


def get_random_patch_new(img,
                     downsample_scale,):

    device = img.device
    batch_size = img.shape[0]
    in_channels = img.shape[1]
    patch_images = patchify(in_channels, img, downsample_scale)
    num_patchs = patch_images.shape[1]

    rec_patchs = torch.zeros(img.shape[0],
                             in_channels,
                             downsample_scale[0],
                             downsample_scale[1],
                             downsample_scale[2],
                             device=device)
    index = []

    for b in range(batch_size):
        # get the random patch index
        p_sum = 0
        while p_sum == 0:
            random_index = np.random.randint(0, num_patchs)
            random_patch = patch_images[b, random_index]
            p_sum = random_patch.sum()

            random_patch = random_patch.reshape(shape=(downsample_scale[0], downsample_scale[1], downsample_scale[2], in_channels))
            random_patch = torch.einsum("hpqc->chpq", random_patch)

        index.append(random_index)
        rec_patchs[b] = random_patch

    return rec_patchs, index


class TwoConv(nn.Sequential):
    """two convolutions."""
    def __init__(
            self,
            dim: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
        """
        super().__init__()

        conv_0 = Convolution(dim, in_chns, out_chns, act=act, norm=norm, dropout=dropout, padding=1)
        conv_1 = Convolution(dim, out_chns, out_chns, act=act, norm=norm, dropout=dropout, padding=1)
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)

class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    def __init__(
            self,
            dim: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            dropout: Union[float, tuple] = 0.0,
            pool_size=(2, 2, 2)
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
        """
        super().__init__()

        max_pooling = Pool["MAX", dim](kernel_size=pool_size)
        convs = TwoConv(dim, in_chns, out_chns, act, norm, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
            self,
            dim: int,
            in_chns: int,
            cat_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            halves: bool = True,
            pool_size = (2, 2, 2)
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            halves: whether to halve the number of channels during upsampling.
        """
        super().__init__()

        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(dim, in_chns, up_chns, pool_size, mode=upsample)
        self.convs = TwoConv(dim, cat_chns + up_chns, out_chns, act, norm, dropout)

    def forward(self, x: torch.Tensor, x_e: torch.Tensor):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
        dimensions = len(x.shape) - 2
        sp = [0] * (dimensions * 2)
        for i in range(dimensions):
            if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                sp[i * 2 + 1] = 1
        x_0 = torch.nn.functional.pad(x_0, sp, "replicate")

        x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        return x

class DeepUNet(SegmentationNetwork):
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
    use_this_for_batch_size_computation_3D = 520000000
    def cons_stages(self, pools, region):
        stage = [(copy.deepcopy(region[0]), copy.deepcopy(region[1]))]
        for pool in reversed(pools):
            for i, r in enumerate(region):
                region[i][0] = region[i][0] * pool[0]
                region[i][1] = region[i][1] * pool[1]
                region[i][2] = region[i][2] * pool[2]
            stage.append((copy.deepcopy(region[0]), copy.deepcopy(region[1])))

        return stage

    def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 4,
            features: Sequence[int] = (32, 32, 64, 128, 256),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            pool_size = ((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
            select_reconstruct_region=[[0, 0, 0], [8, 8, 8]], # 重构范围,
            first_level_region = (32, 32, 32),
            two_level_region = (16, 16, 16),
            pretrain=True,
    ):
        super().__init__()
        deepth = len(pool_size)
        self.deepth = deepth
        self.in_channels = in_channels
        self.do_ds = False
        fea = features
        print(f"BasicUNet features: {fea}.")
        self.select_reconstruct_region = select_reconstruct_region
        self.stages = self.cons_stages(pool_size, select_reconstruct_region)
        print(f"self.stages is {self.stages}")
        self.pool_size_all = self.get_pool_size_all(pool_size)
        self.window_size = torch.tensor(first_level_region) // torch.tensor(self.pool_size_all)
        print(f"window size is {self.window_size}")
        self.pretrain = pretrain
        ## get patches of region
        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        self.drop = nn.Dropout()
        self.conv_0 = TwoConv(3, in_channels, features[0], act, norm, dropout)
        self.nonlin = nn.LeakyReLU
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = InitWeights_He(1e-2)
        self.conv_op = nn.Conv3d
        self.norm_op = nn.InstanceNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = 4
        self.final_nonlin = lambda x: x
        self._deep_supervision = True
        self.do_ds = True
        self.downs = nn.ModuleList([])

        self.convolutional_upsampling = True
        self.convolutional_pooling = True
        self.upscale_logits = False


        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        upsample_mode = 'trilinear'
        pool_op = nn.MaxPool3d
        calib_pool = nn.AvgPool3d
        transpconv = nn.ConvTranspose3d
        pool_op_kernel_sizes = [(2, 2, 2)] * 5
        conv_kernel_sizes = [(3, 3, 3)] * (5 + 1)
        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if self.conv_op == nn.Conv3d:
            self.max_num_features = self.MAX_NUM_FILTERS_3D
        else:
            self.max_num_features = self.MAX_FILTERS_2D

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

        output_features = 32
        input_features = 4
        input_calib = 4

        for d in range(5):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            basic_block=ConvDropoutNormNonlin
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, 2,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            self.calib_blocks_context.append(StackedConvLayers(input_calib, output_features, 2,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
                self.cb.append(calib_pool(pool_op_kernel_sizes[d]))
            input_features = output_features
            input_calib = output_features
            output_features = int(np.round(output_features * 2))

            output_features = min(output_features, self.max_num_features)

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

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[5]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[5]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        old_dropout_p = self.dropout_op_kwargs['p']
        self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(5):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != 5 - 1 and not self.convolutional_upsampling:
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
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))
            self.kick_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(self.conv_blocks_localization[ds][-1].output_channels, 4,
                                            1, 1, 0, 1, 1, False))
            self.kick_outputs.append(nn.Conv3d(self.kick_blocks_localization[ds][-1].output_channels, 4,
                                             1, 1, 0, 1, 1, False))
        for ds in range(len(self.calib_blocks_context)):
            self.calib_outputs.append(nn.Conv3d(self.calib_blocks_context[ds].output_channels, 4,
                                              1, 1, 0, 1, 1, False))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(5 - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

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

        for d in range(deepth):
            self.downs.append(Down(3, fea[d], fea[d+1], act=act, norm=norm, pool_size=pool_size[d]))

        self.ups = nn.ModuleList([])
        for d in range(deepth):
            self.ups.append(UpCat(3, fea[deepth-d], fea[deepth-d-1], fea[deepth-d-1], act, norm, dropout, pool_size=pool_size[deepth-d-1], upsample=upsample))

        self.decoder_pred = nn.Conv3d(fea[0], out_channels, 1, 1)

        if pretrain:
            bottom_feature = features[-1]
            self.pred_mask_region = nn.Linear(bottom_feature, 9)# 一个region 4个 patch
            self.contrast_learning_head = nn.Linear(bottom_feature, 384)
            self.pred_mask_region_position = nn.Linear(bottom_feature, 8)

    def get_pool_size_all(self, pool_size):
        p_all = [1, 1, 1]
        for p in pool_size:
            p_all[0] = p_all[0] * p[0]
            p_all[1] = p_all[1] * p[1]
            p_all[2] = p_all[2] * p[2]
        return p_all 

    def wrap_feature_selection(self, feature, region_box):
        # feature: b, c, d, w, h
        return feature[..., region_box[0][0]:region_box[1][0], region_box[0][1]:region_box[1][1], region_box[0][2]:region_box[1][2]]

    def get_local_images(self, images):
        images = self.wrap_feature_selection(images, region_box=self.stages[-1])
        return images

    def forward_encoder(self, x):
        x = self.conv_0(x)
        x_downs = [x]
        for d in range(self.deepth):
            x = self.downs[d](x)
            x_downs.append(x)
        return x_downs

    def forward_decoder(self, x_downs):
        x = self.wrap_feature_selection(x_downs[-1], self.stages[0])

        for d in range(self.deepth):
            x = self.ups[d](x, self.wrap_feature_selection(x_downs[self.deepth-d-1], self.stages[d+1]))
        logits = self.decoder_pred(x)
        return logits

    def forward(self, x):
        device = x[0].device
        images = x[0].detach()
        label = x[1][0].detach()
        label = to_binary(label)
        skips = []
        seg_outputs = []
        calib_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            images = self.conv_blocks_context[d](images)
            skips.append(images)
            if not self.convolutional_pooling:
                images = self.td[d](images)

        images = self.conv_blocks_context[-1](images)

        for u in range(len(self.tu)):
            images = self.tu[u](images)
            images = torch.cat((images, skips[-(u + 1)]), dim=1)
            images = self.conv_blocks_localization[u](images)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](images)))

        calib = seg_outputs[-1]
        for d in range(len(self.calib_blocks_context) - 1):
            calib = self.calib_blocks_context[d](calib)
            if not self.convolutional_pooling:
                calib = self.cb[d](calib)
            calib_outputs.append(self.calib_outputs[d](calib))

        local_images = self.get_local_images(label)
        if self.pretrain:
            # mask_ratio = torch.clamp(torch.rand(1), 0.4, 0.75)
            mask_ratio = 0.4
            label, mask = mask_func(label, self.in_channels, mask_ratio, (16, 16, 16), (6, 6, 6), mask_value=0.0)
            region_mask_labels = get_mask_labels(label.shape[0], 3*3*3, mask, 2*2*2, device)
            region_mask_position = get_mask_labelsv2(label.shape[0], 3*3*3, mask, 2*2*2, device=device)

            label_mask = self.wrap_feature_selection(label, region_box=self.stages[-1])

        hidden_states_out = self.forward_encoder(label)
        logits = self.forward_decoder(hidden_states_out)  

        if self.pretrain:
            # print(hidden_states_out.shape)
            classifier_hidden_states = rearrange(hidden_states_out[-1], "b c (d m) (w n) (h l) -> b c d w h (m n l)", m=self.window_size[0], n=self.window_size[1], l=self.window_size[2])
            classifier_hidden_states = classifier_hidden_states.mean(dim=-1)
            with torch.no_grad():
                hidden_states_out_2 = self.forward_encoder(label)
            encode_feature = hidden_states_out[-1]
            encode_feature_2 = hidden_states_out_2[-1]

            x4_reshape = encode_feature.flatten(start_dim=2, end_dim=4)
            x4_reshape = x4_reshape.transpose(1, 2)

            x4_reshape_2 = encode_feature_2.flatten(start_dim=2, end_dim=4)
            x4_reshape_2 = x4_reshape_2.transpose(1, 2)

            contrast_pred = self.contrast_learning_head(x4_reshape.mean(dim=1))
            contrast_pred_2 = self.contrast_learning_head(x4_reshape_2.mean(dim=1))

            pred_mask_feature = classifier_hidden_states.flatten(start_dim=2, end_dim=4)
            pred_mask_feature = pred_mask_feature.transpose(1, 2)
            mask_region_pred = self.pred_mask_region(pred_mask_feature)

            pred_mask_feature_position = classifier_hidden_states.flatten(start_dim=2, end_dim=4)
            pred_mask_feature_position = pred_mask_feature_position.transpose(1, 2)
            mask_region_position_pred = self.pred_mask_region_position(pred_mask_feature_position)

            if self._deep_supervision and self.do_ds:
                return_seg_out = tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
                return_calib = tuple([calib_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], calib_outputs[:-1][::-1])])
            else:
                return_seg_out = seg_outputs[-1]
                return_calib = calib_outputs[-1]

            return {
                "calib": return_calib,
                "seg_out": return_seg_out,
                "logits": logits,
                'images': local_images,
                "pred_mask_region": mask_region_pred,
                "pred_mask_region_position": mask_region_position_pred,
                "mask_position_lables": region_mask_position,
                "mask": mask,
                "label_mask": label_mask,
                "mask_labels": region_mask_labels,
                "contrast_pred_1": contrast_pred,
                "contrast_pred_2": contrast_pred_2,
            }
        else :
            return logits

