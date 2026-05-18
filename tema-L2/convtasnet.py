import lighter

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import permutations


class ConvTasNet(lighter.Model):
    def __init__(self, N, L, B, H, P, X, R, C, mask_nonlinear='relu'):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(ConvTasNet, self).__init__()

        self.mask_nonlinear = mask_nonlinear
        self.encoder = Encoder(L, N)
        self.separator = TemporalConvNet(N, B, H, P, X, R, C, mask_nonlinear)
        self.decoder = Decoder(N, L)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        """
        :param mixture: torch.Tensor of shape (B, T) where B is the batch size and T the time length
        """
        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        
        return est_source


class SI_SNR_PIT(nn.Module):
    def __init__(self, return_more_info=False, *args, **kwargs):
        super(SI_SNR_PIT, self).__init__(*args, **kwargs)

        self.EPS = 1e-7
        self.info = return_more_info
    
    def forward(self, sources_true, sources_pred):
        """
        :param sources_true: torch.Tensor, ground-truth separated sources (bs, C, T)
        :param sources_pred: torch.Tensor, predicted separated sources (bs, C, T)
        """
        
        bs, C, T = sources_true.size()

        # 1. Normalization
        mean_target = torch.mean(sources_true, dim=2, keepdim=True, dtype=float) 
        mean_estimate = torch.mean(sources_pred, dim=2, keepdim=True, dtype=float) 
        zero_mean_target = sources_true - mean_target
        zero_mean_estimate = sources_pred - mean_estimate

        # 2. SI-SNR with PIT
        s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
        s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS  # [B, 1, C, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + self.EPS)
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + self.EPS)  # [B, C, C]        

        # 3. Get max SNR between each predicted source and target source possible permutation
        perms = sources_true.new_tensor(list(permutations(range(C))), dtype=torch.long)
        # one-hot, [C!, C, C]
        index = torch.unsqueeze(perms, 2)
        perms_one_hot = sources_true.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
        # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
        snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr.float(), perms_one_hot.float()])
        max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
        # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
        max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
        max_snr /= C

        # 4. Compute loss and return sources
        loss = 0 - torch.mean(max_snr)
        reordered_estimateed_sources = self.reorder_source(sources_pred, perms, max_snr_idx)

        if self.info:
            return loss, max_snr, sources_pred, reordered_estimateed_sources
        else:
            return loss
    
    def get_mask(self, source, source_lengths):
        """
        :returns: mask [B, 1, T]
        """
        B, _, T = source.size()
        mask = source.new_ones((B, 1, T))
        for i in range(B):
            mask[i, :, source_lengths[i]:] = 0
        return mask

    def reorder_source(self, source, perms, max_snr_idx):
        """
        :param perms: [C!, C], permutations
        :param max_snr_idx: [B], each item is between [0, C!)
        """
        B, C, *_ = source.size()
        # [B, C], permutation whose SI-SNR is max of each utterance
        # for each utterance, reorder estimate source according this permutation
        max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
        # print('max_snr_perm', max_snr_perm)
        # maybe use torch.gather()/index_select()/scatter() to impl this?
        reorder_source = torch.zeros_like(source)
        for b in range(B):
            for c in range(C):
                reorder_source[b, c] = source[b, max_snr_perm[b][c]]
        return reorder_source


class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, mask_nonlinear='relu', causal=False, norm_type="gLN"):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in 1-D convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            mask_nonlinear: which non-linear function to generate mask
        """
        super(TemporalConvNet, self).__init__()

        self.C = C
        self.mask_nonlinear = mask_nonlinear
        
        layer_norm = GlobalLayerNorm(N)  # <- LN acts on the last dimension, but we have (bs, N, K) -> we need to transpose, then re-transpose
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)

        repeats = []
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2**x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks += [TemporalBlock(B, H, P, stride=1,
                                         padding=padding,
                                         dilation=dilation,
                                         norm_type=norm_type,
                                         causal=causal)]
                
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        # [bs, B, K] -> [bs, C*N, K]
        mask_conv1x1 = nn.Conv1d(B, C*N, 1, bias=False)
        # Put together
        self.network = nn.Sequential(layer_norm,
                                     bottleneck_conv1x1,
                                     temporal_conv_net,
                                     mask_conv1x1)

    def forward(self, mixture_w):
        '''
        :param mixture_w: torch.Tensor of shape (B, N, K)
        :returns: torch.Tensor of shape (B, C, N, K), where C is the number of sources
        '''
        bs, N, K = mixture_w.size()
        score = self.network(mixture_w)  # [bs, N, K] -> [bs, C*N, K]
        score = score.view(bs, self.C, N, K) # [bs C*N, K] -> [bs, C, N, K]
        if self.mask_nonlinear == 'softmax':
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            est_mask = F.relu(score)
        elif self.mask_nonlinear == 'sigmoid':
            est_mask = F.sigmoid(score)
        else:
            raise ValueError('Unsupported mask non-linear function')
        return est_mask


class Encoder(nn.Module):
    def __init__(self, L, N):
        """
        :param L: kernel/filter size
        :param N: number of kernels for Conv1d
        """
        super(Encoder, self).__init__()
        
        self.L, self.N = L, N
        self.conv1d = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)
        self.relu = nn.ReLU()

    def forward(self, mixture):
        """
        :param mixture: torch.Tensor of shape (bs, T), where bs is the batch size and T the n.o. samples
        :returns: torch.Tensor of shape (bs, N, K), where K is the new time dimension 
        """
        mixture = torch.unsqueeze(mixture, 1)  # (bs, 1, T)
        mixture_w = self.relu(self.conv1d(mixture))  # (bs, N, K)
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L):
        """
        :param N: same as Encoder
        :param L: same as Encoder
        """
        super(Decoder, self).__init__()
        
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        :param mixture_w: torch.Tensor of shape (bs, N, K) - the output of Encoder
        :param est_mask: torch.Tensor of shape (bs, C, N, K), where C is the number of sources
        :returns: torch.Tensor of shape (bs, C, T) - the estimated sources, with same length as input mixture
        """

        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # (bs, C, N, K)
        source_w = torch.transpose(source_w, 2, 3) # (bs, C, K, N)

        est_source = self.basis_signals(source_w)  # (bs, C, K, L) <- nn.Linear only applied over the last dimension!
        est_source = overlap_and_add(est_source, self.L//2) # (bs, C, T)
        
        return est_source


"""
ConvTas-Net utils taken from:

https://github.com/kaituoxu/Conv-TasNet/blob/master/src/utils.py

and

https://github.com/kaituoxu/Conv-TasNet/blob/master/src/conv_tasnet.py
"""

def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.

    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where

        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length

    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]
    
    subframe_length = math.gcd(frame_length, frame_step)  
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step).to(signal.device)
    frame = signal.new_tensor(frame).long() # .clone().detach()
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="gLN", causal=False):
        super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation, norm_type,
                                        causal)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        out = self.net(x)

        return out + residual


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="gLN", causal=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=False)
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm,
                                     pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm,
                                     pointwise_conv)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        return self.net(x)


class Chomp1d(nn.Module):
    """
    To ensure the output length is the same as the input.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: [M, H, Kpad]
        Returns:
            [M, H, K]
        """
        return x[:, :, :-self.chomp_size].contiguous()


# Modified with nn.LayerNorm and reshapes
class ChannelwiseLayerNorm(nn.Module):
    """
    Channel-wise Layer Normalization (cLN)

    By default, nn.LayerNorm acts on the last dimension, but we need it for the middle one.
    Solution: reshape -> layer norm -> reshape back
    """
    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channel_size)

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        :param y: torch.Tensor of shape (B, N, K)
        :return: normalized y on dimension 1
        """
        y_ = y.permute(0, 2, 1)
        y_ = self.norm(y_)

        return y_.permute(0, 2, 1)


def chose_norm(norm_type, channel_size):
    """The input of normlization will be (M, C, K), where M is batch size,
       C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    else: # norm_type == "BN":
        # Given input (B, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """

        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) # [B, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-7, 0.5) + self.beta
        return gLN_y