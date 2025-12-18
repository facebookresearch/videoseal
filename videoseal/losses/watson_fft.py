# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# adapted from https://github.com/SteffenCzolbe/PerceptualSimilarity

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-10

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import numpy as np


class RGB2YCbCr(nn.Module):
    def __init__(self):
        super().__init__()
        transf = torch.tensor([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]]).transpose(0, 1)
        self.transform = nn.Parameter(transf, requires_grad=False)
        bias = torch.tensor([0, 0.5, 0.5])
        self.bias = nn.Parameter(bias, requires_grad=False)
    
    def forward(self, rgb):
        N, C, H, W = rgb.shape
        assert C == 3
        rgb = rgb.transpose(1,3)
        cbcr = torch.matmul(rgb, self.transform)
        cbcr += self.bias
        return cbcr.transpose(1,3)


class ColorWrapper(nn.Module):
    """
    Extension for single-channel loss to work on color images
    """
    def __init__(self, lossclass, args, kwargs, trainable=False):
        """
        Parameters:
        lossclass: class of the individual loss functions
        trainable: bool, if True parameters of the loss are trained.
        args: tuple, arguments for instantiation of loss fun
        kwargs: dict, key word arguments for instantiation of loss fun
        """
        super().__init__()
        
        # submodules
        self.add_module('to_YCbCr', RGB2YCbCr())
        self.add_module('ly', lossclass(*args, **kwargs))
        self.add_module('lcb', lossclass(*args, **kwargs))
        self.add_module('lcr', lossclass(*args, **kwargs))
        
        # weights
        self.w_tild = nn.Parameter(torch.zeros(3), requires_grad=trainable)
        
    @property    
    def w(self):
        return F.softmax(self.w_tild, dim=0)
        
    def forward(self, input, target):
        # convert color space
        input = self.to_YCbCr(input)
        target = self.to_YCbCr(target)
        
        ly = self.ly(input[:,[0],:,:], target[:,[0],:,:])
        lcb = self.lcb(input[:,[1],:,:], target[:,[1],:,:])
        lcr = self.lcr(input[:,[2],:,:], target[:,[2],:,:])
        
        w = self.w
        
        return ly * w[0] + lcb * w[1] + lcr * w[2]


class GreyscaleWrapper(nn.Module):
    """
    Maps 3 channel RGB or 1 channel greyscale input to 3 greyscale channels
    """
    def __init__(self, lossclass, args, kwargs):
        """
        Parameters:
        lossclass: class of the individual loss function
        args: tuple, arguments for instantiation of loss fun
        kwargs: dict, key word arguments for instantiation of loss fun
        """
        super().__init__()
        
        # submodules
        self.add_module('loss', lossclass(*args, **kwargs))

    def to_greyscale(self, tensor):
        return tensor[:,[0],:,:] * 0.3 + tensor[:,[1],:,:] * 0.59 + tensor[:,[2],:,:] * 0.11

    def forward(self, input, target):
        (N,C,X,Y) = input.size()

        if N == 3:
            # convert input to greyscale
            input = self.to_greyscale(input)
            target = self.to_greyscale(target)

        # input in now greyscale, expand to 3 channels
        input = input.expand(N, 3, X, Y)
        target = target.expand(N, 3, X, Y)

        return self.loss.forward(input, target)


class Rfft2d(nn.Module):
    """
    Blockwhise 2D FFT
    for fixed blocksize of 8x8
    """
    def __init__(self, blocksize=8, interleaving=False):
        """
        Parameters:
        """
        super().__init__() # call super constructor
        
        self.blocksize = blocksize
        self.interleaving = interleaving
        if interleaving:
            self.stride = self.blocksize // 2
        else:
            self.stride = self.blocksize
        
        self.unfold = torch.nn.Unfold(kernel_size=self.blocksize, padding=0, stride=self.stride)
        return
        
    def forward(self, x):
        """
        performs 2D blockwhise DCT
        
        Parameters:
        x: tensor of dimension (N, 1, h, w)
        
        Return:
        tensor of dimension (N, k, b, b/2, 2)
        where the 2nd dimension indexes the block. Dimensions 3 and 4 are the block real FFT coefficients. 
        The last dimension is pytorches representation of complex values
        """
        
        (N, C, H, W) = x.shape
        assert (C == 1), "FFT is only implemented for a single channel"
        assert (H >= self.blocksize), "Input too small for blocksize"
        assert (W >= self.blocksize), "Input too small for blocksize"
        assert (H % self.stride == 0) and (W % self.stride == 0), "FFT is only for dimensions divisible by the blocksize"
        
        # unfold to blocks
        x = self.unfold(x)
        # now shape (N, 64, k)
        (N, _, k) = x.shape
        x = x.view(-1,self.blocksize,self.blocksize,k).permute(0,3,1,2)
        # now shape (N, #k, b, b)
        # perform DCT
        # coeff = fft.rfft(x)
        coeff = fft.rfft2(x)
        coeff = torch.view_as_real(coeff)
        
        return coeff / self.blocksize**2
    
    def inverse(self, coeff, output_shape):
        """
        performs 2D blockwhise inverse rFFT
        
        Parameters:
        output_shape: Tuple, dimensions of the outpus sample
        """
        if self.interleaving:
            raise Exception('Inverse block FFT is not implemented for interleaving blocks!')
        
        # perform iRFFT
        x = fft.irfft(coeff, dim=2, signal_sizes=(self.blocksize, self.blocksize))
        (N, k, _, _) = x.shape
        x = x.permute(0,2,3,1).view(-1, self.blocksize**2, k)
        x = F.fold(x, output_size=(output_shape[-2], output_shape[-1]), kernel_size=self.blocksize, padding=0, stride=self.blocksize)
        return x * (self.blocksize**2)

def softmax(a, b, factor=1):
    concat = torch.cat([a.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
    softmax_factors = F.softmax(concat * factor, dim=-1)
    return a * softmax_factors[:,:,:,:,0] + b * softmax_factors[:,:,:,:,1]

class WatsonDistanceFft(nn.Module):
    """
    Loss function based on Watsons perceptual distance.
    Based on FFT quantization
    """
    def __init__(self, blocksize=8, trainable=False, reduction='sum'):
        """
        Parameters:
        blocksize: int, size of the Blocks for discrete cosine transform 
        trainable: bool, if True parameters of the loss are trained and dropout is enabled.
        reduction: 'sum' or 'none', determines return format
        """
        super().__init__()
        self.trainable = trainable
        
        # input mapping
        blocksize = torch.as_tensor(blocksize)
        
        # module to perform 2D blockwise rFFT
        self.add_module('fft', Rfft2d(blocksize=blocksize.item(), interleaving=False))
    
        # parameters
        self.weight_size = (blocksize, blocksize // 2 + 1)
        self.blocksize = nn.Parameter(blocksize, requires_grad=False)
        # init with uniform QM
        self.t_tild = nn.Parameter(torch.zeros(self.weight_size), requires_grad=trainable)
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=trainable) # luminance masking
        w = torch.tensor(0.2) # contrast masking
        self.w_tild = nn.Parameter(torch.log(w / (1- w)), requires_grad=trainable) # inverse of sigmoid
        self.beta = nn.Parameter(torch.tensor(1.), requires_grad=trainable) # pooling
        
        # phase weights
        self.w_phase_tild = nn.Parameter(torch.zeros(self.weight_size) -2., requires_grad=trainable)
        
        # dropout for training
        self.dropout = nn.Dropout(0.5 if trainable else 0)
        
        # reduction
        self.reduction = reduction
        if reduction not in ['sum', 'none']:
            raise Exception('Reduction "{}" not supported. Valid values are: "sum", "none".'.format(reduction))

    @property
    def t(self):
        # returns QM
        qm = torch.exp(self.t_tild)
        return qm
    
    @property
    def w(self):
        # return luminance masking parameter
        return torch.sigmoid(self.w_tild)
    
    @property
    def w_phase(self):
        # return weights for phase
        w_phase =  torch.exp(self.w_phase_tild)
        # set weights of non-phases to 0
        if not self.trainable:
            w_phase[0,0] = 0.
            w_phase[0,self.weight_size[1] - 1] = 0.
            w_phase[self.weight_size[1] - 1,self.weight_size[1] - 1] = 0.
            w_phase[self.weight_size[1] - 1, 0] = 0.
        return w_phase
    
    def forward(self, input, target):
        # fft
        c0 = self.fft(target)
        c1 = self.fft(input)
        
        N, K, H, W, _ = c0.shape
        
        # get amplitudes
        c0_amp = torch.norm(c0 + EPS, p='fro', dim=4)
        c1_amp = torch.norm(c1 + EPS, p='fro', dim=4)
        
        # luminance masking
        avg_lum = torch.mean(c0_amp[:,:,0,0])
        t_l = self.t.view(1, 1, H, W).expand(N, K, H, W)
        t_l = t_l * (((c0_amp[:,:,0,0] + EPS) / (avg_lum + EPS)) ** self.alpha).view(N, K, 1, 1)
        
        # contrast masking
        s = softmax(t_l, (c0_amp.abs() + EPS)**self.w * t_l**(1 - self.w))
        
        # pooling
        watson_dist = (((c0_amp - c1_amp) / s).abs() + EPS) ** self.beta
        watson_dist = self.dropout(watson_dist) + EPS
        watson_dist = torch.sum(watson_dist, dim=(1,2,3))
        watson_dist = watson_dist ** (1 / self.beta)
        
        # get phases
        c0_phase = torch.atan2( c0[:,:,:,:,1], c0[:,:,:,:,0] + EPS) 
        c1_phase = torch.atan2( c1[:,:,:,:,1], c1[:,:,:,:,0] + EPS)
        
        # angular distance
        phase_dist = torch.acos(torch.cos(c0_phase - c1_phase)*(1 - EPS*10**3)) * self.w_phase # we multiply with a factor ->1 to prevent taking the gradient of acos(-1) or acos(1). The gradient in this case would be -/+ inf
        phase_dist = self.dropout(phase_dist)
        phase_dist = torch.sum(phase_dist, dim=(1,2,3))
        
        # perceptual distance
        distance = watson_dist + phase_dist
        
        # reduce
        if self.reduction == 'sum':
            distance = torch.sum(distance)
        
        return distance
    
