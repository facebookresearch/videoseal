# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torchvision import transforms

from ..modules.jnd import JND
from .embedder import Embedder
from .extractor import Extractor
from .videoseal import Videoseal


class BaselineHiddenEmbedder(Embedder):
    def __init__(
        self,
        encoder_path: str,
        nbits: int = 48,
    ) -> None:
        super(BaselineHiddenEmbedder, self).__init__()
        self.encoder = torch.jit.load(encoder_path).eval()
        self.nbits = nbits
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.postprocess = transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return torch.randint(0, 2, (bsz, self.nbits))

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        msgs = 2 * msgs.float() - 1
        imgs = self.preprocess(imgs)
        imgs_w = self.encoder(imgs, msgs)
        imgs_w = self.postprocess(imgs_w)
        return imgs_w


class BaselineHiddenExtractor(Extractor):
    def __init__(
        self,
        decoder_path: str
    ) -> None:
        super(BaselineHiddenExtractor, self).__init__()
        self.decoder = torch.jit.load(decoder_path).eval()
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(
        self,
        imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            The extracted messages.
        """
        imgs = self.preprocess(imgs)
        msgs = self.decoder(imgs)  # b k
        # add +1 to the last dimension to make it b k+1 (WAM compatible)
        msgs = torch.cat([torch.zeros(msgs.size(0), 1).to(msgs.device), msgs], dim=1)
        return msgs


class BaselineMBRSEmbedder(Embedder):
    def __init__(
        self,
        encoder_path: str
    ) -> None:
        super(BaselineMBRSEmbedder, self).__init__()
        self.encoder = torch.jit.load(encoder_path).eval()
        self.nbits = 256
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.postprocess = transforms.Normalize(mean=[-1, -1, -1],std=[2, 2, 2])

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return torch.randint(0, 2, (bsz, self.nbits))

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        msgs = msgs.float()
        imgs_w = self.encoder(self.preprocess(imgs), msgs)
        imgs_w = self.postprocess(imgs_w)
        return imgs_w - imgs


class BaselineMBRSExtractor(Extractor):
    def __init__(
        self,
        decoder_path: str,
    ) -> None:
        super(BaselineMBRSExtractor, self).__init__()
        self.decoder = torch.jit.load(decoder_path).eval()
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def forward(
        self,
        imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            The extracted messages.
        """
        imgs = self.preprocess(imgs)
        msgs = 2* self.decoder(imgs) -1  # b k
        # add +1 to the last dimension to make it b k+1 (WAM compatible)
        msgs = torch.cat([torch.zeros(msgs.size(0), 1).to(msgs.device), msgs], dim=1)
        return msgs


class BaselineCINEmbedder(Embedder):
    def __init__(
        self,
        encoder_path: str
    ) -> None:
        super(BaselineCINEmbedder, self).__init__()
        self.encoder = torch.jit.load(encoder_path).eval()
        self.nbits = 30
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.postprocess = transforms.Normalize(mean=[-1, -1, -1],std=[2, 2, 2])

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return torch.randint(0, 2, (bsz, self.nbits))

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        msgs = msgs.float()
        imgs_w = self.encoder(self.preprocess(imgs), msgs)
        imgs_w = self.postprocess(imgs_w)
        return imgs_w - imgs


class BaselineCINExtractor(Extractor):
    def __init__(
        self,
        decoder_path: str,
    ) -> None:
        """
        CIN decoder:
        - works at resolution 128x128
        - outputs msgs ≈ between 0,1
        """
        super(BaselineCINExtractor, self).__init__()
        self.decoder = torch.jit.load(decoder_path).eval()
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def forward(
        self,
        imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            The extracted messages.
        """
        imgs = self.preprocess(imgs)
        # scale the output to be between -1,1
        msgs = 2 * self.decoder(imgs) - 1 # b k
        # add +1 to the last dimension to make it b k+1 (WAM compatible)
        msgs = torch.cat([torch.zeros(msgs.size(0), 1).to(msgs.device), msgs], dim=1)
        return msgs


class BaselineWAMEmbedder(Embedder):
    def __init__(
        self,
        encoder_path: str,
        nbits: int = 32,
    ) -> None:
        super(BaselineWAMEmbedder, self).__init__()
        self.encoder = torch.jit.load(encoder_path).eval()
        self.nbits = nbits
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.postprocess = transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return torch.randint(0, 2, (bsz, self.nbits))

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        imgs = self.preprocess(imgs)
        imgs_w = self.encoder(imgs, msgs)
        imgs_w = self.postprocess(imgs_w)
        return imgs_w


class BaselineWAMExtractor(Extractor):
    def __init__(
        self,
        decoder_path: str
    ) -> None:
        super(BaselineWAMExtractor, self).__init__()
        self.decoder = torch.jit.load(decoder_path).eval()
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(
        self,
        imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            The extracted messages.
        """
        imgs = self.preprocess(imgs)
        msgs = self.decoder(imgs)  # b 1+k h w
        msgs = msgs.mean(dim=[-2, -1])  # b 1+k
        return msgs


class BaselineTrustmarkEmbedder(Embedder):
    def __init__(
        self,
        encoder_path: str,
        nbits: int = 100,
    ) -> None:
        super(BaselineTrustmarkEmbedder, self).__init__()
        self.encoder = torch.jit.load(encoder_path).eval()
        self.nbits = nbits
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.postprocess = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])

    def get_random_msg(self, bsz: int = 1, nb_repetitions=1) -> torch.Tensor:
        return torch.randint(0, 2, (bsz, self.nbits))

    def forward(
        self,
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        msgs = msgs.float()
        imgs_w = self.encoder(self.preprocess(imgs), msgs)
        imgs_w = self.postprocess(imgs_w)
        return imgs_w - imgs


class BaselineTrustmarkExtractor(Extractor):
    def __init__(
        self,
        decoder_path: str
    ) -> None:
        super(BaselineTrustmarkExtractor, self).__init__()
        self.decoder = torch.jit.load(decoder_path).eval()
        # the network is trained with the following normalization
        self.preprocess = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def forward(
        self,
        imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            The extracted messages.
        """
        imgs = self.preprocess(imgs)
        msgs = self.decoder(imgs)  # b k
        # add +1 to the last dimension to make it b k+1 (WAM compatible)
        msgs = torch.cat([torch.zeros(msgs.size(0), 1).to(msgs.device), msgs], dim=1)
        return msgs


def build_baseline(
        method: str,
        attenuation: JND = None,
        scaling_w: float = 1.0,
        scaling_i: float = 1.0,
        img_size: int = 256,
        clamp: bool = True,
        chunk_size: int = 1,
        step_size: int = 1,
    ) -> Videoseal:
    assert os.path.exists(os.path.join('ckpts', 'hidden_encoder_48b.pt')), """
Please download the baseline models first.  
See docs/baselines.md for instructions, or run:
`pip install huggingface_hub; huggingface-cli download tangtianzhong/img-wm-torchscript --cache-dir .cache; mkdir ckpts; find .cache/models--tangtianzhong--img-wm-torchscript/snapshots/845dc751783db2a03a4b14ea600b0a4a9aba89aa -type l -exec cp --dereference {} ckpts/ \; sleep 5 ;rm -rf .cache`
"""
    if method == 'hidden':
        scaling_w = 0.2
        encoder_path = 'ckpts/hidden_encoder_48b.pt'
        decoder_path = 'ckpts/hidden_decoder_48b.pt'
        embedder = BaselineHiddenEmbedder(encoder_path)
        extractor = BaselineHiddenExtractor(decoder_path)
    elif method == 'mbrs':
        scaling_w = 1.0
        encoder_path = 'ckpts/mbrs_256_m256_encoder.pt'
        decoder_path = 'ckpts/mbrs_256_m256_decoder.pt'
        embedder = BaselineMBRSEmbedder(encoder_path)
        extractor = BaselineMBRSExtractor(decoder_path)
    elif method == 'cin':
        scaling_w = 1.0
        img_size = 128
        encoder_path = 'ckpts/cin_nsm_encoder.pt'
        decoder_path = 'ckpts/cin_nsm_decoder.pt'
        embedder = BaselineCINEmbedder(encoder_path)
        extractor = BaselineCINExtractor(decoder_path)
    elif method == 'wam':
        scaling_w = 2.0
        attenuation = JND(in_channels=1, out_channels=3, blue=True)
        encoder_path = 'ckpts/wam_encoder.pt'
        decoder_path = 'ckpts/wam_decoder.pt'
        embedder = BaselineWAMEmbedder(encoder_path)
        extractor = BaselineWAMExtractor(decoder_path)
    elif method == 'wam_noattenuation':
        scaling_w = 0.01
        encoder_path = 'ckpts/wam_encoder.pt'
        decoder_path = 'ckpts/wam_decoder.pt'
        embedder = BaselineWAMEmbedder(encoder_path)
        extractor = BaselineWAMExtractor(decoder_path)
    elif method == 'trustmark':
        scaling_w = 0.95  # set to 0.95 in the repo of TrustMark's authors
        encoder_path = 'ckpts/trustmark_encoder_q.pt'
        decoder_path = 'ckpts/trustmark_decoder_q.pt'
        embedder = BaselineTrustmarkEmbedder(encoder_path)
        extractor = BaselineTrustmarkExtractor(decoder_path)
    elif method == 'trustmark_scaling0p5':
        scaling_w = 0.5
        encoder_path = 'ckpts/trustmark_encoder_q.pt'
        decoder_path = 'ckpts/trustmark_decoder_q.pt'
        embedder = BaselineTrustmarkEmbedder(encoder_path)
        extractor = BaselineTrustmarkExtractor(decoder_path)
    else:
        raise ValueError(f'Unknown method: {method}')
    return Videoseal(
        embedder = embedder, 
        detector = extractor, 
        augmenter = None,
        attenuation = attenuation, 
        scaling_w = scaling_w, 
        scaling_i = scaling_i, 
        img_size = img_size, 
        clamp = clamp, 
        chunk_size = chunk_size,
        step_size = step_size,
    )



if __name__ == '__main__':
    # Test the baseline models
    pass