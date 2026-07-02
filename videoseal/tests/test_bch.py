# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test the BCH error-correction codec for VideoSeal payloads.

Verifies that BCHMessageCodec:
  - packs `data_bits` + parity into a `total_bits` codeword,
  - recovers the data bits exactly when <= t errors are introduced,
  - degrades gracefully (no crash, shape preserved) when errors exceed t.

Run with:
    python -m videoseal.tests.test_bch
"""

import torch

from videoseal.utils.bch import BCHMessageCodec


def test_roundtrip_corrects_up_to_t_errors():
    for data_bits in (100, 64, 40):
        codec = BCHMessageCodec(data_bits=data_bits, total_bits=256)
        assert codec.code_bits <= 256
        msgs = torch.randint(0, 2, (8, data_bits))
        code = codec.encode(msgs)
        assert code.shape == (8, 256)

        # flip exactly t bits per row, inside the codeword -> must recover exactly
        recv = code.clone()
        for b in range(recv.shape[0]):
            idx = torch.randperm(codec.code_bits)[: codec.t]
            recv[b, idx] ^= 1
        rec, ok = codec.decode(recv)
        assert torch.equal(rec, msgs), f"data_bits={data_bits}: rows not recovered"
        assert bool(ok.all())


def test_clean_roundtrip_is_identity():
    codec = BCHMessageCodec(data_bits=64, total_bits=256)
    msgs = torch.randint(0, 2, (4, 64))
    rec, ok = codec.decode(codec.encode(msgs))
    assert torch.equal(rec, msgs)
    assert bool(ok.all())


def test_exceeding_t_does_not_crash():
    codec = BCHMessageCodec(data_bits=64, total_bits=256)
    msgs = torch.randint(0, 2, (2, 64))
    recv = codec.encode(msgs) ^ 1  # flip everything -> uncorrectable
    rec, ok = codec.decode(recv)
    assert rec.shape == (2, 64)
    assert ok.dtype == torch.bool


def test_accepts_unbatched():
    codec = BCHMessageCodec(data_bits=40, total_bits=256)
    msg = torch.randint(0, 2, (40,))
    code = codec.encode(msg)
    assert code.shape == (1, 256)
    rec, ok = codec.decode(code)
    assert torch.equal(rec[0], msg)


if __name__ == "__main__":
    test_roundtrip_corrects_up_to_t_errors()
    test_clean_roundtrip_is_identity()
    test_exceeding_t_does_not_crash()
    test_accepts_unbatched()
    print("all BCH codec tests passed")
