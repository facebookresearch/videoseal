# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BCH error-correction for VideoSeal payloads.

VideoSeal embeds a fixed ``nbits`` message and the detector returns ``nbits``
*soft* logits. Bit-errors are common after compression / geometric / valuemetric
edits, and a single flipped bit makes an exact-ID match fail. ``BCHMessageCodec``
spends some of the raw payload on BCH parity so that fewer **data** bits are
carried but up to ``t`` bit-errors are silently corrected on decode — trading
payload capacity for exact-ID robustness (the same scheme Adobe's TrustMark uses).

Non-invasive: it wraps the message around the existing ``embed`` / ``detect`` API,
no change to the model's forward path.

    from videoseal.utils.bch import BCHMessageCodec
    codec = BCHMessageCodec(data_bits=64, total_bits=model.nbits)

    payload = codec.encode(my_msgs)                  # (B, data_bits) -> (B, nbits)
    imgs_w  = model.embed(imgs, msgs=payload, is_video=False)["imgs_w"]

    preds   = model.detect(imgs_w, is_video=False)["preds"]   # (B, 1+nbits)
    bits    = (preds[:, 1:] > 0).long()                       # hard nbits
    msgs, ok = codec.decode(bits)                             # (B, data_bits), (B,)

Requires ``bchlib`` (``pip install bchlib``).
"""

from __future__ import annotations

from typing import Tuple

import torch


def _bits_to_bytes(bits) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for i, v in enumerate(bits):
        if int(v) & 1:
            out[i // 8] |= 1 << (7 - i % 8)
    return bytes(out)


def _bytes_to_bits(data: bytes, n: int) -> list:
    return [(data[i // 8] >> (7 - i % 8)) & 1 for i in range(n)]


class BCHMessageCodec:
    """Pack ``data_bits`` of payload + BCH parity into a ``total_bits`` codeword.

    Picks the largest correction power ``t`` (over GF(2^m)) whose codeword still
    fits ``total_bits``; encode/decode operate on batched bit tensors.

    Args:
        data_bits:  number of usable data bits per message.
        total_bits: width of the model payload (``model.nbits``, e.g. 256).
        m:          Galois field order (n = 2^m - 1 code bits); 8 suits 256-bit.
    """

    def __init__(self, data_bits: int, total_bits: int, m: int = 8) -> None:
        import bchlib

        self.data_bits = int(data_bits)
        self.total_bits = int(total_bits)
        self.data_bytes = (self.data_bits + 7) // 8

        self.bch = None
        self.t = None
        for t in range(1, 2 ** m):
            try:
                cand = bchlib.BCH(t, m=m)
            except Exception:  # invalid (t, m)
                continue
            if (self.data_bytes + cand.ecc_bytes) * 8 <= self.total_bits:
                self.bch, self.t = cand, t  # keep the largest fitting t
            elif self.bch is not None:
                break  # ecc only grows with t
        if self.bch is None:
            raise ValueError(
                f"no BCH(m={m}) fits {data_bits} data bits in {total_bits} payload bits"
            )
        self.ecc_bytes = self.bch.ecc_bytes
        self.code_bits = (self.data_bytes + self.ecc_bytes) * 8

    def __repr__(self) -> str:
        return (f"BCHMessageCodec(data_bits={self.data_bits}, total_bits={self.total_bits}, "
                f"t={self.t}, code_bits={self.code_bits})")

    @torch.no_grad()
    def encode(self, msgs: torch.Tensor) -> torch.Tensor:
        """(B, data_bits) 0/1 tensor -> (B, total_bits) codeword (zero-padded)."""
        if msgs.dim() == 1:
            msgs = msgs.unsqueeze(0)
        out = []
        for row in msgs.tolist():
            data = _bits_to_bytes(row[: self.data_bits])
            data = (data + bytes(self.data_bytes))[: self.data_bytes]
            ecc = bytes(self.bch.encode(data))
            code = (_bytes_to_bits(data, self.data_bytes * 8)
                    + _bytes_to_bits(ecc, self.ecc_bytes * 8))
            out.append(code + [0] * (self.total_bits - len(code)))
        return torch.tensor(out, dtype=msgs.dtype, device=msgs.device)

    @torch.no_grad()
    def decode(self, bits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, total_bits) hard 0/1 tensor -> ((B, data_bits) msgs, (B,) bool ok).

        ``ok`` is False where BCH reported the codeword uncorrectable; the returned
        data bits are then best-effort (uncorrected).
        """
        if bits.dim() == 1:
            bits = bits.unsqueeze(0)
        db, eb = self.data_bytes, self.ecc_bytes
        msgs, oks = [], []
        for row in bits.tolist():
            data = bytearray(_bits_to_bytes(row[: db * 8])[:db])
            ecc = bytearray(_bits_to_bytes(row[db * 8: (db + eb) * 8])[:eb])
            try:
                nerr = self.bch.decode(data, ecc)
                self.bch.correct(data, ecc)
                ok = nerr is not None and nerr >= 0
            except Exception:  # uncorrectable
                ok = False
            msgs.append(_bytes_to_bits(bytes(data), self.data_bits))
            oks.append(bool(ok))
        return (torch.tensor(msgs, dtype=bits.dtype, device=bits.device),
                torch.tensor(oks, dtype=torch.bool, device=bits.device))
