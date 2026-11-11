# BCH error-correction for exact-ID

VideoSeal embeds a fixed `nbits` payload; the detector returns `nbits` soft
logits. A few bits flip after common edits (compression, resize/crop, geometric,
noise, blur, valuemetric), and a single flipped bit makes an **exact-ID** match
fail. `videoseal.utils.bch.BCHMessageCodec` spends part of the payload on BCH
parity so fewer **data** bits are carried but up to `t` extraction errors are
silently corrected — trading capacity for exact-ID robustness.

It is non-invasive: the codec wraps the message around the existing
`embed` / `detect` API, with no change to the model.

```python
import torch, videoseal
from videoseal.utils.bch import BCHMessageCodec

model = videoseal.load("videoseal")
codec = BCHMessageCodec(data_bits=64, total_bits=model.nbits)   # 64 user bits + parity

msgs    = torch.randint(0, 2, (1, codec.data_bits))
payload = codec.encode(msgs)                                    # (B, 64) -> (B, nbits)
imgs_w  = model.embed(imgs, msgs=payload, is_video=False)["imgs_w"]

preds   = model.detect(imgs_w, is_video=False)["preds"]         # (B, 1 + nbits)
bits    = (preds[:, 1:1 + model.nbits] > 0).long()
recovered, ok = codec.decode(bits)                              # (B, 64), (B,) bool
```

`BCHMessageCodec` picks the largest correction power `t` (GF(2^m), `m=8` by
default) whose codeword fits `total_bits`. For the 256-bit payload that is:

| Data bits | BCH `t` (errors corrected) |
|--:|--:|
| 100 | 19 |
| 64  | 24 |
| 40  | 27 |

Stronger correction (fewer data bits) buys more exact-ID robustness. In external
still-image benchmarking across 38 distortions, exact-ID rose from **21.6%** (raw
256-bit) to **73.2%** (40 data bits) at unchanged image quality — a better trade
than raising embed strength, which costs visual quality for less gain.

Requires `bchlib` (in the project dependencies).
