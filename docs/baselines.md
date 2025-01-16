# Baselines

For all baselines used in the work, we provide the links to the original repositories and the specific models used in the evaluation.
To load them, you should create the torchscript files for the encoder and decoder and load them in the [`videoseal/models/baselines.py`](../videoseal/models/baselines.py) file.

> [!TIP]
> TorchScript is a way to create serializable and optimizable models from PyTorch code. Any model can be saved in TorchScript format by using `torch.jit.script(model)` and loaded with `torch.jit.load("model.pt")`, without the need to have the original model code.


## On H2

The baselines are stored in the following directory:
```
/checkpoint/pfz/projects/videoseal/baselines-watermarking
```

## Creating torchscript files

To create the torchscript files, you can use the following code:
```python
# Load the model
encoder_model = ...
decoder_model = ...

# Create the torchscript files
encoder_model = torch.jit.script(encoder_model)
decoder_model = torch.jit.script(decoder_model)

# Save the models
torch.jit.save(encoder_model, "encoder_model.pt")
torch.jit.save(decoder_model, "decoder_model.pt")
```

Issues will be raised in the different repositories since most of the models are not provided in torchscript format.
You will need to replace some code in the original repositories to save the models in torchscript format.

## Links to the original repositories

- [HiDDeN](https://github.com/facebookresearch/stable_signature/blob/main/hidden/notebooks/demo.ipynb), model used: 48bits replicate
- [CIN](https://github.com/rmpku/CIN), model used: cinNet&nsmNet
- [MBRS](https://github.com/jzyustc/MBRS), model used: EC_42.pth
- [TrustMark](https://github.com/adobe/trustmark/), model used: Q
- [WAM](https://github.com/facebookresearch/watermark-anything), model used: WAM trained on COCO
