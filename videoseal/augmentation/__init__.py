from .geometric import Crop, HorizontalFlip, Identity, Perspective, Resize, Rotate
from .valuemetric import JPEG, Brightness, Contrast, GaussianBlur, Hue, MedianFilter, Saturation
from .video import H264, H264rgb, H265

def get_validation_augs(
    is_video: bool = False
) -> list:
    """
    Get the validation augmentations.
    """
    augs = [
        (Identity,          [0]),  # No parameters needed for identity
        (HorizontalFlip,    [0]),  # No parameters needed for flip
        (Rotate,            [10, 30, 45, 90]),  # (min_angle, max_angle)
        (Resize,            [0.5, 0.75]),  # size ratio
        (Crop,              [0.5, 0.75]),  # size ratio
        (Perspective,       [0.2, 0.5, 0.8]),  # distortion_scale
        (Brightness,        [0.5, 1.5]),
        (Contrast,          [0.5, 1.5]),
        (Saturation,        [0.5, 1.5]),
        (Hue,               [-0.5, -0.25, 0.25, 0.5]),
        (JPEG,              [40, 60, 80]),
        (GaussianBlur,      [3, 5, 9, 17]),
        (MedianFilter,      [3, 5, 9, 17])
    ]
    if is_video:
        comps = [
            (H264, [32, 40, 48]),
            (H264rgb, [32, 40, 48]),
            (H265, [32, 40, 48])
        ]
        augs.extend(comps)
    return augs