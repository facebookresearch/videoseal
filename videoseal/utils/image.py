import torch


def create_diff_img(img1, img2):
    """
    Create a difference image between two images.

    Parameters:
        img1 (torch.Tensor): The first image tensor of shape 3xHxW.
        img2 (torch.Tensor): The second image tensor of shape 3xHxW.

    Returns:
        torch.Tensor: The difference image tensor of shape 3xHxW.
    """
    diff = img1 - img2
    # diff = 0.5 + 10*(img1 - img2)
    # normalize the difference image
    diff = (diff - diff.min()) / ((diff.max() - diff.min()) + 1e-6)
    diff = 2*torch.abs(diff - 0.5)
    # diff = 20*torch.abs(diff)
    return diff.clamp(0, 1)


if __name__ == '__main__':
    pass
