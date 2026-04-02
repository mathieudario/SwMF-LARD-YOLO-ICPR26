import cv2
import numpy as np
import torch
import torch.nn.functional as F


# Image properties to monitor
IMG_BRIGTHNESS = "brightness"
IMG_SATURATION = "saturation"
IMG_ENTROPY = "entropy"
IMG_EDGE_AMOUNT = "edge_amount"


def compute_img_mean_brightness(im: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean brightness level of the given image.

    Note:
        The luminance formula for grayscale conversion:
            (0.2989 * R + 0.5870 * G + 0.1140 * B)

    Args:
        im (torch.Tensor, [N, C, H, W]): The image to compute the metrics on.

    Returns:
        (torch.Tensor, [N,]) The mean brightness for every image of the batch.
    """
    # Make sure the im tensor is in [N, C, H, W] format
    if im.dim() == 3:
        im = im.unsqueeze(0)
    if im.dim() != 4:
        raise ValueError(f"Input image is expected to be 4-dimensional ([N,C,H,W]), received {im.dim()}-dim.")

    # Convert RGB to grayscale - by batch
    if im.shape[1] == 3:  # If image is RGB
        gray_im = 0.2989 * im[:, 0, :, :] + 0.5870 * im[:, 1, :, :] + 0.1140 * im[:, 2, :, :]
    else:  # If image is already grayscale
        gray_im = im

    # Compute average of all pixel values & divide by 255 to normalize to [0, 1]
    mean_brightness = gray_im.mean(dim=[1,2]).float() #/ 255.0

    return mean_brightness  # Return as torch.Tensor


def compute_img_mean_saturation(im: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean saturation level of the given image.

    Args:
        im (torch.Tensor): The image to compute the metrics on. [3, H, W]-torch or [H, W, 3]-numpy.
    
    Returns:
        (torch.Tensor, [N,]) The mean saturation level in [0, 1].
        The mean saturation level, in [0, 1].
    """
    # Make sure the im tensor is in [N, C, H, W] format
    if im.dim() == 3:
        im = im.unsqueeze(0)
    if im.dim() != 4:
        raise ValueError(f"Input image is expected to be 4-dimensional ([N,C,H,W]), received {im.dim()}-dim.")

    # Convert PyTorch tensor to NumPy array
    if im.shape[1] == 3:  # NCHW format
        im_np = im.permute(0, 2, 3, 1).numpy()  # Convert to HWC
    else:
        im_np = im.numpy()

    # Convert to HSV
    hsv_im = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2HSV) for im in im_np])

    # Extract saturation channel (S) and compute mean
    mean_saturation = torch.Tensor(hsv_im[:, :, :, 1].mean(axis=(1,2))).float() #/ 255.0
    
    return mean_saturation


def compute_img_mean_entropy(im: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean entropy of the given image.

    Args:
        im (torch.Tensor): Input image tensor. [N, 3, H, W], float value [0, 1].

    Returns:
        The mean entropy of the image, normalized by the maximum possible entropy.
    """
    # Make sure the im tensor is in [N, C, H, W] format
    if im.dim() == 3:
        im = im.unsqueeze(0)
    if im.dim() != 4:
        raise ValueError(f"Input image is expected to be 4-dimensional ([N,C,H,W]), received {im.dim()}-dim.")

    # Convert RGB to grayscale - by batch
    if im.shape[1] == 3:  # If image is RGB
        gray_im = 0.2989 * im[:, 0, :, :] + 0.5870 * im[:, 1, :, :] + 0.1140 * im[:, 2, :, :]
    else:  # If image is already grayscale
        gray_im = im

    # Convert to integer bins and compute histogram
    gray_im = gray_im * 255.0
    gray_im = gray_im.clamp(0, 255).round().to(torch.uint8)
    hist = torch.stack([torch.bincount(im.flatten(), minlength=256) for im in gray_im]).float()

    # Compute probabilities
    pixel_nb = gray_im[0].numel()
    prob = hist / pixel_nb + 1e-10  # Avoid log(0) by adding a small value

    # Compute entropy: H = -sum(p(i) * log2(p(i)))
    entropy = -torch.sum(prob * torch.log2(prob), dim=1)
    max_entropy = 8  # log2(256) = 8
    nrm_entropy = entropy / max_entropy

    return nrm_entropy  # Back to python floats


def compute_img_edge_amount(im: torch.Tensor) -> torch.Tensor:
    """
    Compute the edge amount of the given image.

    Args:
        im (torch.Tensor): Input image tensor. [3, H, W].
    
    Returns:
        The normalized edge amount, in [0, 1].
    """
    # Make sure the im tensor is in [N, C, H, W] format
    if im.dim() == 3:
        im = im.unsqueeze(0)
    if im.dim() != 4:
        raise ValueError(f"Input image is expected to be 4-dimensional ([N,C,H,W]), received {im.dim()}-dim.")

    # Convert RGB to grayscale - by batch
    if im.shape[1] == 3:  # If image is RGB
        gray_im = 0.2989 * im[:, 0, :, :] + 0.5870 * im[:, 1, :, :] + 0.1140 * im[:, 2, :, :]
    else:  # If image is already grayscale
        gray_im = im

    # Enforce 2D grayscale and float32
    gray_im = gray_im.float() * 255.0
    gray_im = gray_im.squeeze(1) if gray_im.dim() == 4 else gray_im

    # Apply Laplace filter (kernel size 3)
    laplace_kernel = torch.tensor([[1,  1, 1],
                                   [1, -8, 1],
                                   [1,  1, 1]], dtype=torch.float32, device=im.device)
    laplace_kernel = laplace_kernel.view(1, 1, 3, 3)  # Shape for conv2d
    filtered_im = F.conv2d(gray_im.unsqueeze(1), laplace_kernel, padding=1)
    filtered_im = filtered_im.squeeze(1)  # Remove channel dims

    # Count pixels with absolute value > 25 as edges
    edges_nb = (torch.abs(filtered_im) > 25).sum(dim=(1,2))
    total_nb = gray_im[0].numel()

    # Normalize edge amount to [0, 1]
    return edges_nb / total_nb


#######################################
#           Image properties
#######################################

IMAGE_PROPERTIES = {
    IMG_BRIGTHNESS : compute_img_mean_brightness,
    IMG_SATURATION : compute_img_mean_saturation,
    IMG_ENTROPY    : compute_img_mean_entropy,
    IMG_EDGE_AMOUNT: compute_img_edge_amount,
}

