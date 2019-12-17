"""
    Module for image utils
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from pathlib import Path


class ImageSpecifications:
    """
        Class that represents specifications of an image
    """
    imageShape_max = (376, 1242)
    img_extension = ".png"

    # Class factors
    uu_cat = 50
    um_cat = 100
    umm_cat = 150


def alpha_overlay(img, gt_image, color=(0, 255, 0), alpha=0.5):
    """
        Method for visualizing mask above image. Part of this code was taken from https://github.com/ternaus/TernausNet/blob/master/Example.ipynb

        params:
            img         : source image
            gt_image    : ground truth image(mask)
            color       : mask color
            alpha       : overlaing parameter in equation: (1-alpha)*img + alpha*gt_image
    """
    # assert len(gt_image.shape) == len(img.shape), "Error: Ground truth and source images has different shapes."
    # assert gt_image.shape[0] == 1, "Error: Ground truth image has channels more then 1."

    gt_image = np.dstack((gt_image, gt_image, gt_image)) * np.array(color)
    gt_image = gt_image.astype(np.uint8)
    weighted_sum = cv2.addWeighted(gt_image, (1 - alpha), img, alpha, 0.)
    img2 = img.copy()

    if color == (255, 0, 0):
        channel_pos = 0
    elif color == (0, 255, 0):
        channel_pos = 1
    elif color == (0, 0, 255):
        channel_pos = 2
    else:
        raise ValueError("Wrong color: {}. Color should be 'red', 'green' or 'blue'.".format(color))

    ind = gt_image[:, :, channel_pos] > 0
    img2[ind] = weighted_sum[ind]
    return img2


def pad(img, required_size=ImageSpecifications.imageShape_max, background_value=0):
    """
        Padding for ground truth images
        
        params:
            img                 : image for padding
            required_size       : size after padding
            background_value    : 1 if 'img' is validation map then else 0
    """

    assert len(required_size) == 2, "required_size dimmention isn't equals 2."
    assert img.shape[0] <= required_size[0], "height of image greater then height of required_size."
    assert img.shape[1] <= required_size[1], "width of image greater then width of required_size."

    if (img.shape[0] == required_size[0]) and (img.shape[1] == required_size[1]):
        return img

    if background_value == 0:
        new_img = np.zeros(required_size, dtype=img.dtype)
        new_img[:img.shape[0], :img.shape[1]] = img
        return new_img
    elif background_value == 1:
        new_img = np.ones(required_size, dtype=img.dtype) * 255
        new_img[:img.shape[0], :img.shape[1]] = img
        return new_img
    else:
        raise ValueError("\'background_value\' is not valid: {}".format(background_value))


def normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = np.array(mean, dtype=np.float32) * 255
    std = np.array(std, dtype=np.float32) * 255
    denominator = np.reciprocal(std, dtype=np.float32)

    img2 = img.copy().astype(dtype=np.float32)
    img2 -= mean
    img2 *= denominator
    return img2


def UnNormalize_tensor(tensor: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: UnNormalized image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    return tensor.clamp(0.0, 1.0)


def getImageFromUnitTensor(tensor: torch.Tensor):
    """
        tensor is expected to be in CWH format
        where C = 3
    """
    tensor2pil = transforms.ToPILImage()
    image = tensor2pil(tensor)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def getMaskFromTensor(tensor: torch.Tensor):
    """
        This functions extracts the mask from the unthresholded output of a NN
        Args:
            tensor (Tensor): Tensor image of size (1, H, W)
        Returns:
            Tensor: UnNormalized image.
    """
    # Find sigmoid probability and apply a probability Threshold

    # tensor = F.relu(tensor)
    tensor = torch.sigmoid(tensor)

    return tensor


def getGroundTruth(fileNameGT, make_pad=False):
    """
        Returns the ground truth maps for roadArea and the validArea 
        
        param:
            fileNameGT  : ground truth file name
            make_pad    : 
    """
    # Read GT   
    assert fileNameGT.is_file(), "Cannot find: {}".format(fileNameGT)
    full_gt = cv2.imread(str(fileNameGT), 1)

    # attention: OpenCV reads in as BGR, so first channel has Blue / road GT
    roadArea = full_gt[:, :, 0] > 0
    validArea = full_gt[:, :, 2] > 0

    if not make_pad:
        return roadArea, validArea

    roadArea = roadArea.astype(dtype=np.uint8) * 255
    validArea = validArea.astype(dtype=np.uint8) * 255

    roadArea = pad(roadArea)
    validArea = pad(validArea, background_value=1)

    roadArea = (roadArea / 255).astype(dtype=np.bool)
    validArea = (validArea / 255).astype(dtype=np.bool)

    return roadArea, validArea
