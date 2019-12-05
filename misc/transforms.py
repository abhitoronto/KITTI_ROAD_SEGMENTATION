"""
    This module contains tranforms definitions for training, validating and tesing stages.
"""

import cv2
import numpy as np

from albumentations import (
    HorizontalFlip,
    PadIfNeeded,
    Normalize,
    Rotate,
    ToGray,
    RandomBrightnessContrast,
    CLAHE,
    RandomShadow,
    HueSaturationValue,
    IAASharpen,
    OneOf,
    Compose,
)

def train_transformations(prob=1.0):
    return Compose([
        PadIfNeeded(min_height=384, min_width=1280, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), always_apply=True),
        OneOf([HorizontalFlip(p=0.5), Rotate(limit=20, p=0.3)], p=0.5),
        OneOf([ToGray(p=0.3), 
            RandomBrightnessContrast(p=0.5), 
            CLAHE(p=0.5),
            IAASharpen(p=0.45)], p=0.5),
        RandomShadow(p=0.4),
        HueSaturationValue(p=0.3),
        Normalize(always_apply=True)], p=prob)


def valid_tranformations(prob=1.0):
    return Compose([
        PadIfNeeded(min_height=384, min_width=1280, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), always_apply=True),
        Normalize(always_apply=True)], p=prob)
 

def test_trasformations(prob=1.0):
    return Compose([
        PadIfNeeded(min_height=384, min_width=1280, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), always_apply=True)], p=prob)


def train_transformations_a2d2(prob=1.0):
    return Compose([
        PadIfNeeded(min_height=1208, min_width=1920, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0),
                    always_apply=True),
        OneOf([HorizontalFlip(p=0.5), Rotate(limit=20, p=0.3)], p=0.5),
        OneOf([HorizontalFlip(p=0.5), Rotate(limit=20, p=0.3)], p=0.5),
        OneOf([ToGray(p=0.3),
               RandomBrightnessContrast(p=0.5),
               CLAHE(p=0.5),
               IAASharpen(p=0.45)], p=0.5),
        RandomShadow(p=0.4),
        HueSaturationValue(p=0.3),
        Normalize(always_apply=True)], p=prob)


def valid_tranformations_a2d2(prob=1.0):
    return Compose([
        PadIfNeeded(min_height=1208, min_width=1920, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0),
                    always_apply=True),
        Normalize(always_apply=True)], p=prob)


def test_trasformations_a2d2(prob=1.0):
    return Compose([
        PadIfNeeded(min_height=1208, min_width=1920, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0),
                    always_apply=True)], p=prob)