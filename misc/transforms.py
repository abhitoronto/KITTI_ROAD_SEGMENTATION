"""
    This module contains transforms definitions for training, validating and testing stages.
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


def transform_normalize_img(prob=1.0):
    return Compose([
        Normalize(always_apply=True, mean=[0.63263481, 0.63265741, 0.62899464], std=[0.25661512, 0.25698695, 0.2594808])], p=prob)

def train_transformations_a2d2(prob=1.0):
    return Compose([
        PadIfNeeded(min_height=int(320), min_width=int(480), border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0),
                    always_apply=True),
        OneOf([HorizontalFlip(p=0.5), Rotate(limit=20, p=0.3)], p=0.5),
        OneOf([ToGray(p=0.3),
               RandomBrightnessContrast(p=0.5),
               CLAHE(p=0.5),
               IAASharpen(p=0.45)], p=0.5),
        RandomShadow(p=0.4),
        HueSaturationValue(p=0.3)], p=prob)

def transform_normalize_lidar(prob=1.0):
    return Compose([
        Normalize(always_apply=True, mean=[0.16190466, 0.16355008, 0.16611974], std=[0.25623018, 0.256974, 0.25811531])], p=prob)


def valid_transformations_a2d2(prob=1.0):
    return Compose([
        PadIfNeeded(min_height=int(320), min_width=int(480), border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0),
                    always_apply=True)], p=prob)


def test_transformations_a2d2(prob=1.0):
    return Compose([
        PadIfNeeded(min_height=int(320), min_width=int(480), border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0),
                    always_apply=True)], p=prob)