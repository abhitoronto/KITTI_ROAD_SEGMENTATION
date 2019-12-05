"""
    The Main module
"""

import cv2
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
import pickle

import utils.utils as utils
import utils.img_utils as imutils

from models.reknetm1 import RekNetM1
from models.reknetm2 import RekNetM2
from models.lidcamnet_fcn import LidCamNetEarlyFusion, LidCamNetLateFusion

from data_processing.road_dataset import a2d2_dataset, a2d2_ip_input_file, a2d2_upsample_input_file, a2d2_output_file
from data_processing.data_processing import crossval_split_a2d2

from misc.losses import BCEJaccardLoss, CCEJaccardLoss
from misc.polylr_scheduler import PolyLR
from misc.transforms import (
    train_transformations_a2d2,
    valid_tranformations_a2d2,
)

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

# For reproducibility
torch.manual_seed(111)


def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Argument parser for the main module. Main module represents train procedure.")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to the root dir where will be stores models.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the a2d2 dataset dir which contains pickle files")
    parser.add_argument("--fold", type=int, default=1, help="Num of a validation fold.")
    
    # optimizer options
    parser.add_argument("--optim", type=str, default="SGD", help="Type of optimizer: SGD or Adam")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rates for optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optim.")
    
    # Scheduler options
    parser.add_argument("--scheduler", type=str, default="poly", help="Type of a scheduler for LR scheduling.")
    parser.add_argument("--step-st", type=int, default=5, help="Step size for StepLR schedule.")
    parser.add_argument("--milestones", type=str, default="30,70,90", help="List with milestones for MultiStepLR schedule.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma parameter for StepLR and MultiStepLR schedule.")
    parser.add_argument("--patience", type=int, default=5, help="Patience parameter for ReduceLROnPlateau schedule.")

    # model params
    parser.add_argument("--model-type", type=str, default="lcn_early", help="Type of model. Can be 'lcn_late' and 'lcn_early'.")
    parser.add_argument("--init-type", type=str, default="He", help="Initialization type. Can be 'He' or 'Xavier'.")
    parser.add_argument("--act-type", type=str, default="relu", help="Activation type. Can be ReLU, CELU or FTSwish+.")
    parser.add_argument("--enc-bn-enable", type=int, default=1, help="Batch normalization enabling in encoder module.")
    parser.add_argument("--dec-bn-enable", type=int, default=1, help="Batch normalization enabling in decoder module.")
    parser.add_argument("--skip-conn", type=int, default=0, help="Skip-connection in context module.")

    # other options
    parser.add_argument("--n-epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of examples per batch.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of loading workers.")
    parser.add_argument("--device-ids", type=str, default="0", help="ID of devices for multiple GPUs.")
    parser.add_argument("--alpha", type=float, default=0, help="Modulation factor for custom loss.")
    parser.add_argument("--status-every", type=int, default=1, help="Status every parameter.")
    
    args = parser.parse_args()

    #Console logger definition
    console_logger = logging.getLogger("console-logger")
    console_logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    console_logger.addHandler(ch)

    console_logger.info(args)

    # number of classes
    num_classes = 1

    # Model definition
    if args.model_type == "lcn_early":
        model = LidCamNetEarlyFusion(num_classes=num_classes, bn_enable=True)
        console_logger.info("Using LinCamNet-early as the model.")
    elif args.model_type == "lcn_late":
        model = LidCamNetLateFusion(num_classes=num_classes, bn_enable=True)
        console_logger.info("Using LinCamNet-late as the model.")
    else:
        raise ValueError("Unknown model type: {}".format(args.model_type))

    console_logger.info("Number of trainable parameters: {}".format(utils.count_params(model)[1]))

    # Move model to devices
    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    cudnn.benchmark = True

    # Loss definition
    # TODO: Find a reason for using this loss.
    loss = BCEJaccardLoss(alpha=args.alpha)

    dataset_path = Path(args.dataset_path)
    images = read_pickle_file(str(dataset_path / a2d2_ip_input_file))
    masks = read_pickle_file(str(dataset_path / a2d2_output_file))

    # Use data subset
    images = images[:-round(len(images) / 5)]
    masks = masks[:-round(len(masks) / 5)]

    # train-val splits for cross-validation by a fold
    ((train_imgs, train_masks), 
        (valid_imgs, valid_masks)) = crossval_split_a2d2(imgs_paths=images, masks_paths=masks, fold=args.fold)

    # Define training/validation/ dataset
    train_dataset = a2d2_dataset(img_paths=train_imgs, mask_paths=train_masks, transforms=train_transformations_a2d2())
    valid_dataset = a2d2_dataset(img_paths=valid_imgs, mask_paths=valid_masks, transforms=valid_tranformations_a2d2())
    # valid_fmeasure_datset = a2d2_dataset(img_paths=valid_imgs, mask_paths=valid_masks)

    # Create Data Loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=torch.cuda.device_count(), num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    console_logger.info("Train dataset length: {}".format(len(train_dataset)))
    console_logger.info("Validation dataset length: {}".format(len(valid_dataset)))

    # Optim definition
    if args.optim == "SGD":
        optim = SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum)
        console_logger.info("Uses the SGD optimizer with initial lr={0} and momentum={1}".format(args.lr, args.momentum))
    else:
        optim = Adam(params=model.parameters(), lr=args.lr)
        console_logger.info("Uses the Adam optimizer with initial lr={0}".format(args.lr))

    if args.scheduler == "step":
        lr_scheduler = StepLR(optimizer=optim, step_size=args.step_st, gamma=args.gamma)
        console_logger.info("Uses the StepLR scheduler with step={} and gamma={}.".format(args.step_st, args.gamma))
    elif args.scheduler == "multi-step":
        lr_scheduler = MultiStepLR(optimizer=optim, milestones=[int(m) for m in (args.milestones).split(",")], gamma=args.gamma)
        console_logger.info("Uses the MultiStepLR scheduler with milestones=[{}] and gamma={}.".format(args.milestones, args.gamma))
    elif args.scheduler == "rlr-plat":
        lr_scheduler = ReduceLROnPlateau(optimizer=optim, patience=args.patience, verbose=True)
        console_logger.info("Uses the ReduceLROnPlateau scheduler.")
    elif args.scheduler == "poly":
        lr_scheduler = PolyLR(optimizer=optim, num_epochs=args.n_epochs, alpha=args.gamma)
        console_logger.info("Uses the PolyLR scheduler.")
    else:
        raise ValueError("Unknown type of schedule: {}".format(args.scheduler))

    valid = utils.binary_validation_routine_a2d2

    utils.train_routine_a2d2(
        args=args,
        console_logger=console_logger,
        root=args.root_dir,
        model=model,
        criterion=loss,
        optimizer=optim,
        scheduler=lr_scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        fm_eval_dataset=None,
        validation=valid,
        fold=args.fold,
        num_classes=num_classes,
        n_epochs=args.n_epochs,
        status_every=args.status_every
    )


def read_pickle_file(filename: str):
    if Path(filename).is_file():
        with open(filename, 'rb') as handle:
            return pickle.load(handle)
    else:
        raise ValueError(f'Pickle file {filename} does not exist')


if __name__ == "__main__":
    main()