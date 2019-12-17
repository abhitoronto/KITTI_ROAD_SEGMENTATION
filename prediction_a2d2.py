"""
    This module contains the code that perform prediction on a given image or images
"""

import cv2
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils.utils as utils
from utils.img_utils import alpha_overlay, normalize

from data_processing.road_dataset import load_rgb_image_, numpy_to_tensor, load_mask_
from data_processing.road_dataset import a2d2_dataset, a2d2_dataset_no_lidar

from models.lidcamnet_fcn import LidCamNet, LidCamNetLateFusion, LidCamNetEarlyFusion

from misc.transforms import valid_transformations_a2d2, \
                            train_transformations_a2d2, \
                            transform_normalize_lidar, \
                            transform_normalize_img
from misc.losses import BCEJaccardLoss, CCEJaccardLoss


# def predict(model_: nn.Module, img_path: str, gt_path: str, path2save: str, multi_input: bool, thresh: float=0.5):
#     """
#         Perform prediction for single image
#         Params:
#             model      : NN model
#             img_path   : path to an image
#             path2save  : Dir to save the image
#             thresh     : prediction threshold
#     """
#
#     img_path = Path(img_path)
#     gt_path = Path(gt_path)
#     path2save = Path(path2save)
#
#     if not img_path.exists():
#         raise FileNotFoundError("File '{}' not found.".format(str(img_path)))
#     if not gt_path.exists():
#         raise FileNotFoundError("File '{}' not found.".format(str(gt_path)))
#     if not path2save.is_dir():
#         raise RuntimeError("File '{}' is not dir.".format(str(path2save)))
#
#     dest_path = str(path2save / img_path.name)
#     gt_file_name = 'gt_' + str(img_path.name)
#     dest_path_gt = str(path2save) + '/' + gt_file_name
#
#     src_img = cv2.imread(str(img_path))
#     gt_mask = load_mask_(str(gt_path))
#
#     transform = test_trasformations_a2d2()
#     augmented = transform(image=src_img)
#     src_img = augmented["image"]
#
#     img2predict = src_img.copy()
#     img2predict = cv2.cvtColor(img2predict, cv2.COLOR_BGR2RGB).astype(dtype=np.float32)
#     img2predict = normalize(img2predict)
#
#     img2predict = utils.to_gpu(numpy_to_tensor(img2predict).unsqueeze(0).contiguous()).float()
#
#     with torch.set_grad_enabled(False):
#         predict = model_(img2predict)
#
#     #Probs
#     predict = F.sigmoid(predict).squeeze(0).squeeze(0)
#
#     mask = (predict > thresh).cpu().numpy().astype(dtype=np.uint8)
#     overlayed_img = alpha_overlay(src_img, mask)
#     overlayed_gt_img = alpha_overlay(src_img, gt_mask)
#
#     #save
#     cv2.imwrite(dest_path, overlayed_img)
#     cv2.imwrite(dest_path_gt, overlayed_gt_img)
#
#     #show
#     #cv2.imshow("Predicted", overlayed_img)
#     #cv2.imshow("Target", overlayed_gt_img)
#     #cv2.waitKey(0)
#     #cv2.destroyAllWindows()
#
#     print("Image '{}' was processed successfully.".format(str(img_path)))


def predict_batch(model: nn.Module, paths2images: list, gt_paths: list, path2save: str, multi_input: bool, thresh=0.5):
    """
        Perfrom prediction for a batch images
        Params:
            model          : NN models
            path2images     : list of Paths to images
            path2save       : Directory to save images
            gt_path         :
            thresh          : prediction threshold
    """

    path2save = Path(path2save)

    if not path2save.is_dir():
        raise RuntimeError("File '{}' is not dir.".format(str(path2save)))

    if multi_input:
        dataset = a2d2_dataset(img_paths=paths2images, mask_paths=gt_paths,\
                               transform_image= valid_transformations_a2d2(),\
                               normalize_image= transform_normalize_img(),\
                               normalize_lidar= transform_normalize_lidar())
        loss = BCEJaccardLoss(alpha=0)
        dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

        _, target_images, output_images = utils.binary_validation_routine_a2d2(model, loss, dataloader, save_image=True)

        for idx, (img_t, img_o, gt) in enumerate(zip(target_images, output_images, gt_paths)):
            if not Path(gt).is_file():
                raise RuntimeError("'{}' is not a file.".format(str(gt)))

            t_name = str(path2save) + '/' + 'target_' + str(Path(gt).name)
            o_name = str(path2save) + '/' + 'prediction_' + str(Path(gt).name)

            cv2.imwrite(t_name, img_t)
            cv2.imwrite(o_name, img_o)

            print(f'{idx} images were processed.')
        metrics = utils.fmeasure_evaluation_a2d2(model, dataset)

    else:

        dataset = a2d2_dataset_no_lidar(img_paths=paths2images, mask_paths=gt_paths,
                               transform_image= valid_transformations_a2d2(),\
                               normalize_image= transform_normalize_img())
        loss = BCEJaccardLoss(alpha=0)
        dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

        _, target_images, output_images = utils.binary_validation_routine_a2d2_no_lidar(model, loss, dataloader, save_image=True)

        for idx, (img_t, img_o, gt) in enumerate(zip(target_images, output_images, gt_paths)):
            if not Path(gt).is_file():
                raise RuntimeError("'{}' is not a file.".format(str(gt)))

            t_name = str(path2save) + '/' + 'target_' + str(Path(gt).name)
            o_name = str(path2save) + '/' + 'prediction_' + str(Path(gt).name)

            cv2.imwrite(t_name, img_t)
            cv2.imwrite(o_name, img_o)

            print(f'{idx} images were processed.')
        metrics = utils.fmeasure_evaluation_a2d2_no_lidar(model, dataset)
    info_str = "-" * 30
    info_str += "\nMaxF: {0}".format(metrics["MaxF"])
    info_str += "\nAvgPrec: {0}".format(metrics["AvgPrec"])
    info_str += "\nPRE: {0}".format(metrics["PRE_wp"][0])
    info_str += "\nREC: {0}".format(metrics["REC_wp"][0])
    info_str += "\nFPR: {0}".format(metrics["FPR_wp"][0])
    info_str += "\nFNR: {0}\n".format(metrics["FNR_wp"][0])
    print(info_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prediction module parameters.")
    parser.add_argument("--mode", type=str, default="single", help="Model of prediction. Can be 'single' or 'multiple'.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a model or models. If path is a dir then models from this dir will be averaged.")
    parser.add_argument("--model-type", type=str, default="lcn")
    parser.add_argument("--path2image", type=str, required=True, help="Path to a single image or pickle file with images. (multiple)")
    parser.add_argument("--path2gt", type=str, required=True, help="Path to a single gt image or pickle file with images. (multiple)")
    parser.add_argument("--path2save", type=str, required=True, help="Directory to save the file")
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--num-images", type=int, default=10)

    args = parser.parse_args()

    model_path = Path(args.model_path)

    multi_input = False

    if args.model_type == "lcn":
        model = LidCamNet(num_classes=1,
            bn_enable=True)
        print("Uses LidCamNet as the model.")
    elif args.model_type == "lcn_early":
        model = LidCamNetEarlyFusion(num_classes=1,
            bn_enable=True)
        multi_input = True
        print("Uses LidCamNet early fusion as the model.")
    elif args.model_type == "lcn_late":
        model = LidCamNetLateFusion(num_classes=1,
            bn_enable=True)
        multi_input = True
        print("Uses LidCamNet late fusion as the model.")
    else:
        raise ValueError("Unknown model type: {}".format(args.model_type))

    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=None).cuda()

    if model_path.is_file():
        state = torch.load(str(model_path))
        model.load_state_dict(state["model"])
    else:
        raise ValueError('model-path is not a file')

    if args.mode == "single":
        predict_batch(model=model, paths2images=[[args.path2image, '', '', '']], gt_paths=[args.path2gt], path2save=args.path2save, thresh=args.thresh, multi_input=multi_input)
    elif args.mode == "multiple":
        # Read Pickle files for image paths and extract the test set
        img_paths = utils.read_pickle_file(str(args.path2image))
        gt_paths = utils.read_pickle_file(str(args.path2gt))
        img_paths = img_paths[-round(len(img_paths)/4):-round(len(img_paths)/4) + args.num_images]
        gt_paths = gt_paths[-round(len(gt_paths)/4):-round(len(gt_paths)/4) + args.num_images]

        predict_batch(model=model, paths2images=img_paths, gt_paths=gt_paths, path2save=args.path2save, thresh=args.thresh, multi_input=multi_input)
    else:
        raise ValueError("Unknown mode: {}".format(args.mode))