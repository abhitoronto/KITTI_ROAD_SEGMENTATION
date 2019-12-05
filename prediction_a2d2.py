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

import utils.utils as utils
from utils.img_utils import alpha_overlay, normalize

from data_processing.road_dataset import load_image_, numpy_to_tensor

from models.reknetm1 import RekNetM1
from models.reknetm2 import RekNetM2
from models.lidcamnet_fcn import LidCamNet

from misc.transforms import test_trasformations

def predict(models: nn.ModuleList, img_path, path2save, thresh=0.5):
    """
        Perfrom prediction for single image
        Params:
            models     : NN models
            img_path   : path to an image
            path2save  :
            thresh     : prediction threshold
    """

    img_path = Path(img_path)

    if not img_path.exists():
        raise FileNotFoundError("File '{}' not found.".format(str(img_path)))

    src_img = cv2.imread(str(img_path))

    transform = test_trasformations()
    augmented = transform(image=src_img)
    src_img = augmented["image"]

    img2predict = src_img.copy()
    img2predict = cv2.cvtColor(img2predict, cv2.COLOR_BGR2RGB).astype(dtype=np.float32)
    img2predict = normalize(img2predict)

    img2predict = utils.to_gpu(numpy_to_tensor(img2predict).unsqueeze(0).contiguous()).float()

    if len(models) == 1:
        #evaluate mode
        model = models[0].eval()

        with torch.set_grad_enabled(False):
            predict = model(img2predict)
    
        #Probs
        predict = F.sigmoid(predict).squeeze(0).squeeze(0)

        mask = (predict > thresh).cpu().numpy().astype(dtype=np.uint8)
        overlayed_img = alpha_overlay(src_img, mask)
    else:
        #Averaging all predictions for one point of test data
        sum_predicts = utils.to_gpu(torch.zeros((1, 1, src_img.shape[0], src_img.shape[1])).float())    
        
        for model in models:
            model.eval()
            with torch.set_grad_enabled(False):
                predict = model(img2predict)
            sum_predicts += F.sigmoid(predict)
        
        predict = (sum_predicts / len(models)).squeeze(0).squeeze(0).float()

        mask = (predict > thresh).cpu().numpy().astype(dtype=np.uint8)
        overlayed_img = alpha_overlay(src_img, mask)

    #save
    cv2.imwrite(path2save, overlayed_img)
    
    #show
    cv2.imshow("Predicted", overlayed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Image '{}' was processed successfully.".format(str(img_path)))
    

def predict_batch(models: nn.ModuleList, path2images, path2save, thresh=0.5):
    """
        Perfrom prediction for a batch images
        Params:
            models          : NN models
            path2images     : path to an image
            path2save       : should be a dir
            thresh          : prediction threshold
    """

    path2images = Path(path2images)
    path2save = Path(path2save)

    if not path2images.is_dir():
        raise RuntimeError("File '{}' is not dir.".format(str(path2images)))

    if not path2save.is_dir():
        raise RuntimeError("File '{}' is not dir.".format(str(path2save)))

    imgs_paths = sorted(list(path2images.glob("*")))

    count_processed = 0
    for idx, ip in enumerate(imgs_paths):
        src_img = cv2.imread(str(ip))

        transform = test_trasformations()
        augmented = transform(image=src_img)
        src_img = augmented["image"]

        img2predict = src_img.copy()
        img2predict = cv2.cvtColor(img2predict, cv2.COLOR_BGR2RGB).astype(dtype=np.float32)
        img2predict = normalize(img2predict)

        img2predict = utils.to_gpu(numpy_to_tensor(img2predict).unsqueeze(0).contiguous()).float()

        if len(models) == 1:
            model = models[0].eval()
            
            with torch.set_grad_enabled(False):
                predict = model(img2predict)
    
            #Probs
            predict = F.sigmoid(predict).squeeze(0).squeeze(0)

            mask = (predict > thresh).cpu().numpy().astype(dtype=np.uint8)
            overlayed_img = alpha_overlay(src_img, mask)
        else:
            #Averaging all predictions for one point of test data
            sum_predicts = utils.to_gpu(torch.zeros((1, 1, src_img.shape[0], src_img.shape[1])).float())    
        
            for model in models:
                model.eval()
                with torch.set_grad_enabled(False):
                    predict = model(img2predict)
                sum_predicts += F.sigmoid(predict)
        
            predict = (sum_predicts / len(models)).squeeze(0).squeeze(0).float()

            mask = (predict > thresh).cpu().numpy().astype(dtype=np.uint8)
            overlayed_img = alpha_overlay(src_img, mask)

        #save
        cv2.imwrite(str(path2save / "{}".format(ip.name)), overlayed_img)
    
        print("Image '{}' was processed successfully.".format(str(ip)))
        count_processed += 1
    
    print("{} images were processed.".format(count_processed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prediction module parameters.")
    parser.add_argument("--mode", type=str, default="single", help="Model of prediction. Can be 'single' of 'multiple'.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a model or models. If path is a dir then models from this dir will be averaged.")
    parser.add_argument("--model-type", type=str, default="reknetm1")
    parser.add_argument("--path2image", type=str, required=True, help="Path to a single image or dir of images.")
    parser.add_argument("--path2save", type=str, required=True, help="Path to save. Can be a single file or dir.")
    parser.add_argument("--thresh", type=float, default=0.5)

    args = parser.parse_args()

    model_path = Path(args.model_path)

    if args.model_type == "reknetm1":
        model = RekNetM1(num_classes=1, 
            ebn_enable=True, 
            dbn_enable=True, 
            upsample_enable=False, 
            act_type="celu",
            init_type="He")
        print("Uses RekNetM1 as the model.")
    elif args.model_type == "reknetm2":
        model = RekNetM2(num_classes=1,
            ebn_enable=True, 
            dbn_enable=True, 
            act_type="celu",
            upsample_enable=False, 
            init_type="He")
        print("Uses RekNetM2 as the model.")
    elif args.model_type == "lcn":
        model = LidCamNet(num_classes=1,
            bn_enable=False)
        print("Uses LidCamNet as the model.")
    else:
        raise ValueError("Unknown model type: {}".format(args.model_type))

    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=None).cuda()

    model_list = nn.ModuleList()
    if model_path.is_file():
        state = torch.load(str(model_path))
        model.load_state_dict(state["model"])
        model_list.append(model)
    else:
        models_paths = sorted(list(model_path.glob("*")))
        for mp in models_paths:
            p = mp / "model.pt"
            state = torch.load(str(p))
            model.load_state_dict(state["model"])
            model_list.append(model)

    if args.mode == "single":
        predict(models=model_list, img_path=args.path2image, path2save=args.path2save, thresh=args.thresh)
    elif args.mode == "multiple":
        predict_batch(models=model_list, path2images=args.path2image, path2save=args.path2save, thresh=args.thresh)
    else:
        raise ValueError("Unknown mode: {}".format(args.mode))
    