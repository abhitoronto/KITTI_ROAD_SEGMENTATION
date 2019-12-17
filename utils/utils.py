"""
    This module contains some auxilary functions
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import csv
import copy
import json
import random
import logging
from pathlib import Path
import pickle
from datetime import datetime

import numpy as np

from misc.tb_logger import Logger
from utils import img_utils
from misc.metrics import (
    batch_dice, 
    batch_jaccard,
    calculate_confusion_matrix_from_arrays,
    calculate_dices,
    calculate_jaccards,
    evalExp,
    pxEval_maximizeFMeasure
)

def count_params(model: nn.Module) -> (int, int):
    """
        Calculates the total and trainable parameters in model.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (total, trainable)


def to_gpu(x: torch.Tensor):
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def save_model(model_path: str, model, best_jaccard, best_dice, best_uu_metrics, best_um_metrics, best_umm_metrics,
               epoch):
    torch.save({"best_jaccard": best_jaccard, "best_dice": best_dice, "best_uu_metrics": best_uu_metrics,
                "best_um_metrics": best_um_metrics, "best_umm_metrics": best_umm_metrics, "epoch": epoch,
                "model": model}, model_path)

def save_model_a2d2(model_path: str, model, best_jaccard, best_dice,epoch):
        torch.save({"best_jaccard": best_jaccard, "best_dice": best_dice, "epoch": epoch,
                    "model": model}, model_path)


def make_info_string(sep=',', **kwargs):
    """
        Construct an information string in the following view: key1: value1[sep]key2: value2[sep][(keyN: valueN)]
        params:
            sep         : a separator between instances. Possible values: ',', '\n'
            **kwargs    : params
    """
    
    if sep not in [',', '\n']:
        ValueError("Wrong separator: {}. 'sep' must be: ',' or '\n'".format(sep))
    
    info_str = ""
    for key, value in kwargs.items():
        info_str += "{0}: {1}{2} ".format(key, value, sep)
    info_str = info_str[:-2] if info_str[-2] == ',' else info_str[:-1]
    return info_str


def save_runparams(params: dict, file_path: str):
    """
        Perform saving run parameters into file.
    """
    with open(str(file_path), "w") as file:
        file.write(json.dumps(params, indent=True, sort_keys=True))


def write2csv(data, file_name, type_of_header):
    """
        Perform writing info to a csv file.

        params:
            data            : list of data
            file_name       : name of file
            type_of_header  :
    """

    if type_of_header == "maxf":
        header = "epoch,uu_MaxF,um_MaxF,umm_MaxF,mMaxf".split(",")
    elif type_of_header == "avgprec":
        header = "epoch,uu_AvgPrec,um_AvgPrec,umm_AvgPrec,mAvgPrec".split(",")
    elif type_of_header == "prec":
        header = "epoch,uu_PRE,um_PRE,umm_PRE,mPRE".split(",")
    elif type_of_header == "rec":
        header = "epoch,uu_REC,um_REC,umm_REC,mREC".split(",")
    elif type_of_header == "loss":
        header = "epoch,train_loss,valid_loss".split(",")
    elif type_of_header == "jd":
        header = "epoch,Jaccard,DICE".split(",")
    else:
        raise ValueError("Unknown type of header: {}".format(type_of_header))

    data = [header] + data

    with open(file_name, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)


def train_routine(
    args,
    console_logger: logging.Logger,
    root: str, 
    model: nn.Module, 
    criterion, 
    optimizer,
    scheduler,
    train_loader, 
    valid_loader,
    fm_eval_dataset,
    validation,
    fold,
    num_classes=1,
    n_epochs=100,
    status_every=5):

    """
        General trainig routine.
        params:
            args                    : argument parser parameters for saving it
            console_logger          : logger object for logging
            root                    : root dir where stores trained models
            model                   : model for training
            criterion               : loss function
            optimizer               : SGD, Adam or other
            scheduler               : learning rate scheduler
            train_loader            :
            valid_loader            :
            fm_eval_dataset         : dataset for F-max evaluation
            validation              : validation routine
            fold                    : number of fold
            num_classes             : number of classes
            n_epochs                : number of training epochs
            status_every            : the parameter which controls the frequency of status printing
    """

    #Load model if it exists
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)

    model_root = root / args.model_type / 'model{}'.format(fold)
    model_root.mkdir(exist_ok=True, parents=True)

    #CSV
    csvs_path = model_root / 'csv'
    csvs_path.mkdir(exist_ok=True, parents=True)

    model_path = model_root / 'model.pt'
    logging_path = model_root / 'train.log'

    #run params saving
    save_runparams(vars(args), file_path=(model_root / 'rparams.txt'))
    
    #file logger definition
    file_logger = logging.getLogger("file-logger")
    file_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(logging_path), mode='w')
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    file_logger.addHandler(fh)

    #Logging to the TensorBoardX
    tbx_logger = Logger(log_dir=str(model_root / "tbxlogs"))

    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state["epoch"]
        best_jaccard = state["best_jaccard"]
        best_dice = state["best_dice"]
        best_uu_metrics = state["best_uu_metrics"]
        best_um_metrics = state["best_um_metrics"]
        best_umm_metrics = state["best_umm_metrics"]
        model.load_state_dict(state["model"]) 
        console_logger.info("\nModel '{0}' was restored. Best Jaccard: {1}, Best DICE: {2}, Epoch: {3}".format(str(model_path), best_jaccard, best_dice, epoch))
    else:
        epoch = 1
        best_jaccard = 0
        best_dice = 0
        best_uu_metrics = {"MaxF": 0, "AvgPrec": 0, "PRE": 0, "REC": 0}
        best_um_metrics = {"MaxF": 0, "AvgPrec": 0, "PRE": 0, "REC": 0}
        best_umm_metrics = {"MaxF": 0, "AvgPrec": 0, "PRE": 0, "REC": 0}
    
    n_epochs = n_epochs + epoch
    best_model = copy.deepcopy(model.state_dict())

    train_losses = []
    valid_losses = []
    jaccards = []
    dices = []

    #CSV data for logging
    maxf_csv_data = []
    avgprec_csv_data = []
    prec_csv_data = []
    rec_csv_data = []
    loss_csv_data = []
    jacc_dice_csv_data = []

    for epoch in range(epoch, n_epochs):
        
        epoch_train_losses = []

        #Train mode
        model.train()
        
        #scheduler step
        scheduler.step()

        try:
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = to_gpu(inputs)
                targets = to_gpu(targets)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(targets, outputs)
                    loss.backward()
                    optimizer.step()
                epoch_train_losses.append(loss.item())

            #Train loss per epoch
            epoch_train_loss = np.mean(epoch_train_losses).astype(dtype=np.float64)
            
            #Validation
            valid_dict = validation(model, criterion, valid_loader)
            uu_metrics, um_metrics, umm_metrics = fmeasure_evaluation([model], fm_eval_dataset)

            train_losses.append(epoch_train_loss)
            valid_losses.append(valid_dict["val_loss"])
            jaccards.append(valid_dict["val_jacc"])
            dices.append(valid_dict["val_dice"])

            if valid_dict["val_jacc"] > best_jaccard:
                best_jaccard = valid_dict["val_jacc"]
                best_dice = valid_dict["val_dice"]
                best_uu_metrics = {"MaxF": uu_metrics["MaxF"], "AvgPrec": uu_metrics["AvgPrec"], "PRE": uu_metrics["PRE_wp"][0], "REC": uu_metrics["REC_wp"][0]}
                best_um_metrics = {"MaxF": um_metrics["MaxF"], "AvgPrec": um_metrics["AvgPrec"], "PRE": um_metrics["PRE_wp"][0], "REC": um_metrics["REC_wp"][0]}
                best_umm_metrics = {"MaxF": umm_metrics["MaxF"], "AvgPrec": umm_metrics["AvgPrec"], "PRE": umm_metrics["PRE_wp"][0], "REC": umm_metrics["REC_wp"][0]}
                best_model = copy.deepcopy(model.state_dict())

            if epoch and (epoch % status_every == 0):
                info_str = "\nEpoch: {}, LR: {}\n".format(epoch, scheduler.get_lr())
                info_str += "-"*30
                info_str += "\nTrain loss: {0}".format(epoch_train_loss)
                info_str += "\nValid loss: {0}".format(valid_dict["val_loss"]) 
                info_str += "\nValid Jaccard: {0}".format(valid_dict["val_jacc"]) 
                info_str += "\nValid DICE: {0}\n".format(valid_dict["val_dice"])
                
                #MaxF, PRE, REC, AvgPrec printing
                info_str += "\nUU_MaxF: {0}".format(uu_metrics["MaxF"])
                info_str += "\nUU_AvgPrec: {0}".format(uu_metrics["AvgPrec"])
                info_str += "\nUU_PRE: {0}".format(uu_metrics["PRE_wp"][0])
                info_str += "\nUU_REC: {0}\n".format(uu_metrics["REC_wp"][0])
                info_str += "\nUM_MaxF: {0}".format(um_metrics["MaxF"])
                info_str += "\nUM_AvgPrec: {0}".format(um_metrics["AvgPrec"])
                info_str += "\nUM_PRE: {0}".format(um_metrics["PRE_wp"][0])
                info_str += "\nUM_REC: {0}\n".format(um_metrics["REC_wp"][0])
                info_str += "\nUMM_MaxF: {0}".format(umm_metrics["MaxF"])
                info_str += "\nUMM_AvgPrec: {0}".format(umm_metrics["AvgPrec"])
                info_str += "\nUMM_PRE: {0}".format(umm_metrics["PRE_wp"][0])
                info_str += "\nUMM_REC: {0}\n".format(umm_metrics["REC_wp"][0])
                
                info_str += "-"*30
                info_str += "\n"
                console_logger.info(info_str)
                
                #Log to file
                info_str = "\nepoch: {}, ".format(epoch)
                info_str += "train_loss: {}, ".format(epoch_train_loss)
                info_str += "val_loss: {}, ".format(valid_dict["val_loss"])
                info_str += "val_jaccard: {}, ".format(valid_dict["val_jacc"])
                info_str += "val_dice: {}\n".format(valid_dict["val_dice"])
                file_logger.info(info_str)

                #Log to the tbX
                tbx_logger.log_scalars(tag="losses", values={"train_loss": epoch_train_loss, "valid_loss": valid_dict["val_loss"]}, step=epoch)
                tbx_logger.log_scalars(tag="metrics", values={"jaccard": valid_dict["val_jacc"], "DICE": valid_dict["val_dice"]}, step=epoch)
                #MaxF 
                tbx_logger.log_scalars(tag="MaxF", values={"uu_maxF": uu_metrics["MaxF"], "um_maxF": um_metrics["MaxF"], "umm_maxF": umm_metrics["MaxF"], "mmaxF": (uu_metrics["MaxF"] + um_metrics["MaxF"] + umm_metrics["MaxF"])/3}, step=epoch)
                #AvgPrec
                tbx_logger.log_scalars(tag="AvgPrec", values={"uu_AvgPrec": uu_metrics["AvgPrec"], "um_AvgPrec": um_metrics["AvgPrec"], "umm_AvgPrec": umm_metrics["AvgPrec"], "mAvgPrec": (uu_metrics["AvgPrec"] + um_metrics["AvgPrec"] + umm_metrics["AvgPrec"])/3}, step=epoch)
                #PRE
                tbx_logger.log_scalars(tag="PRE", values={"uu_PRE": uu_metrics["PRE_wp"][0], "um_PRE": um_metrics["PRE_wp"][0], "umm_PRE": umm_metrics["PRE_wp"][0], "mPRE": (uu_metrics["PRE_wp"][0] + um_metrics["PRE_wp"][0] + umm_metrics["PRE_wp"][0])/3}, step=epoch)
                #REC
                tbx_logger.log_scalars(tag="REC", values={"uu_REC": uu_metrics["REC_wp"][0], "um_REC": um_metrics["REC_wp"][0], "umm_REC": umm_metrics["REC_wp"][0], "mREC": (uu_metrics["REC_wp"][0] + um_metrics["REC_wp"][0] + umm_metrics["REC_wp"][0])/3}, step=epoch)

                #Log to csv
                maxf_csv_data.append("{},{},{},{},{}".format(epoch, uu_metrics["MaxF"], um_metrics["MaxF"], umm_metrics["MaxF"], (uu_metrics["MaxF"] + um_metrics["MaxF"] + umm_metrics["MaxF"])/3).split(","))
                avgprec_csv_data.append("{},{},{},{},{}".format(epoch, uu_metrics["AvgPrec"], um_metrics["AvgPrec"], umm_metrics["AvgPrec"], (uu_metrics["AvgPrec"] + um_metrics["AvgPrec"] + umm_metrics["AvgPrec"])/3).split(","))
                prec_csv_data.append("{},{},{},{},{}".format(epoch, uu_metrics["PRE_wp"][0], um_metrics["PRE_wp"][0], umm_metrics["PRE_wp"][0], (uu_metrics["PRE_wp"][0] + um_metrics["PRE_wp"][0] + umm_metrics["PRE_wp"][0])/3).split(","))
                rec_csv_data.append("{},{},{},{},{}".format(epoch, uu_metrics["REC_wp"][0], um_metrics["REC_wp"][0], umm_metrics["REC_wp"][0], (uu_metrics["REC_wp"][0] + um_metrics["REC_wp"][0] + umm_metrics["REC_wp"][0])/3).split(","))
                loss_csv_data.append("{},{},{}".format(epoch, epoch_train_loss, valid_dict["val_loss"]).split(","))
                jacc_dice_csv_data.append("{},{},{}".format(epoch, valid_dict["val_jacc"], valid_dict["val_dice"]).split(","))

        except KeyboardInterrupt:
            console_logger.info("KeyboardInterrupt, saving snapshot.")
            save_model(str(model_path), best_model, best_jaccard, best_dice, best_uu_metrics, best_um_metrics, best_umm_metrics, epoch)
            console_logger.info("Done!")
    
    info_str = "\nTraining process is done!\n" + "*"*30
    info_str += "\nTrain loss: {0}".format(np.mean(train_losses).astype(dtype=np.float64)) 
    info_str += "\nValid loss: {0}".format(np.mean(valid_losses).astype(dtype=np.float64))
    info_str += "\nMean Jaccard: {0}".format(np.mean(jaccards).astype(dtype=np.float64))
    info_str += "\nMean DICE: {0}".format(np.mean(dices).astype(dtype=np.float64))
    info_str += "\nBest Jaccard: {0}".format(best_jaccard)
    info_str += "\nBest DICE: {0}".format(best_dice)
    info_str += "\nBest UU_Metrics: {0}".format(best_uu_metrics)
    info_str += "\nBest UM_Metrics: {0}".format(best_um_metrics)
    info_str += "\nBest UMM_Metrics: {0}".format(best_umm_metrics)
    info_str += "\nMean MaxF: {0}\n".format((best_uu_metrics["MaxF"] + best_um_metrics["MaxF"] + best_umm_metrics["MaxF"])/3)
    info_str += "*"*30

    console_logger.info(info_str)
    file_logger.info(info_str)

    #model saving
    save_model(str(model_path), best_model, best_jaccard, best_dice, best_uu_metrics, best_um_metrics, best_umm_metrics, n_epochs)

    #Save to CSV
    write2csv(data=maxf_csv_data, file_name=str(csvs_path / "maxf.csv"), type_of_header="maxf")
    write2csv(data=avgprec_csv_data, file_name=str(csvs_path / "avgprec.csv"), type_of_header="avgprec")
    write2csv(data=prec_csv_data, file_name=str(csvs_path / "prec.csv"), type_of_header="prec")
    write2csv(data=rec_csv_data, file_name=str(csvs_path / "rec.csv"), type_of_header="rec")
    write2csv(data=loss_csv_data, file_name=str(csvs_path / "loss.csv"), type_of_header="loss")
    write2csv(data=jacc_dice_csv_data, file_name=str(csvs_path / "jd.csv"), type_of_header="jd")


def train_routine_a2d2(
        args,
        console_logger: logging.Logger,
        root: str,
        model: nn.Module,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        fm_eval_dataset,
        validation,
        fold,
        num_classes=1,
        n_epochs=100,
        status_every=5):
    """
        General trainig routine.
        params:
            args                    : argument parser parameters for saving it
            console_logger          : logger object for logging
            root                    : root dir where stores trained models
            model                   : model for training
            criterion               : loss function
            optimizer               : SGD, Adam or other
            scheduler               : learning rate scheduler
            train_loader            :
            valid_loader            :
            fm_eval_dataset         : dataset for F-max evaluation
            validation              : validation routine
            fold                    : number of fold
            num_classes             : number of classes
            n_epochs                : number of training epochs
            status_every            : the parameter which controls the frequency of status printing
    """

    # Load model if it exists
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)

    model_root = root / args.model_type / 'model{}'.format(fold)
    model_root.mkdir(exist_ok=True, parents=True)

    # CSV
    csvs_path = model_root / 'csv'
    csvs_path.mkdir(exist_ok=True, parents=True)

    model_path = model_root / 'model.pt'
    logging_path = model_root / 'train.log'

    # run params saving
    save_runparams(vars(args), file_path=(model_root / 'rparams.txt'))

    # file logger definition
    file_logger = logging.getLogger("file-logger")
    file_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(logging_path), mode='w')
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    file_logger.addHandler(fh)

    # Logging to the TensorBoardX
    tbx_logger = Logger(log_dir=str(model_root / "tbxlogs"))

    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state["epoch"]
        best_jaccard = state["best_jaccard"]
        best_dice = state["best_dice"]
        # best_metrics = state["best_metrics"]
        model.load_state_dict(state["model"])
        console_logger.info(
            "\nModel '{0}' was restored. Best Jaccard: {1}, Best DICE: {2}, Epoch: {3}".format(str(model_path),
                                                                                               best_jaccard, best_dice,
                                                                                               epoch))
    else:
        epoch = 1
        best_jaccard = 0
        best_dice = 0
        # best_metrics = {"MaxF": 0, "AvgPrec": 0, "PRE": 0, "REC": 0}

    n_epochs = n_epochs + epoch
    best_model = copy.deepcopy(model.state_dict())

    train_losses = []
    valid_losses = []
    jaccards = []
    dices = []

    # CSV data for logging
    loss_csv_data = []
    jacc_dice_csv_data = []
    # maxf_csv_data = []
    # avgprec_csv_data = []
    # prec_csv_data = []
    # rec_csv_data = []

    for epoch in range(epoch, n_epochs):

        epoch_train_losses = []

        # Train mode
        model.train()

        # scheduler step
        scheduler.step()

        try:
            optimizer.zero_grad()
            for i, (inputs1, inputs2, targets) in enumerate(train_loader):
                inputs1 = to_gpu(inputs1)
                inputs2 = to_gpu(inputs2)
                targets = to_gpu(targets)

                with torch.set_grad_enabled(True):
                    outputs = model(inputs1, inputs2)
                    loss = criterion(targets, outputs)
                    loss = loss/args.batch_factor
                    loss.backward()
                    if i % args.batch_factor == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    epoch_train_losses.append((loss*args.batch_factor).item())

            # Train loss per epoch
            epoch_train_loss = np.mean(epoch_train_losses).astype(dtype=np.float64)

            # Validation
            valid_dict, _, _ = validation(model, criterion, valid_loader, tbx_logger, epoch)
            # metrics = fmeasure_evaluation([model], fm_eval_dataset)

            train_losses.append(epoch_train_loss)
            valid_losses.append(valid_dict["val_loss"])
            jaccards.append(valid_dict["val_jacc"])
            dices.append(valid_dict["val_dice"])

            if valid_dict["val_jacc"] > best_jaccard:
                best_jaccard = valid_dict["val_jacc"]
                best_dice = valid_dict["val_dice"]
                # best_metrics = {"MaxF": metrics["MaxF"], "AvgPrec": metrics["AvgPrec"],
                #                    "PRE": metrics["PRE_wp"][0], "REC": metrics["REC_wp"][0]}
                best_model = copy.deepcopy(model.state_dict())

            if epoch and (epoch % status_every == 0):
                info_str = "\nEpoch: {}, LR: {}\n".format(epoch, scheduler.get_lr())
                info_str += "-" * 30
                info_str += "\nTrain loss: {0}".format(epoch_train_loss)
                info_str += "\nValid loss: {0}".format(valid_dict["val_loss"])
                info_str += "\nValid Jaccard: {0}".format(valid_dict["val_jacc"])
                info_str += "\nValid DICE: {0}\n".format(valid_dict["val_dice"])

                # # MaxF, PRE, REC, AvgPrec printing
                # info_str += "\nUU_MaxF: {0}".format(metrics["MaxF"])
                # info_str += "\nUU_AvgPrec: {0}".format(metrics["AvgPrec"])
                # info_str += "\nUU_PRE: {0}".format(metrics["PRE_wp"][0])
                # info_str += "\nUU_REC: {0}\n".format(metrics["REC_wp"][0])
                info_str += "-" * 30
                info_str += "\n"
                console_logger.info(info_str)

                # Log to file
                info_str = "\nepoch: {}, ".format(epoch)
                info_str += "train_loss: {}, ".format(epoch_train_loss)
                info_str += "val_loss: {}, ".format(valid_dict["val_loss"])
                info_str += "val_jaccard: {}, ".format(valid_dict["val_jacc"])
                info_str += "val_dice: {}\n".format(valid_dict["val_dice"])
                file_logger.info(info_str)

                # Log to the tbX
                tbx_logger.log_scalars(tag="losses",
                                       values={"train_loss": epoch_train_loss, "valid_loss": valid_dict["val_loss"]},
                                       step=epoch)
                tbx_logger.log_scalars(tag="metrics",
                                       values={"jaccard": valid_dict["val_jacc"], "DICE": valid_dict["val_dice"]},
                                       step=epoch)

                console_logger.info("End of  Epoch: Saving snapshot.")
            save_model_a2d2(str(model_path), best_model, best_jaccard, best_dice, epoch)
        except KeyboardInterrupt:
            console_logger.info("KeyboardInterrupt, saving snapshot.")
            #save_model(str(model_path), best_model, best_jaccard, best_dice, best_uu_metrics, best_um_metrics, best_umm_metrics, epoch)
            save_model_a2d2(str(model_path), best_model, best_jaccard, best_dice, epoch)
            console_logger.info("Done!")

    info_str = "\nTraining process is done!\n" + "*" * 30
    info_str += "\nTrain loss: {0}".format(np.mean(train_losses).astype(dtype=np.float64))
    info_str += "\nValid loss: {0}".format(np.mean(valid_losses).astype(dtype=np.float64))
    info_str += "\nMean Jaccard: {0}".format(np.mean(jaccards).astype(dtype=np.float64))
    info_str += "\nMean DICE: {0}".format(np.mean(dices).astype(dtype=np.float64))
    info_str += "\nBest Jaccard: {0}".format(best_jaccard)
    info_str += "\nBest DICE: {0}\n".format(best_dice)
    # info_str += "\nBest Metrics: {0}".format(best_metrics)

    info_str += "*" * 30
    console_logger.info(info_str)
    file_logger.info(info_str)

    # model saving
    # save_model(str(model_path), best_model, best_jaccard, best_dice, best_uu_metrics, best_um_metrics, best_umm_metrics, n_epochs)
    save_model_a2d2(str(model_path), best_model, best_jaccard, best_dice, n_epochs)

    # Save to CSV
    # write2csv(data=maxf_csv_data, file_name=str(csvs_path / "maxf.csv"), type_of_header="maxf")
    # write2csv(data=avgprec_csv_data, file_name=str(csvs_path / "avgprec.csv"), type_of_header="avgprec")
    # write2csv(data=prec_csv_data, file_name=str(csvs_path / "prec.csv"), type_of_header="prec")
    # write2csv(data=rec_csv_data, file_name=str(csvs_path / "rec.csv"), type_of_header="rec")
    write2csv(data=loss_csv_data, file_name=str(csvs_path / "loss.csv"), type_of_header="loss")
    write2csv(data=jacc_dice_csv_data, file_name=str(csvs_path / "jd.csv"), type_of_header="jd")


def binary_validation_routine_a2d2(model: nn.Module, criterion, valid_loader, tbx_logger=None, epoch: int=0, \
                                   save_image: bool=False):
    """
        This method by the given criterion, model and validation loader calculates Jaccard and DICE metrics with the validation loss for binary problem
    """
    with torch.set_grad_enabled(False):
        valid_losses = []
        jaccards = []
        dices = []

        target_images = []
        output_images = []

        model.eval()
        for idx, (inputs1, inputs2, targets) in enumerate(valid_loader):
            inputs1 = to_gpu(inputs1)
            inputs2 = to_gpu(inputs2)
            targets = to_gpu(targets)
            outputs = model(inputs1, inputs2)

            loss = criterion(targets, outputs)
            valid_losses.append(loss.item())
            jaccards += batch_jaccard(targets, (outputs > 0).float())
            dices += batch_dice(targets, (outputs > 0).float())

            if save_image:
                # Fix tensors to be images
                outputs = img_utils.getMaskFromTensor(outputs)
                inputs1 = img_utils.UnNormalize_tensor(inputs1, mean=[0.63263481, 0.63265741, 0.62899464], std=[0.25661512, 0.25698695, 0.2594808])

                # Log Images
                zeros = to_gpu(torch.zeros(targets.size()))
                mask_target = torch.cat((zeros, targets, zeros), 1)
                mask_output = torch.cat((zeros, outputs, zeros), 1)
                img_target = torch.add(inputs1 * 0.70, mask_target * 0.30)
                img_output = torch.add(inputs1 * 0.70, mask_output * 0.30)

                img_target = img_utils.getImageFromUnitTensor(torch.squeeze(img_target).cpu())
                img_output = img_utils.getImageFromUnitTensor(torch.squeeze(img_output).cpu())

                target_images.append(img_target)
                output_images.append(img_output)

        # Calculates losses
        valid_loss = np.mean(valid_losses).astype(dtype=np.float64)
        valid_jaccard = np.mean(jaccards).astype(dtype=np.float64)
        valid_dice = np.mean(dices).astype(dtype=np.float64)

        if not save_image:
            # Fix tensors to be images
            outputs = img_utils.getMaskFromTensor(outputs)
            inputs1 = img_utils.UnNormalize_tensor(inputs1, mean=[0.63263481, 0.63265741, 0.62899464], std=[0.25661512, 0.25698695, 0.2594808])

            # Log Images
            zeros = to_gpu(torch.zeros(targets.size()))
            mask_target = torch.cat((zeros, targets, zeros), 1)
            mask_output = torch.cat((zeros, outputs, zeros), 1)
            img_target = torch.add(inputs1 * 0.70, mask_target * 0.30)
            img_output = torch.add(inputs1 * 0.70, mask_output * 0.30)
            tbx_logger.log_image( torch.squeeze(img_target), epoch, dataformats='CHW', title='target')
            tbx_logger.log_image( torch.squeeze(img_output), epoch, dataformats='CHW', title='predicted')
        #'Size of images not equal for grid formation'
        # tbx_logger.log_image(torch.squeeze(inputs1), epoch, dataformats='CHW', title='inout image')
        # tbx_logger.log_image(torch.squeeze(targets), epoch, dataformats='HW', title='target_mask')
        # tbx_logger.log_image(torch.squeeze(outputs), epoch, dataformats='HW', title='output_mask')

        return {"val_loss": valid_loss, "val_jacc": valid_jaccard, "val_dice": valid_dice}, target_images, output_images


def fmeasure_evaluation_a2d2(model: nn.Module, valid_dataset):
    """
        This method by the given models and validation dataset calculates F-max measure, Precision, Recall and others metrics
    """

    # Eval mode for all models
    model.eval()

    thresh = np.array(range(0, 256)) / 255.0

    # UU
    totalFP = np.zeros(thresh.shape)
    totalFN = np.zeros(thresh.shape)
    totalPosNum = 0
    totalNegNum = 0

    for idx, batch in enumerate(valid_dataset):
        img, img2, mask = batch
        img = to_gpu(img.unsqueeze(0).contiguous().float())
        img2 = to_gpu(img2.unsqueeze(0).contiguous().float())
        mask = mask.squeeze().data.cpu().numpy().astype(dtype=np.bool)

        with torch.set_grad_enabled(False):
            predict = model(img, img2)
        predict = torch.sigmoid(predict)

        probs = (predict).squeeze(0).squeeze(0).data.cpu().numpy().astype(dtype=np.float32)

        FN, FP, posNum, negNum = evalExp(mask, probs, thresh, validMap=None, validArea=None)

        totalFP += FP
        totalFN += FN
        totalPosNum += posNum
        totalNegNum += negNum

    metrics = pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh=thresh)

    return metrics


def train_routine_a2d2_no_lidar(
        args,
        console_logger: logging.Logger,
        root: str,
        model: nn.Module,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        fm_eval_dataset,
        validation,
        fold,
        num_classes=1,
        n_epochs=100,
        status_every=5):
    """
        General trainig routine.
        params:
            args                    : argument parser parameters for saving it
            console_logger          : logger object for logging
            root                    : root dir where stores trained models
            model                   : model for training
            criterion               : loss function
            optimizer               : SGD, Adam or other
            scheduler               : learning rate scheduler
            train_loader            :
            valid_loader            :
            fm_eval_dataset         : dataset for F-max evaluation
            validation              : validation routine
            fold                    : number of fold
            num_classes             : number of classes
            n_epochs                : number of training epochs
            status_every            : the parameter which controls the frequency of status printing
    """

    # Load model if it exists
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)

    model_root = root / args.model_type / 'model{}'.format(fold)
    model_root.mkdir(exist_ok=True, parents=True)

    # CSV
    csvs_path = model_root / 'csv'
    csvs_path.mkdir(exist_ok=True, parents=True)

    model_path = model_root / 'model.pt'
    logging_path = model_root / 'train.log'

    # run params saving
    save_runparams(vars(args), file_path=(model_root / 'rparams.txt'))

    # file logger definition
    file_logger = logging.getLogger("file-logger")
    file_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(logging_path), mode='w')
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    file_logger.addHandler(fh)

    # Logging to the TensorBoardX
    tbx_logger = Logger(log_dir=str(model_root / "tbxlogs"))

    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state["epoch"]
        best_jaccard = state["best_jaccard"]
        best_dice = state["best_dice"]
        # best_metrics = state["best_metrics"]
        model.load_state_dict(state["model"])
        console_logger.info(
            "\nModel '{0}' was restored. Best Jaccard: {1}, Best DICE: {2}, Epoch: {3}".format(str(model_path),
                                                                                               best_jaccard, best_dice,
                                                                                               epoch))
    else:
        epoch = 1
        best_jaccard = 0
        best_dice = 0
        # best_metrics = {"MaxF": 0, "AvgPrec": 0, "PRE": 0, "REC": 0}

    n_epochs = n_epochs + epoch
    best_model = copy.deepcopy(model.state_dict())

    train_losses = []
    valid_losses = []
    jaccards = []
    dices = []

    # CSV data for logging
    loss_csv_data = []
    jacc_dice_csv_data = []
    # maxf_csv_data = []
    # avgprec_csv_data = []
    # prec_csv_data = []
    # rec_csv_data = []

    for epoch in range(epoch, n_epochs):

        epoch_train_losses = []

        # Train mode
        model.train()

        # scheduler step
        scheduler.step()

        try:
            optimizer.zero_grad()
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = to_gpu(inputs)
                targets = to_gpu(targets)

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(targets, outputs)
                    loss = loss/args.batch_factor
                    loss.backward()
                    if i % args.batch_factor == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    epoch_train_losses.append((loss*args.batch_factor).item())

            # Train loss per epoch
            epoch_train_loss = np.mean(epoch_train_losses).astype(dtype=np.float64)

            # Validation
            valid_dict, _, _ = validation(model, criterion, valid_loader, tbx_logger, epoch)
            # metrics = fmeasure_evaluation([model], fm_eval_dataset)

            train_losses.append(epoch_train_loss)
            valid_losses.append(valid_dict["val_loss"])
            jaccards.append(valid_dict["val_jacc"])
            dices.append(valid_dict["val_dice"])

            if valid_dict["val_jacc"] > best_jaccard:
                best_jaccard = valid_dict["val_jacc"]
                best_dice = valid_dict["val_dice"]
                # best_metrics = {"MaxF": metrics["MaxF"], "AvgPrec": metrics["AvgPrec"],
                #                    "PRE": metrics["PRE_wp"][0], "REC": metrics["REC_wp"][0]}
                best_model = copy.deepcopy(model.state_dict())

            if epoch and (epoch % status_every == 0):
                info_str = "\nEpoch: {}, LR: {}\n".format(epoch, scheduler.get_lr())
                info_str += "-" * 30
                info_str += "\nTrain loss: {0}".format(epoch_train_loss)
                info_str += "\nValid loss: {0}".format(valid_dict["val_loss"])
                info_str += "\nValid Jaccard: {0}".format(valid_dict["val_jacc"])
                info_str += "\nValid DICE: {0}\n".format(valid_dict["val_dice"])

                # # MaxF, PRE, REC, AvgPrec printing
                # info_str += "\nUU_MaxF: {0}".format(uu_metrics["MaxF"])
                # info_str += "\nUU_AvgPrec: {0}".format(uu_metrics["AvgPrec"])
                # info_str += "\nUU_PRE: {0}".format(uu_metrics["PRE_wp"][0])
                # info_str += "\nUU_REC: {0}\n".format(uu_metrics["REC_wp"][0])
                # info_str += "\nUM_MaxF: {0}".format(um_metrics["MaxF"])
                # info_str += "\nUM_AvgPrec: {0}".format(um_metrics["AvgPrec"])
                # info_str += "\nUM_PRE: {0}".format(um_metrics["PRE_wp"][0])
                # info_str += "\nUM_REC: {0}\n".format(um_metrics["REC_wp"][0])
                # info_str += "\nUMM_MaxF: {0}".format(umm_metrics["MaxF"])
                # info_str += "\nUMM_AvgPrec: {0}".format(umm_metrics["AvgPrec"])
                # info_str += "\nUMM_PRE: {0}".format(umm_metrics["PRE_wp"][0])
                # info_str += "\nUMM_REC: {0}\n".format(umm_metrics["REC_wp"][0])
                info_str += "-" * 30
                info_str += "\n"
                console_logger.info(info_str)

                # Log to file
                info_str = "\nepoch: {}, ".format(epoch)
                info_str += "train_loss: {}, ".format(epoch_train_loss)
                info_str += "val_loss: {}, ".format(valid_dict["val_loss"])
                info_str += "val_jaccard: {}, ".format(valid_dict["val_jacc"])
                info_str += "val_dice: {}\n".format(valid_dict["val_dice"])
                file_logger.info(info_str)

                # Log to the tbX
                tbx_logger.log_scalars(tag="losses",
                                       values={"train_loss": epoch_train_loss, "valid_loss": valid_dict["val_loss"]},
                                       step=epoch)
                tbx_logger.log_scalars(tag="metrics",
                                       values={"jaccard": valid_dict["val_jacc"], "DICE": valid_dict["val_dice"]},
                                       step=epoch)
                # # MaxF
                # tbx_logger.log_scalars(tag="MaxF", values={"uu_maxF": uu_metrics["MaxF"], "um_maxF": um_metrics["MaxF"],
                #                                            "umm_maxF": umm_metrics["MaxF"], "mmaxF": (uu_metrics[
                #                                                                                           "MaxF"] +
                #                                                                                       um_metrics[
                #                                                                                           "MaxF"] +
                #                                                                                       umm_metrics[
                #                                                                                           "MaxF"]) / 3},
                #                        step=epoch)
                # # AvgPrec
                # tbx_logger.log_scalars(tag="AvgPrec",
                #                        values={"uu_AvgPrec": uu_metrics["AvgPrec"], "um_AvgPrec": um_metrics["AvgPrec"],
                #                                "umm_AvgPrec": umm_metrics["AvgPrec"], "mAvgPrec": (uu_metrics[
                #                                                                                        "AvgPrec"] +
                #                                                                                    um_metrics[
                #                                                                                        "AvgPrec"] +
                #                                                                                    umm_metrics[
                #                                                                                        "AvgPrec"]) / 3},
                #                        step=epoch)
                # # PRE
                # tbx_logger.log_scalars(tag="PRE",
                #                        values={"uu_PRE": uu_metrics["PRE_wp"][0], "um_PRE": um_metrics["PRE_wp"][0],
                #                                "umm_PRE": umm_metrics["PRE_wp"][0], "mPRE": (uu_metrics["PRE_wp"][0] +
                #                                                                              um_metrics["PRE_wp"][0] +
                #                                                                              umm_metrics["PRE_wp"][
                #                                                                                  0]) / 3}, step=epoch)
                # # REC
                # tbx_logger.log_scalars(tag="REC",
                #                        values={"uu_REC": uu_metrics["REC_wp"][0], "um_REC": um_metrics["REC_wp"][0],
                #                                "umm_REC": umm_metrics["REC_wp"][0], "mREC": (uu_metrics["REC_wp"][0] +
                #                                                                              um_metrics["REC_wp"][0] +
                #                                                                              umm_metrics["REC_wp"][
                #                                                                                  0]) / 3}, step=epoch)
                #
                # # Log to csv
                # maxf_csv_data.append(
                #     "{},{},{},{},{}".format(epoch, uu_metrics["MaxF"], um_metrics["MaxF"], umm_metrics["MaxF"],
                #                             (uu_metrics["MaxF"] + um_metrics["MaxF"] + umm_metrics["MaxF"]) / 3).split(
                #         ","))
                # avgprec_csv_data.append(
                #     "{},{},{},{},{}".format(epoch, uu_metrics["AvgPrec"], um_metrics["AvgPrec"], umm_metrics["AvgPrec"],
                #                             (uu_metrics["AvgPrec"] + um_metrics["AvgPrec"] + umm_metrics[
                #                                 "AvgPrec"]) / 3).split(","))
                # prec_csv_data.append("{},{},{},{},{}".format(epoch, uu_metrics["PRE_wp"][0], um_metrics["PRE_wp"][0],
                #                                              umm_metrics["PRE_wp"][0], (
                #                                                          uu_metrics["PRE_wp"][0] + um_metrics["PRE_wp"][
                #                                                      0] + umm_metrics["PRE_wp"][0]) / 3).split(","))
                # rec_csv_data.append("{},{},{},{},{}".format(epoch, uu_metrics["REC_wp"][0], um_metrics["REC_wp"][0],
                #                                             umm_metrics["REC_wp"][0], (
                #                                                         uu_metrics["REC_wp"][0] + um_metrics["REC_wp"][
                #                                                     0] + umm_metrics["REC_wp"][0]) / 3).split(","))
                # loss_csv_data.append("{},{},{}".format(epoch, epoch_train_loss, valid_dict["val_loss"]).split(","))
                # jacc_dice_csv_data.append(
                #     "{},{},{}".format(epoch, valid_dict["val_jacc"], valid_dict["val_dice"]).split(","))
            console_logger.info("End of  Epoch: Saving snapshot.")
            save_model_a2d2(str(model_path), best_model, best_jaccard, best_dice, epoch)
        except KeyboardInterrupt:
            console_logger.info("KeyboardInterrupt, saving snapshot.")
            #save_model(str(model_path), best_model, best_jaccard, best_dice, best_uu_metrics, best_um_metrics, best_umm_metrics, epoch)
            save_model_a2d2(str(model_path), best_model, best_jaccard, best_dice, epoch)
            console_logger.info("Done!")

    info_str = "\nTraining process is done!\n" + "*" * 30
    info_str += "\nTrain loss: {0}".format(np.mean(train_losses).astype(dtype=np.float64))
    info_str += "\nValid loss: {0}".format(np.mean(valid_losses).astype(dtype=np.float64))
    info_str += "\nMean Jaccard: {0}".format(np.mean(jaccards).astype(dtype=np.float64))
    info_str += "\nMean DICE: {0}".format(np.mean(dices).astype(dtype=np.float64))
    info_str += "\nBest Jaccard: {0}".format(best_jaccard)
    info_str += "\nBest DICE: {0}\n".format(best_dice)
    # info_str += "\nBest UU_Metrics: {0}".format(best_uu_metrics)
    # info_str += "\nBest UM_Metrics: {0}".format(best_um_metrics)
    # info_str += "\nBest UMM_Metrics: {0}".format(best_umm_metrics)
    # info_str += "\nMean MaxF: {0}\n".format((best_uu_metrics["MaxF"] + best_um_metrics["MaxF"] + best_umm_metrics["MaxF"])/3)

    info_str += "*" * 30
    console_logger.info(info_str)
    file_logger.info(info_str)

    # model saving
    # save_model(str(model_path), best_model, best_jaccard, best_dice, best_uu_metrics, best_um_metrics, best_umm_metrics, n_epochs)
    save_model_a2d2(str(model_path), best_model, best_jaccard, best_dice, n_epochs)

    # Save to CSV
    # write2csv(data=maxf_csv_data, file_name=str(csvs_path / "maxf.csv"), type_of_header="maxf")
    # write2csv(data=avgprec_csv_data, file_name=str(csvs_path / "avgprec.csv"), type_of_header="avgprec")
    # write2csv(data=prec_csv_data, file_name=str(csvs_path / "prec.csv"), type_of_header="prec")
    # write2csv(data=rec_csv_data, file_name=str(csvs_path / "rec.csv"), type_of_header="rec")
    write2csv(data=loss_csv_data, file_name=str(csvs_path / "loss.csv"), type_of_header="loss")
    write2csv(data=jacc_dice_csv_data, file_name=str(csvs_path / "jd.csv"), type_of_header="jd")


def binary_validation_routine_a2d2_no_lidar(model: nn.Module, criterion, valid_loader, tbx_logger=None, epoch: int=0, \
                                            save_image: bool=False):
    """
        This method by the given criterion, model and validation loader calculates Jaccard and DICE metrics with the validation loss for binary problem
    """
    with torch.set_grad_enabled(False):
        valid_losses = []
        jaccards = []
        dices = []

        target_images = []
        output_images = []

        model.eval()
        for idx, (inputs, targets) in enumerate(valid_loader):
            inputs = to_gpu(inputs)
            targets = to_gpu(targets)
            outputs = model(inputs)

            loss = criterion(targets, outputs)
            valid_losses.append(loss.item())
            jaccards += batch_jaccard(targets, (outputs > 0).float())
            dices += batch_dice(targets, (outputs > 0).float())

            if save_image:
                # Fix tensors to be images
                outputs = img_utils.getMaskFromTensor(outputs)
                inputs = img_utils.UnNormalize_tensor(inputs, mean=[0.63263481, 0.63265741, 0.62899464], std=[0.25661512, 0.25698695, 0.2594808])

                # Log Images
                zeros = to_gpu(torch.zeros(targets.size()))
                mask_target = torch.cat((zeros, targets, zeros), 1)
                mask_output = torch.cat((zeros, outputs, zeros), 1)
                img_target = torch.add(inputs * 0.70, mask_target * 0.30)
                img_output = torch.add(inputs * 0.70, mask_output * 0.30)

                img_target = img_utils.getImageFromUnitTensor(torch.squeeze(img_target).cpu())
                img_output = img_utils.getImageFromUnitTensor(torch.squeeze(img_output).cpu())

                target_images.append(img_target)
                output_images.append(img_output)

        # Calculates losses
        valid_loss = np.mean(valid_losses).astype(dtype=np.float64)
        valid_jaccard = np.mean(jaccards).astype(dtype=np.float64)
        valid_dice = np.mean(dices).astype(dtype=np.float64)

        if not save_image:
            # Fix tensors to be images
            outputs = img_utils.getMaskFromTensor(outputs)
            inputs = img_utils.UnNormalize_tensor(inputs, mean=[0.63263481, 0.63265741, 0.62899464], std=[0.25661512, 0.25698695, 0.2594808])

            # Log Images
            zeros = to_gpu(torch.zeros(targets.size()))
            mask_target = torch.cat((zeros, targets, zeros), 1)
            mask_output = torch.cat((zeros, outputs, zeros), 1)
            img_target = torch.add(inputs * 0.70, mask_target * 0.30)
            img_output = torch.add(inputs * 0.70, mask_output * 0.30)
            tbx_logger.log_image(torch.squeeze(img_target), epoch, dataformats='CHW', title='target')
            tbx_logger.log_image(torch.squeeze(img_output), epoch, dataformats='CHW', title='predicted')
        # 'Size of images not equal for grid formation'
        # tbx_logger.log_image(torch.squeeze(inputs), epoch, dataformats='CHW', title='inout image')
        # tbx_logger.log_image(torch.squeeze(targets), epoch, dataformats='HW', title='target_mask')
        # tbx_logger.log_image(torch.squeeze(outputs), epoch, dataformats='HW', title='output_mask')

        return {"val_loss": valid_loss, "val_jacc": valid_jaccard, "val_dice": valid_dice}, target_images, output_images


def fmeasure_evaluation_a2d2_no_lidar(model: nn.Module, valid_dataset):
    """
        This method by the given models and validation dataset calculates F-max measure, Precision, Recall and others metrics
    """

    # Eval mode for all models
    model.eval()

    thresh = np.array(range(0, 256)) / 255.0

    # UU
    totalFP = np.zeros(thresh.shape)
    totalFN = np.zeros(thresh.shape)
    totalPosNum = 0
    totalNegNum = 0

    for idx, batch in enumerate(valid_dataset):
        img, mask = batch
        img = to_gpu(img.unsqueeze(0).contiguous().float())
        mask = mask.squeeze().data.cpu().numpy().astype(dtype=np.bool)

        with torch.set_grad_enabled(False):
            predict = model(img)
        predict = torch.sigmoid(predict)

        probs = (predict).squeeze(0).squeeze(0).data.cpu().numpy().astype(dtype=np.float32)

        FN, FP, posNum, negNum = evalExp(mask, probs, thresh, validMap=None, validArea=None)

        totalFP += FP
        totalFN += FN
        totalPosNum += posNum
        totalNegNum += negNum

    metrics = pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh=thresh)

    return metrics


def binary_validation_routine(model: nn.Module, criterion, valid_loader):
    """
        This method by the given criterion, model and validation loader calculates Jaccard and DICE metrics with the validation loss for binary problem
    """
    with torch.set_grad_enabled(False):
        valid_losses = []
        jaccards = []
        dices = []

        model.eval()
        for idx, batch in enumerate(valid_loader):
            inputs, targets = batch
            inputs = to_gpu(inputs)
            targets = to_gpu(targets)
            outputs = model(inputs)

            loss = criterion(targets, outputs)
            valid_losses.append(loss.item())
            jaccards += batch_jaccard(targets, (outputs > 0).float())
            dices += batch_dice(targets, (outputs > 0).float())

        # Calculates losses
        valid_loss = np.mean(valid_losses).astype(dtype=np.float64)
        valid_jaccard = np.mean(jaccards).astype(dtype=np.float64)
        valid_dice = np.mean(dices).astype(dtype=np.float64)

        return {"val_loss": valid_loss, "val_jacc": valid_jaccard, "val_dice": valid_dice}


def multi_validation_routine(model: nn.Module, criterion, valid_loader, num_classes=1):
    """
        This method by the given criterion, model and validation loader calculates Jaccard and DICE metrics with the validation loss for a multi-class problem
    """
    with torch.set_grad_enabled(False):
        valid_losses = []

        # Eval mode
        model.eval()

        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint32)
        for idx, batch in enumerate(valid_loader):
            inputs, targets = batch
            inputs = to_gpu(inputs)
            targets = to_gpu(targets)
            outputs = model(inputs)

            loss = criterion(targets, outputs)
            valid_losses.append(loss.item())

            output = outputs.data.cpu().numpy().argmax(axis=1)  # output classes
            target = targets.data.cpu().numpy()

            confusion_matrix += calculate_confusion_matrix_from_arrays(output, target, num_classes)

        confusion_matrix = confusion_matrix[1:, 1:]  # remove background

        # Jaccards and Dices
        jaccards = calculate_jaccards(confusion_matrix)
        dices = calculate_dices(confusion_matrix)

        mean_valid_loss = np.mean(valid_losses).astype(dtype=np.float64)
        mean_jaccard = np.mean(jaccards).astype(dtype=np.float64)
        mean_dice = np.mean(dices).astype(dtype=np.float64)

        jaccards_pc = {"Jaccard_{}".format(cls + 1): jaccard for cls, jaccard in enumerate(jaccards)}
        dices_pc = {"DICE_{}".format(cls + 1): dice for cls, dice in enumerate(dices)}

        return {"val_loss": mean_valid_loss, "val_jacc": mean_jaccard, "val_dice": mean_dice,
                "per_class_jacc": jaccards_pc, "per_class_dice": dices_pc}



def fmeasure_evaluation(models: nn.ModuleList, valid_dataset):
    """
        This method by the given models and validation dataset calculates F-max measure, Precision, Recall and others metrics
    """

    #Eval mode for all models
    for model in models:
        model.eval() 

    thresh = np.array(range(0,256))/255.0
    
    #UU
    uu_totalFP = np.zeros( thresh.shape )
    uu_totalFN = np.zeros( thresh.shape )
    uu_totalPosNum = 0
    uu_totalNegNum = 0

    #UM
    um_totalFP = np.zeros( thresh.shape )
    um_totalFN = np.zeros( thresh.shape )
    um_totalPosNum = 0
    um_totalNegNum = 0

    #UMM
    umm_totalFP = np.zeros( thresh.shape )
    umm_totalFN = np.zeros( thresh.shape )
    umm_totalPosNum = 0
    umm_totalNegNum = 0


    for idx, batch in enumerate(valid_dataset):
        img, mask, path = batch
        img = to_gpu(img.unsqueeze(0).contiguous().float())
        mask = mask.astype(dtype=np.bool)

        #Averaging all predictions for one point of validation data
        sum_predicts = to_gpu(torch.zeros((1, 1, mask.shape[0], mask.shape[1])).float())    
        
        for model in models:
            with torch.set_grad_enabled(False):
                predict = model(img)
            sum_predicts += F.sigmoid(predict)
        
        probs = (sum_predicts / len(models)).squeeze(0).squeeze(0).data.cpu().numpy().astype(dtype=np.float32)

        FN, FP, posNum, negNum = evalExp(mask, probs, thresh, validMap=None, validArea=None)

        cat = path.split("/")[-1].split(".")[0].split("_")[0]
        if cat == "uu":
            uu_totalFP += FP
            uu_totalFN += FN
            uu_totalPosNum += posNum
            uu_totalNegNum += negNum    
        elif cat == "um":
            um_totalFP += FP
            um_totalFN += FN
            um_totalPosNum += posNum
            um_totalNegNum += negNum
        else:
            umm_totalFP += FP
            umm_totalFN += FN
            umm_totalPosNum += posNum
            umm_totalNegNum += negNum

    uu_metrcis = pxEval_maximizeFMeasure(uu_totalPosNum, uu_totalNegNum, uu_totalFN, uu_totalFP, thresh=thresh)
    um_metrcis = pxEval_maximizeFMeasure(um_totalPosNum, um_totalNegNum, um_totalFN, um_totalFP, thresh=thresh)
    umm_metrcis = pxEval_maximizeFMeasure(umm_totalPosNum, umm_totalNegNum, umm_totalFN, umm_totalFP, thresh=thresh)

    return uu_metrcis, um_metrcis, umm_metrcis


def read_pickle_file(filename: str):
    if Path(filename).is_file():
        with open(filename, 'rb') as handle:
            return pickle.load(handle)
    else:
        raise ValueError(f'Pickle file {filename} does not exist')


def shutdown_system():
    os.system("shutdown -h now")