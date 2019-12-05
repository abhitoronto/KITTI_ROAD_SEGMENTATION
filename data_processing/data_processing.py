"""
    This module perform preparation of the train and validation datasets for KITTI benchmark
"""

import cv2
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from utils.img_utils import (
        getGroundTruth,
        ImageSpecifications,
)

train_masks_dir = "train_masks"
valid_masks_dir = "valid_masks"
gt_image_2_dir = "gt_image_2"                           #   Ground truth train and validation masks 
image_2_dir = "image_2"                                 #   RGB images
droped_valid_image_2_dir = "droped_valid_image_2"       #   source train images without validation areas
holdout_dir = "hold_out"                                #   this dir contains a hold-out dataset which will be used in evaluation mode.

def prepare_train_valid(path2train):
    """
        Select train and validation masks and save it in separate dirs.
        Params:
            path2train: path to a dir which contains 'gt_image' and 'image_2' subdirs
    """

    path2train = Path(path2train)

    if not path2train.exists():
        raise ValueError("Dir '{}' is not exist.".format(path2train))

    gt_image_2 = path2train / gt_image_2_dir
    image_2 = path2train / image_2_dir

    images_path_list = sorted(list(image_2.glob('*')))
    gt_images_path_list = sorted(list(gt_image_2.glob('*')))

    #Deletion of lane examples 
    gt_images_path_list = gt_images_path_list[95:]

    assert len(images_path_list) == len(gt_images_path_list), "Error: 'images_path_list' and 'gt_images_path_list' has different sizes."

    #Train mask dir
    train_masks = path2train / train_masks_dir
    train_masks.mkdir(parents=True, exist_ok=True)

    #Valid mask dir
    valid_masks = path2train / valid_masks_dir
    valid_masks.mkdir(parents=True, exist_ok=True)

    for idx, img_name in enumerate(images_path_list):
        img_name = str(img_name)
        print("Processing '{}' file.".format(img_name))

        db_name = img_name.split('/')[-1].split('.')[0].split('_')[0]
        num = img_name.split('/')[-1].split('.')[0].split('_')[1]

        road_mask, valid_mask = getGroundTruth(gt_images_path_list[idx])
        
        # if db_name == "uu":
        #     factor = ImageSpecifications.uu_cat
        # elif db_name == "um":
        #     factor = ImageSpecifications.um_cat
        # elif db_name == "umm":
        #     factor = ImageSpecifications.umm_cat
        # else:
        #     raise ValueError("Unknowm category name: {}".format(db_name))

        factor = 255

        road_mask = road_mask.astype(dtype=np.uint8) * factor
        valid_mask = (~valid_mask).astype(dtype=np.uint8) * factor
    
        cv2.imwrite(str(train_masks / (db_name + '_' + num + ImageSpecifications.img_extension)), road_mask)
        cv2.imwrite(str(valid_masks / (db_name + '_' + num + ImageSpecifications.img_extension)), valid_mask)


def drop_valid(path2train):
    """
        Drops validation areas from original train images.
        Params:
            path2train: path that contains 'image_2' and 'valid_masks' subdirs
    """

    path2train = Path(path2train)

    droped_valid_image_2 = path2train / droped_valid_image_2_dir
    droped_valid_image_2.mkdir(parents=True, exist_ok=True)

    src_images_paths = sorted(list((path2train / image_2_dir).glob("*")))
    valid_masks_paths = sorted(list((path2train / valid_masks_dir).glob("*")))

    assert len(src_images_paths) == len(valid_masks_paths), "Error: src_images_paths and valid_masks_paths has different length."

    for i in range(0, len(src_images_paths)):
        src_image = cv2.imread(str(src_images_paths[i]))
        valid_mask = cv2.imread(str(valid_masks_paths[i]), 0)
        src_image = src_image * np.expand_dims((valid_mask == 0), axis=2).astype(dtype=np.uint8)
        
        img_name = str(src_images_paths[i]).split("/")[-1]        
        cv2.imwrite(str(droped_valid_image_2 / img_name), src_image)
        print("Image {0} porcessed.".format(str(src_images_paths[i])))


def crossval_split(images_paths: str, masks_paths: str, fold=5):
    """
        Splits images and masks by two sets: train and validation by folds with a small stratification by categories 'uu', 'um' and 'umm'. 
        Possible value for 'fold' is: 1, 2, 3, 4, 5
        
        params:
            images_paths      :   dir with source images(without validation area)
            masks_paths       :   dir with masks
            fold              :   number of validation fold
    """

    images_paths = Path(images_paths)
    masks_paths = Path(masks_paths)

    images_paths = sorted(list(map(str, images_paths.glob("*"))))
    masks_paths = sorted(list(map(str, masks_paths.glob("*"))))

    if len(images_paths) < 5:
        raise RuntimeError("Length of images_paths less then 5.")

    if fold not in range(1,6):
        raise ValueError("Invalid fold number: {}. 'fold' can be 1,2,3,4 or 5.".format(fold))

    #Urban unmarked
    uu_imgs_paths = list(filter(lambda p: p.split("/")[-1].startswith("uu_"), images_paths))
    uu_masks_paths = list(filter(lambda p: p.split("/")[-1].startswith("uu_"), masks_paths))

    #Urban marked
    um_imgs_paths = list(filter(lambda p: p.split("/")[-1].startswith("um_"), images_paths))
    um_masks_paths = list(filter(lambda p: p.split("/")[-1].startswith("um_"), masks_paths))

    #Urban multiple mark
    umm_imgs_paths = list(filter(lambda p: p.split("/")[-1].startswith("umm_"), images_paths))
    umm_masks_paths = list(filter(lambda p: p.split("/")[-1].startswith("umm_"), masks_paths))

    assert len(uu_imgs_paths) == len(uu_masks_paths), "Error. uu_imgs_paths and uu_masks_paths has differnet length."
    assert len(um_imgs_paths) == len(um_masks_paths), "Error. um_imgs_paths and um_masks_paths has differnet length."
    assert len(umm_imgs_paths) == len(umm_masks_paths), "Error. umm_imgs_paths and umm_masks_paths has differnet length."

    uu_imgs_per_fold = round(len(uu_imgs_paths) / 5)
    um_imgs_per_fold = round(len(um_imgs_paths) / 5)
    umm_imgs_per_fold = round(len(umm_imgs_paths) / 5)

    #train urban unmarked
    if fold == 5:
        #UU
        valid_uu_imgs_paths = uu_imgs_paths[-(len(uu_imgs_paths)-uu_imgs_per_fold*4):]
        valid_uu_masks_paths = uu_masks_paths[-(len(uu_imgs_paths)-uu_imgs_per_fold*4):]
        train_uu_imgs_paths = list(set(uu_imgs_paths) - set(valid_uu_imgs_paths))
        train_uu_masks_paths = list(set(uu_masks_paths) - set(valid_uu_masks_paths))

        #UM
        valid_um_imgs_paths = um_imgs_paths[-(len(um_imgs_paths)-um_imgs_per_fold*4):]
        valid_um_masks_paths = um_masks_paths[-(len(um_imgs_paths)-um_imgs_per_fold*4):]
        train_um_imgs_paths = list(set(um_imgs_paths) - set(valid_um_imgs_paths))
        train_um_masks_paths = list(set(um_masks_paths) - set(valid_um_masks_paths))

        #UMM
        valid_umm_imgs_paths = umm_imgs_paths[-(len(umm_imgs_paths)-umm_imgs_per_fold*4):]
        valid_umm_masks_paths = umm_masks_paths[-(len(umm_imgs_paths)-umm_imgs_per_fold*4):]
        train_umm_imgs_paths = list(set(umm_imgs_paths) - set(valid_umm_imgs_paths))
        train_umm_masks_paths = list(set(umm_masks_paths) - set(valid_umm_masks_paths))
    else:
        #UU
        valid_uu_imgs_paths = uu_imgs_paths[:fold*uu_imgs_per_fold][-uu_imgs_per_fold:]
        valid_uu_masks_paths = uu_masks_paths[:fold*uu_imgs_per_fold][-uu_imgs_per_fold:]
        train_uu_imgs_paths = list(set(uu_imgs_paths) - set(valid_uu_imgs_paths))
        train_uu_masks_paths = list(set(uu_masks_paths) - set(valid_uu_masks_paths))

        #UM
        valid_um_imgs_paths = um_imgs_paths[:fold*um_imgs_per_fold][-um_imgs_per_fold:]
        valid_um_masks_paths = um_masks_paths[:fold*um_imgs_per_fold][-um_imgs_per_fold:]
        train_um_imgs_paths = list(set(um_imgs_paths) - set(valid_um_imgs_paths))
        train_um_masks_paths = list(set(um_masks_paths) - set(valid_um_masks_paths))

        #UMM
        valid_umm_imgs_paths = umm_imgs_paths[:fold*umm_imgs_per_fold][-umm_imgs_per_fold:]
        valid_umm_masks_paths = umm_masks_paths[:fold*umm_imgs_per_fold][-umm_imgs_per_fold:]
        train_umm_imgs_paths = list(set(umm_imgs_paths) - set(valid_umm_imgs_paths))
        train_umm_masks_paths = list(set(umm_masks_paths) - set(valid_umm_masks_paths))

    #total train
    train_imgs_total = train_uu_imgs_paths + train_um_imgs_paths + train_umm_imgs_paths
    train_masks_total = train_uu_masks_paths + train_um_masks_paths + train_umm_masks_paths

    #total valid
    valid_imgs_total = valid_uu_imgs_paths + valid_um_imgs_paths + valid_umm_imgs_paths
    valid_masks_total = valid_uu_masks_paths + valid_um_masks_paths + valid_umm_masks_paths

    return ((train_imgs_total, train_masks_total), (valid_imgs_total, valid_masks_total))


def crossval_split_a2d2(imgs_paths: list, masks_paths: list, fold=5):
    """
        Splits images and masks by two sets: train and validation by folds with a small stratification by categories 'uu', 'um' and 'umm'.
        Possible value for 'fold' is: 1, 2, 3, 4, 5

        params:
            imgs_paths      :   list with source images(without validation area)
            masks_paths       :   list with masks
            fold              :   number of validation fold
    """

    if len(imgs_paths) < 5:
        raise RuntimeError("Length of imgs_paths less then 5.")

    if fold not in range(1, 6):
        raise ValueError("Invalid fold number: {}. 'fold' can be 1,2,3,4 or 5.".format(fold))

    assert len(imgs_paths) == len(masks_paths), "Error: imgs_paths and masks_paths has different length."

    imgs_per_fold = round(len(imgs_paths) / 5)

    # train urban unmarked
    if fold == 5:
        valid_imgs_paths = imgs_paths[-(len(imgs_paths) - imgs_per_fold * 4):]
        valid_masks_paths = masks_paths[-(len(imgs_paths) - imgs_per_fold * 4):]
        train_imgs_paths = imgs_paths[:(imgs_per_fold * 4)]
        train_masks_paths = masks_paths[:(imgs_per_fold * 4)]
    else:
        valid_imgs_paths = imgs_paths[(fold-1)*imgs_per_fold:fold * imgs_per_fold]
        valid_masks_paths = masks_paths[(fold-1)*imgs_per_fold:fold * imgs_per_fold]
        train_imgs_paths = imgs_paths[:(fold-1)*imgs_per_fold] + imgs_paths[fold*imgs_per_fold:]
        train_masks_paths = masks_paths[:(fold-1)*imgs_per_fold] + masks_paths[fold*imgs_per_fold:]

    return ((train_imgs_paths, train_masks_paths), (valid_imgs_paths, valid_masks_paths))


def prepare_holdout_dataset(path2data, test_fraction=0.2, original=True, rnd=False, state=111):
    """
        This method splits a train dataset on train and hold-out data and save this data in a separately directory.
        Params:
            path2data       : dir which contains 'training' and 'testing' subdirs.
            test_fraction   : part of a test data in percent
            orignal         : droped or original
            rnd             : random selection or not
            state           : state for random generator
    """

    assert test_fraction <= 0.9 and test_fraction > 0, "Error. Invalid test_fraction param: {}".format(test_fraction)

    path2data = Path(path2data)
    holdout_path = path2data / holdout_dir

    #Creating of hold-out dir and subdirs
    holdout_path.mkdir(exist_ok=True, parents=True)
    (holdout_path / 'imgs').mkdir(exist_ok=True, parents=True)
    (holdout_path / 'masks').mkdir(exist_ok=True, parents=True)

    train_images_paths = path2data / 'training' / droped_valid_image_2_dir if not original else path2data / 'training' / image_2_dir
    train_masks_paths = path2data / 'training' / train_masks_dir

    images_paths = sorted(list(map(str, train_images_paths.glob("*"))))
    masks_paths = sorted(list(map(str, train_masks_paths.glob("*"))))

    #Urban unmarked
    uu_imgs_paths = list(filter(lambda p: p.split("/")[-1].startswith("uu_"), images_paths))
    uu_masks_paths = list(filter(lambda p: p.split("/")[-1].startswith("uu_"), masks_paths))

    #Urban marked
    um_imgs_paths = list(filter(lambda p: p.split("/")[-1].startswith("um_"), images_paths))
    um_masks_paths = list(filter(lambda p: p.split("/")[-1].startswith("um_"), masks_paths))

    #Urban multiple mark
    umm_imgs_paths = list(filter(lambda p: p.split("/")[-1].startswith("umm_"), images_paths))
    umm_masks_paths = list(filter(lambda p: p.split("/")[-1].startswith("umm_"), masks_paths))

    assert len(uu_imgs_paths) == len(uu_masks_paths), "Error. uu_imgs_paths and uu_masks_paths has differnet length."
    assert len(um_imgs_paths) == len(um_masks_paths), "Error. um_imgs_paths and um_masks_paths has differnet length."
    assert len(umm_imgs_paths) == len(umm_masks_paths), "Error. umm_imgs_paths and umm_masks_paths has differnet length."

    if rnd:
        #train\holdout uu
        train_uu_imgs, test_uu_imgs, train_uu_masks, test_uu_masks = train_test_split(uu_imgs_paths, uu_masks_paths, test_size=test_fraction, random_state=state)
        #train\holdout um
        train_um_imgs, test_um_imgs, train_um_masks, test_um_masks = train_test_split(um_imgs_paths, um_masks_paths, test_size=test_fraction, random_state=state)
        #train\holdout umm
        train_umm_imgs, test_umm_imgs, train_umm_masks, test_umm_masks = train_test_split(umm_imgs_paths, umm_masks_paths, test_size=test_fraction, random_state=state)
    else:
        #UU
        uu_len = len(uu_imgs_paths)
        n = round(uu_len * test_fraction)
        step = round(uu_len//n)
        test_uu_imgs, test_uu_masks = uu_imgs_paths[::step], uu_masks_paths[::step]
        train_uu_imgs, train_uu_masks = list(set(uu_imgs_paths) - set(test_uu_imgs)), list(set(uu_masks_paths) - set(test_uu_masks))

        #UM
        um_len = len(um_imgs_paths)
        n = round(um_len * test_fraction)
        step = round(um_len//n)
        test_um_imgs, test_um_masks = um_imgs_paths[::step], um_masks_paths[::step]
        train_um_imgs, train_um_masks = list(set(um_imgs_paths) - set(test_um_imgs)), list(set(um_masks_paths) - set(test_um_masks))

        #UMM
        umm_len = len(umm_imgs_paths)
        n = round(umm_len * test_fraction)
        step = round(umm_len//n)
        test_umm_imgs, test_umm_masks = umm_imgs_paths[::step], umm_masks_paths[::step]
        train_umm_imgs, train_umm_masks = list(set(umm_imgs_paths) - set(test_umm_imgs)), list(set(umm_masks_paths) - set(test_umm_masks))

    #Total train
    total_train_imgs = train_uu_imgs + train_um_imgs + train_umm_imgs
    total_train_masks = train_uu_masks + train_um_masks + train_umm_masks

    #Total test(hold-out)
    total_test_imgs = test_uu_imgs + test_um_imgs + test_umm_imgs
    total_test_masks = test_uu_masks + test_um_masks + test_umm_masks

    if len(list( (holdout_path / 'imgs').glob("*") )) > 0:
        raise RuntimeError("Dir {} already exists. Possible it contains already prepared hold-out test images.".format(str(holdout_path / 'imgs')))

    if len(list( (holdout_path / 'masks').glob("*") )) > 0:
        raise RuntimeError("Dir {} already exists. Possible it contains already prepared hold-out test masks.".format(str(holdout_path / 'masks')))

    print("------------INFO-------------")
    print("Total train images: {}".format(len(total_train_imgs)))
    print("Moving {} images and {} masks to the hold-out dirs {}, {} respectively".format(len(total_test_imgs), len(total_test_masks), str(holdout_path / 'imgs'), str(holdout_path / 'masks')))
    print("------------INFO-------------")

    #Moving test images and masks to a hold-out dir
    for idx in range(0, len(total_test_imgs)):
        img_path = Path(total_test_imgs[idx])
        mask_path = Path(total_test_masks[idx])

        #move
        img_path.rename(holdout_path / 'imgs' / img_path.name)
        mask_path.rename(holdout_path / 'masks' / mask_path.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Argument parser for data_processing module")
    ap.add_argument("--train-dir", type=str, required=True, help="Path to a train dir(should contain 'gt_image_2' and 'image_2' subdirs).")
    args = ap.parse_args()

    prepare_train_valid(args.train_dir)
    drop_valid(args.train_dir)
#     prepare_holdout_dataset(path2data=args.train_dir, original=False)

    print("Done!")