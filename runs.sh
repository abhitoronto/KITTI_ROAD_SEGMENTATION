#!/bin/bash

#   ======================================================
#
#   This code was written for runing training procecess.
#
#   ======================================================

#
#   ============LidCamNet=============
#
#fold2 early
python3 main_a2d2.py \
    --root-dir="./trained_models/" \
    --dataset-path="/hdd/a2d2-data/dataset/" \
    --optim="Adam" \
    --lr=0.0005 \
    --model-type="lcn_early" \
    --batch-size=2 \
    --batch-factor=8 \
    --fold=2 \
    --n-epochs=25 \
    --scheduler="poly" \
    --gamma=0.9 \
    --alpha=0.3 \
    --num-workers=4 \



#fold2 lcn
python3 main_a2d2.py \
    --root-dir="./trained_models/" \
    --dataset-path="/hdd/a2d2-data/dataset/" \
    --optim="Adam" \
    --lr=0.0005 \
    --model-type="lcn" \
    --batch-size=2 \
    --batch-factor=8 \
    --fold=2 \
    --n-epochs=20 \
    --scheduler="poly" \
    --gamma=0.9 \
    --alpha=0.3 \
    --num-workers=4 \

python3 prediction_a2d2.py --mode=single --model-path=./trained_models/lcn/model4/model.pt --model-type=lcn \
--path2image=/media/a2d2-data/camera_lidar_semantic/20180807_145028/camera/cam_front_center/20180807145028_camera_frontcenter_000009786.png \
--path2save=/home/abhinavg/Desktop --num-images=5 \
--path2gt=/hdd/a2d2-data/camera_lidar_semantic/20180807_145028/image-gt/cam_front_center/20180807145028_image-gt_frontcenter_000009786.png

##fold2
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "lcn" \
#    --enc-bn-enable 0 \
#    --batch-size 32 \
#    --fold 2 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold3
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "lcn" \
#    --enc-bn-enable 0 \
#    --batch-size 32 \
#    --fold 3 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold4
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "lcn" \
#    --enc-bn-enable 0 \
#    --batch-size 32 \
#    --fold 4 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold5
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "lcn" \
#    --enc-bn-enable 0 \
#    --batch-size 32 \
#    --fold 5 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
#
##
##   ============RekNetM1=============
##
## fold 1
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm1" \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 16 \
#    --fold 1 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold 2
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm1" \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 16 \
#    --fold 2 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold 3
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm1" \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 16 \
#    --fold 3 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold 4
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm1" \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 16 \
#    --fold 4 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold 5
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm1" \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 16 \
#    --fold 5 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
#
##
##   ============RekNetM2 + PolyLR w\o attention=============
##
## fold 1
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm2" \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 16 \
#    --fold 1 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold 2
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm2" \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 8 \
#    --fold 2 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold 3
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm2" \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 8 \
#    --fold 3 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold 4
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm2" \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 8 \
#    --fold 4 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold 5
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm2" \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 8 \
#    --fold 5 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##
##   ============RekNetM2 + PolyLR + Attention=============
##
## fold 1
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm2" \
#    --attention 1 \
#    --skip-conn 1 \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 8 \
#    --fold 1 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold 2
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm2" \
#    --attention 1 \
#    --skip-conn 1 \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 8 \
#    --fold 2 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold 3
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm2" \
#    --attention 1 \
#    --skip-conn 1 \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 8 \
#    --fold 3 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold 4
#python3 main.py \
#    --root-dir 'trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm2" \
#    --attention 1 \
#    --skip-conn 1 \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 8 \
#    --fold 4 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3
#
##fold 5
#python3 main.py \
#    --root-dir 'trained_trained_models/' \
#    --dataset-path '../../TMPDataSets/KITTI_ROAD/data_road/' \
#    --act-type "celu" \
#    --optim 'Adam' \
#    --lr 0.0005 \
#    --model-type "reknetm2" \
#    --attention 1 \
#    --skip-conn 1 \
#    --decoder-type 'convTr' \
#    --enc-bn-enable 1 \
#    --dec-bn-enable 1 \
#    --batch-size 8 \
#    --fold 5 \
#    --n-epochs 150 \
#    --scheduler 'poly' \
#    --gamma 0.9 \
#    --alpha 0.3

python3 shutdown_system.py
