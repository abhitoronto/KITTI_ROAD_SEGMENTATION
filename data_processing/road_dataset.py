"""
    This module represents KITTI dataset.
    Key features:
        1:  Loading images and their coresponding masks for training or validation
        2:  Loading images and their coresponding names for test
"""

# general imports
import cv2
import numpy as np
from pathlib import Path

# torch packages imports
import torch
from torch.utils.data import Dataset

# Divide factor
class_factor = 50

# Pickle File names
a2d2_ip_input_file = 'ip_inputs.pkl'
a2d2_upsample_input_file = 'upsample_inputs.pkl'
a2d2_output_file = 'outputs.pkl'

class RoadDataset(Dataset):
    def __init__(self, dataset_path, transforms=None, is_valid=False, is_test=False):
        """
            params:
                dataset_path    : Path to the KITTI dataset which contains two subdirs: 'testing' and 'training'
                transforms      : Transformations for image augmentation
                is_valid        : Validation dataset loading
                is_test         : Loading only images without labels if 'True', else loading images and their labels
        """
        super().__init__()

        self.dataset_path = dataset_path
        self.transforms = transforms
        self.is_valid = is_valid
        self.is_test = is_test

        if self.is_test:
            self.images_paths = sorted(list((self.dataset_path / 'testing' / 'image_2').glob('*')))
            self.labels_paths = None
        else:
            self.images_paths = sorted(list((self.dataset_path / 'training' / 'image_2').glob('*')))
            self.valid_labels_paths = sorted(list((self.dataset_path / 'training' / 'valid_masks').glob('*')))
            self.train_labels_paths = sorted(list((self.dataset_path / 'training' / 'train_masks').glob('*')))

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        if self.is_test:
            img = load_rgb_image_(str(self.images_paths[idx]))
            return numpy_to_tensor(img).float(), str(self.images_paths[idx])
        else:
            if self.is_valid:
                img = load_rgb_image_(str(self.images_paths[idx]))
                valid_mask = load_mask_(str(self.valid_labels_paths[idx]))
                train_mask = load_mask_(str(self.train_labels_paths[idx]))
                img[:, :, 0] *= (train_mask == 0)
                img[:, :, 1] *= (train_mask == 0)
                img[:, :, 2] *= (train_mask == 0)
                # img = img * np.expand_dims((train_mask == 0), axis=2)
                if self.transforms:
                    augmented = self.transforms(image=img, mask=valid_mask)
                    img, valid_mask = augmented["image"], augmented["mask"]
                return numpy_to_tensor(img).float(), torch.from_numpy(np.expand_dims(valid_mask, 0)).float()
            else:
                img = load_rgb_image_(str(self.images_paths[idx]))
                valid_mask = load_mask_(str(self.valid_labels_paths[idx]))
                train_mask = load_mask_(str(self.train_labels_paths[idx]))
                # img = img * np.expand_dims((valid_mask == 0), axis=2)
                img[:, :, 0] *= (valid_mask == 0)
                img[:, :, 1] *= (valid_mask == 0)
                img[:, :, 2] *= (valid_mask == 0)
                if self.transforms:
                    augmented = self.transforms(image=img, mask=train_mask)
                    img, train_mask = augmented["image"], augmented["mask"]
                return numpy_to_tensor(img).float(), torch.from_numpy(np.expand_dims(train_mask, 0)).float()


class RoadDataset2(Dataset):
    """
        This dataset represents a data without source validation areas.
    """
    def __init__(self, img_paths, mask_paths, transforms=None, fmeasure_eval=False):
        """
            params:
                img_paths         : list with paths to source images
                maks_paths        : list with paths to masks
                transforms        : transformations
                fmeasure_eval     : if True then method __getitem__ will return image, mask and it's path. Otherwise image and mask
        """
        super().__init__()
        self.img_paths = sorted(img_paths)
        self.mask_paths = sorted(mask_paths)
        self.transforms = transforms
        self.fmeasure_eval = fmeasure_eval

        assert len(self.img_paths) == len(self.mask_paths), "Error. 'img_paths' and 'mask_paths' has different length."

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = load_rgb_image_(self.img_paths[idx])
        mask = load_mask_(self.mask_paths[idx])

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        if self.fmeasure_eval:
            return numpy_to_tensor(img).float(), mask, self.img_paths[idx]
        else:
            return numpy_to_tensor(img).float(), torch.from_numpy(np.expand_dims(mask, 0)).float()


class a2d2_dataset(Dataset):
    """
        This is the dataset for the Audi a2d2 self driving data.
        It loads
            NN Inputs: RGB images | x-coord dense lidar images | y-"" | z-""
            NN Outputs: binary mask of a specific colour
    """

    def __init__(self, img_paths, mask_paths, transform_image=None, normalize_image=None, normalize_lidar=None):
        """
            params:
                img_paths         : list of lists with path to source RGB and XYZ images
                                    the expected structure: img_paths[0] = [camera, x-img, y-img, z-img]
                mask_paths        : list with paths to masks
        """
        super().__init__()
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform_image = transform_image
        self.normalize_image = normalize_image
        self.normalize_lidar = normalize_lidar

        self.size = (480, 320)

        assert len(self.img_paths) == len(self.mask_paths), "Error. 'img_paths' and 'mask_paths' has different length."

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_cam = load_rgb_image_(self.img_paths[idx][0])
        img_x = load_grayscale_image_(self.img_paths[idx][1])
        img_y = load_grayscale_image_(self.img_paths[idx][2])
        img_z = load_grayscale_image_(self.img_paths[idx][3])
        img_lidar = cv2.merge((img_x, img_y, img_z))
        mask = load_mask_(self.mask_paths[idx])

        img_cam = rescale_frame(img_cam, self.size)
        img_lidar = rescale_frame(img_lidar, self.size)
        mask = rescale_frame(mask, self.size)

        if self.transform_image and self.normalize_image and self.normalize_lidar:
            augmented = self.transform_image(image=img_cam, image2=img_lidar, mask=mask)
            img_cam, image_lidar, mask = augmented["image"], augmented["image2"], augmented["mask"]

            augmented_cam = self.normalize_image(image=img_cam)
            img_cam = augmented_cam["image"]

            augmented_lidar = self.normalize_lidar(image=img_lidar)
            img_lidar = augmented_lidar["image"]

        return [numpy_to_tensor(img_cam).float(),
                numpy_to_tensor(img_lidar).float(),
                torch.from_numpy(np.expand_dims(mask, 0)).float()]


class a2d2_dataset_no_lidar(Dataset):
    """
        This is the dataset for the Audi a2d2 self driving data.
        It loads
            NN Inputs: RGB images | x-coord dense lidar images | y-"" | z-""
            NN Outputs: binary mask of a specific colour
    """

    def __init__(self, img_paths, mask_paths,transform_image=None, normalize_image=None):
        """
            params:
                img_paths         : list of lists with path to source RGB and XYZ images
                                    the expected structure: img_paths[0] = [camera, x-img, y-img, z-img]
                mask_paths        : list with paths to masks
        """
        super().__init__()
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform_image = transform_image
        self.normalize_image = normalize_image

        self.size = (480, 320)

        assert len(self.img_paths) == len(self.mask_paths), "Error. 'img_paths' and 'mask_paths' has different length."

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_cam = load_rgb_image_(self.img_paths[idx][0])
        mask = load_mask_(self.mask_paths[idx])

        img_cam = rescale_frame(img_cam, self.size)
        mask = rescale_frame(mask, self.size)

        if self.transform_image and self.normalize_image:
            augmented = self.transform_image(image=img_cam, mask=mask)
            img_cam, mask = augmented["image"], augmented["mask"]

            augmented_cam = self.normalize_image(image=img_cam)
            img_cam = augmented_cam["image"]

        return [numpy_to_tensor(img_cam).float(),
                torch.from_numpy(np.expand_dims(mask, 0)).float()]


def numpy_to_tensor(img: np.ndarray):
    return torch.from_numpy(np.transpose(img, (2, 0, 1)))


def load_rgb_image_(img_path):
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_image_(img_path):
    img = cv2.imread(img_path)
    return img


def load_grayscale_image_(img_path):
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


def load_mask_(mask_path):
    mask = cv2.imread(mask_path, 0)
    return (mask / 255).astype(dtype=np.uint8)


def rescale_frame(frame, size):
    return cv2.resize(frame, size, interpolation =cv2.INTER_AREA)