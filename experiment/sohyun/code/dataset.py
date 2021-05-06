import os
import random
import time
import json
import glob
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from utils import label_accuracy_score
import cv2

import numpy as np
import pandas as pd

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


from utils import seed_everything

seed_everything(21)


# train.json / validation.json / test.json 디렉토리 설정
# dataset_path = "./input/data"
dataset_path = "/opt/ml/input/data"
train_path = dataset_path + "/train.json"
val_path = dataset_path + "/val.json"
all_path = dataset_path + "/train_all.json"
test_path = dataset_path + "/test.json"
kfold_train_path = [f"{dataset_path}/train_data{i}.json" for i in range(5)]
kfold_val_path = [f"{dataset_path}/valid_data{i}.json" for i in range(5)]

# augmentations
_train_transform = [
    A.Resize(256, 256),
    A.RandomResizedCrop(256, 256),
    A.Resize(256, 256),
    # A.CLAHE(),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # imagenet
    ToTensorV2(),
]

_valid_transform = [
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
]

_test_transform = [
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
]


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]["id"] == classID:
            return cats[i]["name"]
    return "None"


class CustomDataLoader(Dataset):
    """COCO format"""

    category_names = [
        "Backgroud",
        "UNKNOWN",
        "General trash",
        "Paper",
        "Paper pack",
        "Metal",
        "Glass",
        "Plastic",
        "Styrofoam",
        "Plastic bag",
        "Battery",
        "Clothing",
    ]

    def __init__(self, data_dir, mode="train", transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos["file_name"]))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        # images /= 255.0

        if self.mode in ("train", "val"):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos["id"])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = get_classname(anns[i]["category_id"], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i]) * pixel_value, masks)
            masks = masks.astype(np.float32)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]

            return images, masks, image_infos

        if self.mode == "test":
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]

            return images, image_infos

    def __len__(self):
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


class PseudoTrainset(Dataset):
    """COCO format"""

    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.transform = transform
        self.coco = COCO(data_dir)
        self.dataset_path = "/opt/ml/input/data/"
        self.category_names = [
            "Backgroud",
            "UNKNOWN",
            "General trash",
            "Paper",
            "Paper pack",
            "Metal",
            "Glass",
            "Plastic",
            "Styrofoam",
            "Plastic bag",
            "Battery",
            "Clothing",
        ]

        self.pseudo_imgs = np.load(self.dataset_path + "pseudo_imgs_path.npy")
        self.pseudo_masks = sorted(glob.glob(self.dataset_path + "pseudo_masks/*.npy"))

    def __getitem__(self, index: int):

        ### Train data ###
        if index < len(self.coco.getImgIds()):
            image_id = self.coco.getImgIds(imgIds=index)
            image_infos = self.coco.loadImgs(image_id)[0]

            images = cv2.imread(self.dataset_path + image_infos["file_name"])
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            # images /= 255.0
            ann_ids = self.coco.getAnnIds(imgIds=image_infos["id"])
            anns = self.coco.loadAnns(ann_ids)
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            ###  mask 생성  ###
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            for i in range(len(anns)):
                className = get_classname(anns[i]["category_id"], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i]) * pixel_value, masks)

        ### Pseudo data ###
        else:
            index -= len(self.coco.getImgIds())
            images = cv2.imread(self.dataset_path + self.pseudo_imgs[index])
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
            # images /= 255.0
            masks = np.load(self.pseudo_masks[index])

        ###  augmentation ###
        masks = masks.astype(np.float32)
        if self.transform is not None:
            transformed = self.transform(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]

        return images, masks

    def __len__(self):
        return len(self.coco.getImgIds()) + len(self.pseudo_imgs)


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(batch_size=20, fold_index=None):
    """dataloader를 반환합니다

    Args:
        batch_size (int) : Defaults to 20.
        fold_index (tuple): train_index와 val_index

    Returns:
        (train_loader, val_loader)
    """
    train_transform = A.Compose(_train_transform)
    val_transform = A.Compose(_valid_transform)

    if fold_index is not None:
        # Kfold
        # train_dataset = CustomDataLoader(
        #     data_dir=kfold_train_path[fold_index],
        #     mode="train",
        #     transform=train_transform,
        # )
        train_dataset = PseudoTrainset(data_dir=train_path, transform=train_transform)

        val_dataset = CustomDataLoader(
            data_dir=kfold_val_path[fold_index], mode="val", transform=val_transform
        )

        # 마지막 step에서 1개 남을땐 drop_last
        drop_last = True if len(train_dataset) % batch_size == 1 else False

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )

    else:
        # Hold Out
        # train dataset
        train_dataset = PseudoTrainset(data_dir=train_path, transform=train_transform)

        # validation dataset
        val_dataset = CustomDataLoader(
            data_dir=val_path, mode="val", transform=val_transform
        )

        # 마지막 step에서 1개 남을땐 drop_last
        drop_last = True if len(train_dataset) % batch_size == 1 else False

        # DataLoader
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )

    return train_loader, val_loader


def get_testloader(batch_size=20):
    test_transform = A.Compose(_test_transform)

    # test dataset
    test_dataset = CustomDataLoader(
        data_dir=test_path, mode="test", transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=2,
        collate_fn=collate_fn,
    )

    return test_loader


if __name__ == "__main__":
    pass

