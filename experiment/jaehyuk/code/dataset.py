import os
import random
import time
import json
import warnings
from albumentations.augmentations.transforms import ColorJitter, Flip, GaussianBlur

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
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
from copy_paste import copy_paste_class
from copy_paste import CopyPaste

sns.set()

from utils import seed_everything

seed_everything(21)

# train.json / validation.json / test.json 디렉토리 설정
# dataset_path = "./input/data"
dataset_path = "/opt/ml/input/data"
train_path = dataset_path + "/train.json"
val_path = dataset_path + "/val.json"
test_path = dataset_path + "/test.json"
all_path = dataset_path + "/train_all.json"
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
# augmentations
_train_transform = [
    # A.Resize(256, 256),
    # A.RandomResizedCrop(256,256),
    # A.Flip(),
    # A.GaussianBlur(),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    # Lighting,
    CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1),  # copy & paste
    A.Normalize(mean=(0.4611, 0.4403, 0.4193), std=(0.2107, 0.2074, 0.2157)),
    ToTensorV2(),
]

_valid_transform = [
    A.Resize(256, 256),
    A.Normalize(mean=(0.4611, 0.4403, 0.4193), std=(0.2107, 0.2074, 0.2157)),
    ToTensorV2(),
]

# _test_transform = [ToTensorV2()]
_test_transform = [
    A.Resize(256, 256),
    A.Normalize(mean=(0.4611, 0.4403, 0.4193), std=(0.2107, 0.2074, 0.2157)),
    ToTensorV2(),
]
def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]["id"] == classID:
            return cats[i]["name"]
    return "None"

@copy_paste_class
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
    def __init__(self, data_dir, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transforms = transform
        self.coco = COCO(data_dir)
        

    def load_example(self, index: int):
        """__getitem__ 함수라고 생각

        Args:
            index (int)

        Returns:
            [dict]]: [image 정보(images,masks,boxex)]
        """
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        # images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            bboxes=[]
            maskses=[]
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                # print(className)
                pixel_value = self.category_names.index(className)
                # masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
                maskses.append(np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks))
                # masks.append(self.coco.annToMask(obj))
                bboxes.append(anns[i]['bbox']+[anns[i]['category_id']]+[i])
            masks = masks.astype(np.float32)
            output = {
            'image': images,
            'masks': maskses,
            'bboxes': bboxes
            }   
            # return images, masks, bboxes,image_infos
            return self.transforms(**output)
            # return images, masks
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transforms is not None:
                transformed = self.transforms(image=images)
                images = transformed["image"]
            
            return images, image_infos

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())
# class CustomDataLoader(Dataset):
#     """COCO format"""

#     category_names = [
#         "Backgroud",
#         "UNKNOWN",
#         "General trash",
#         "Paper",
#         "Paper pack",
#         "Metal",
#         "Glass",
#         "Plastic",
#         "Styrofoam",
#         "Plastic bag",
#         "Battery",
#         "Clothing",
#     ]

#     def __init__(self, data_dir, mode="train", transform=None):
#         super().__init__()
#         self.mode = mode
#         self.transform = transform
#         self.coco = COCO(data_dir)

#     def __getitem__(self, index: int):
#         # dataset이 index되어 list처럼 동작
#         image_id = self.coco.getImgIds(imgIds=index)
#         image_infos = self.coco.loadImgs(image_id)[0]

#         # cv2 를 활용하여 image 불러오기
#         images = cv2.imread(os.path.join(dataset_path, image_infos["file_name"]))
#         images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
#         # images /= 255.0

#         if self.mode in ("train", "val"):
#             ann_ids = self.coco.getAnnIds(imgIds=image_infos["id"])
#             anns = self.coco.loadAnns(ann_ids)

#             # Load the categories in a variable
#             cat_ids = self.coco.getCatIds()
#             cats = self.coco.loadCats(cat_ids)

#             # masks : size가 (height x width)인 2D
#             # 각각의 pixel 값에는 "category id + 1" 할당
#             # Background = 0
#             masks = np.zeros((image_infos["height"], image_infos["width"]))
#             # Unknown = 1, General trash = 2, ... , Cigarette = 11
#             for i in range(len(anns)):
#                 className = get_classname(anns[i]["category_id"], cats)
#                 pixel_value = self.category_names.index(className)
#                 masks = np.maximum(self.coco.annToMask(anns[i]) * pixel_value, masks)
#             masks = masks.astype(np.float32)

#             # transform -> albumentations 라이브러리 활용
#             if self.transform is not None:
#                 transformed = self.transform(image=images, mask=masks)
#                 images = transformed["image"]
#                 masks = transformed["mask"]

#             return images, masks, image_infos

#         if self.mode == "test":
#             # transform -> albumentations 라이브러리 활용
#             if self.transform is not None:
#                 transformed = self.transform(image=images)
#                 images = transformed["image"]

#             return images, image_infos

#     def __len__(self):
#         # 전체 dataset의 size를 return
#         return len(self.coco.getImgIds())


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
    train_transform = A.Compose(_train_transform, bbox_params=A.BboxParams(format="coco", min_visibility=0.05))

    val_transform = A.Compose(_valid_transform)

    if fold_index:
        # Kfold
        dataset = CustomDataLoader(data_dir=all_path, mode='train', transform=train_transform)

        train_subsampler = SubsetRandomSampler(fold_index[0]) # train_index sampler
        val_subsampler = SubsetRandomSampler(fold_index[1]) # val_index sampler

        train_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=train_subsampler,
            num_workers=4, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=val_subsampler, 
            num_workers=4, collate_fn=collate_fn, drop_last=True)

    else:
        # Hold Out
        # train dataset
        train_dataset = CustomDataLoader(
            data_dir=train_path, mode="train", transform=train_transform
        )

        # validation dataset
        val_dataset = CustomDataLoader(
            data_dir=val_path, mode="val", transform=val_transform
        )

        # DataLoader
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

    return train_loader, val_loader


def get_testloader(batch_size = 20):
    # test_transform = A.Compose([A.Resize(256,256),ToTensorV2()])
    test_transform = A.Compose(_test_transform)
    # test dataset
    test_dataset = CustomDataLoader(
        data_dir=test_path, mode="test", transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_fn,
    )

    return test_loader

if __name__ == "__main__":
    pass

    
