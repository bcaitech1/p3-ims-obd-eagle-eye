import os
import random
import time
import json
import warnings
import argparse

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import label_accuracy_score, get_miou, FocalTverskyLoss
import cv2

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch_optimizer import adamp
from utils import *

import numpy as np
import pandas as pd

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import segmentation_models_pytorch as smp
from vgg16 import FCN8s
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dice_loss import SoftDiceLoss
from datetime import datetime, timedelta


import wandb

print("pytorch version: {}".format(torch.__version__))
print("GPU 사용 가능 여부: {}".format(torch.cuda.is_available()))

print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())

device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 가능 여부에 따라 device 정보 저장

# seed 고정
random_seed = 21
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


dataset_path = "/opt/ml/input/data"
anns_file_path = dataset_path + "/" + "train.json"

# Read annotations
with open(anns_file_path, "r") as f:
    dataset = json.loads(f.read())

categories = dataset["categories"]
anns = dataset["annotations"]
imgs = dataset["images"]
nr_cats = len(categories)
nr_annotations = len(anns)
nr_images = len(imgs)

# Load categories and super categories
cat_names = []  # 모든 카테고리 이름
for cat_it in categories:
    cat_names.append(cat_it["name"])

print("Number of categories:", nr_cats)
print("Number of annotations:", nr_annotations)
print("Number of images:", nr_images)

# Count annotations
cat_histogram = np.zeros(nr_cats, dtype=int)
for ann in anns:
    cat_histogram[ann["category_id"]] += 1  # 카테고리별 개수 카운트

# Convert to DataFrame
df = pd.DataFrame({"Categories": cat_names, "Number of annotations": cat_histogram})

# category labeling
sorted_temp_df = df.sort_index()

# background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
sorted_df = pd.DataFrame(["Backgroud"], columns=["Categories"])
sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)

# class (Categories) 에 따른 index 확인 (0~11 : 총 12개)
# sorted_df

category_names = list(sorted_df.Categories)


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]["id"] == classID:
            return cats[i]["name"]
    return "None"


class CustomDataLoader(Dataset):
    """COCO format"""

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
        images /= 255.0

        if self.mode in ("train"):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos["id"])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros(
                (12, image_infos["height"], image_infos["width"]), dtype=np.ubyte
            )
            max_masks = np.zeros((image_infos["height"], image_infos["width"]))

            masks[0] = 1
            existed_class = set([0])
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = get_classname(anns[i]["category_id"], cats)
                class_index = category_names.index(className)
                coco_mask = self.coco.annToMask(anns[i])

                # dice 용
                masks[class_index] = coco_mask
                masks[0] = (masks[0] == 1) & (masks[class_index] != 1)
                existed_class.add(class_index)

                # ce 용
                max_masks = np.maximum(coco_mask * class_index, max_masks)

            masks = masks.astype(np.float32)
            max_masks = torch.from_numpy(max_masks)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]

            return images, masks, image_infos, max_masks, list(existed_class)

        elif self.mode in ("val"):
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
                pixel_value = category_names.index(className)
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

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


parser = argparse.ArgumentParser(description="Semantic Segmentation!!!")
parser.add_argument("--BATCH_SIZE", default=16, type=int)
parser.add_argument("--LEARNING_RATE", default=0.0001, type=float)
parser.add_argument("--SCHEDULER", default="Reduce_lr", type=str)
parser.add_argument("--dc_weight", default=0.7, type=float)

args = parser.parse_args()

now = datetime.now() + timedelta(hours=9)
args.now = now.strftime("%Y%m%d_%H%M")
# ----------------------------------------------------
# TODO
exp_title = "dc100_lr00003_withBack"  # 실험 이름
batch_size = args.BATCH_SIZE  # Mini-batch size
num_epochs = 10
learning_rate = args.LEARNING_RATE
num_workers = 0


# train.json / validation.json / test.json 디렉토리 설정
train_path = dataset_path + "/train.json"
val_path = dataset_path + "/val.json"

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


train_transform = A.Compose([ToTensorV2()])

# train dataset
train_dataset = CustomDataLoader(
    data_dir=train_path, mode="train", transform=train_transform
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_fn,
)

# validation dataset
val_dataset = CustomDataLoader(data_dir=val_path, mode="val", transform=train_transform)
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=collate_fn,
)

np.set_printoptions(4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)

model = smp.DeepLabV3Plus(
    encoder_name="se_resnext101_32x4d",
    classes=12,
)

# model = FCN8s(num_classes=12)
# checkpoint = torch.load("/opt/ml/p3-ims-obd-eagle-eye/experiment/jiyoung/exp/saved/all_class/checkp_9.pt", map_location=device)
# model.load_state_dict(checkpoint)
model = model.to(device)

# x = torch.randn([2, 3, 512, 512]).to(device)
# print("input shape : ", x.shape)
# out = model(x).to(device)
# print("output shape : ", out.size())

criterion = SoftDiceLoss(apply_nonlin=nn.Softmax(dim=1), do_bg=True)
cross_en = nn.CrossEntropyLoss()

optimizer = adamp.AdamP(
    model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-2
)

if args.SCHEDULER == "Reduce_lr":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=1
    )
elif args.SCHEDULER == "cosine_lr":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

# run = wandb.init(project="important")
run = wandb.init()

for epoch in range(num_epochs):

    model.train()
    for step, (images, masks, _, max_masks, existed_class) in enumerate(train_loader):
        optimizer.zero_grad()

        images = torch.stack(images).to(device)  # (batch, channel, height, width)
        masks = torch.stack(masks).long().to(device)  # (batch, channel, height, width)
        max_masks = (
            torch.stack(max_masks).long().to(device)
        )  # (batch, channel, height, width)

        outputs = model(images)

        # batch_loss = 0.0
        # for i, selected in enumerate(existed_class):
        #     each_out = outputs[i, selected]
        #     each_mask = masks[i, selected]
        #     loss, _ = criterion(each_out, each_mask)
        #     batch_loss += loss / images.size(0)

        cross_loss = cross_en(outputs, max_masks)

        dice_loss, _ = criterion(outputs, masks)
        # loss.backward()

        # www = torch.tensor()
        multi_loss = args.dc_weight * dice_loss + (1 - args.dc_weight) * cross_loss
        multi_loss.backward()

        optimizer.step()

        # wandb

        wandb.log(
            {
                "batch_loss": multi_loss.item(),
                "step": step,
                "epoch": epoch,
            }
        )

    # annealing
    # scheduler.step()

    print(f"end train_epoch {epoch}")

    miou_all = []
    model.eval()
    best_score = 0.0
    hist = np.zeros((12, 12))
    with torch.no_grad():
        for step, (images, masks, _) in enumerate(val_loader):

            images = torch.stack(images).to(device)  # (batch, channel, height, width)
            masks = (
                torch.stack(masks).long().to(device)
            )  # (batch, channel, height, width)
            outputs = model(images)

            # loss, sum_cls = criterion(outputs, masks)

            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)

            # 리더보드용 miou 저장
            miou_list = get_miou(masks.detach().cpu().numpy(), outputs, n_class=12)
            miou_all.extend(miou_list)

    print(hist)
    _, _, miou, _ = label_accuracy_score(hist)

    lb_miou = np.nanmean(miou_all)

    if args.SCHEDULER == "Reduce_lr":
        scheduler.step(1 - miou)
    elif args.SCHEDULER == "cosine_lr":
        scheduler.step()

    wandb.log(
        {"lb_miou": lb_miou, "miou": miou, "lr": get_lr(optimizer), "epoch": epoch}
    )

    if miou > best_score:
        os.makedirs(
            f"/opt/ml/p3-ims-obd-eagle-eye/experiment/jiyoung/exp/saved/{args.now}",
            exist_ok=True,
        )
        torch.save(
            model.state_dict(),
            f"/opt/ml/p3-ims-obd-eagle-eye/experiment/jiyoung/exp/saved/{args.now}/best.pt",
        )
