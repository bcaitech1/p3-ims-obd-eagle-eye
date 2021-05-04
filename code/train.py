import os
import random
import time
from dotenv import load_dotenv
import wandb
import numpy as np
import torch
from torch import nn, optim
from config import get_args
from loss import create_criterion
from optimizer import create_optimizer
from scheduler import create_scheduler
from model import get_model
from utils import seed_everything, label_accuracy_score, add_hist, get_miou
from evaluation import save_model
from sklearn.model_selection import KFold
from dataset import get_dataloader
import copy

WANDB = True

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

def denormalize_image(image, mean, std):
    img_cp = image.copy()
    img_cp *= std 
    img_cp += mean
    img_cp *= 255.0
    img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
    return img_cp

def labels():
    l = {}
    for i, label in enumerate(category_names):
        l[i] = label
    return l

def wb_mask(bg_img, pred_mask, true_mask):
      return wandb.Image(bg_img, masks={
    "prediction" : {"mask_data" : pred_mask, "class_labels" : labels()},
    "ground truth" : {"mask_data" : true_mask, "class_labels" : labels()}})

def train(
    args, epoch, num_epochs, model, criterions, optimizer, dataloader, scheduler=None
):
    model.train()
    epoch_loss = 0
    # labels = torch.tensor([]).to(args.device)
    # preds = torch.tensor([]).to(args.device)

    for step, (images, masks, _) in enumerate(dataloader):
        optimizer.zero_grad()

        images = torch.stack(images)  # (batch, channel, height, width)
        masks = torch.stack(masks).long()  # (batch, channel, height, width)

        images, masks = images.to(args.device), masks.to(args.device)

        outputs = model(images)
        flag = criterions[0]
        if flag == "+":
            loss = criterions[1](outputs, masks) + criterions[2](outputs, masks)
        elif flag == "-":
            loss = criterions[1](outputs, masks) - criterions[2](outputs, masks)
        else:
            loss = criterions[1](outputs, masks)
        # loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()
        if (step + 1) % args.LOG_INTERVAL == 0:
            current_lr = get_lr(optimizer)
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} lr: {}".format(
                    epoch + 1,
                    num_epochs,
                    step + 1,
                    len(dataloader),
                    loss.item(),
                    current_lr,
                )
            )
        epoch_loss += loss
    if scheduler:
        scheduler.step()
    return epoch_loss / len(dataloader)


def evaluate(args, model, criterions, dataloader):
    model.eval()
    epoch_loss = 0
    n_class = 12
    example_images = []
    with torch.no_grad():
        hist = np.zeros((n_class, n_class))
        miou_all = []
        for images, masks, _ in dataloader:

            images = torch.stack(images)  # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(args.device), masks.to(args.device)

            outputs = model(images)
            flag = criterions[0]
            if flag == "+":
                loss = criterions[1](outputs, masks) + criterions[2](outputs, masks)
            elif flag == "-":
                loss = criterions[1](outputs, masks) - criterions[2](outputs, masks)
            else:
                loss = criterions[1](outputs, masks)
            epoch_loss += loss

            inputs_np = torch.clone(images).detach().cpu().permute(0,2,3,1).numpy()
            inputs_np = denormalize_image(inputs_np, mean=(0.4611, 0.4403, 0.4193), std=(0.2107, 0.2074, 0.2157))

            example_images.append(wb_mask(inputs_np[0],
                                pred_mask = outputs.argmax(1)[0].detach().cpu().numpy(),
                                true_mask = masks[0].detach().cpu().numpy())
            )

            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

            hist = add_hist(
                hist, masks.detach().cpu().numpy(), outputs, n_class=n_class
            )

            # miou 저장
            miou_list = get_miou(masks.detach().cpu().numpy(), outputs, n_class=n_class)
            miou_all.extend(miou_list)

        acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)

        # TODO 아래 miou 사용 확정시 label_accuracy_score 수정 필요
        # 새로운 miou 사용
        mIou = np.nanmean(miou_all)

        print(f"acc:{acc:.4f}, acc_cls:{acc_cls:.4f}, fwavacc:{fwavacc:.4f}")
    return (epoch_loss / len(dataloader)), mIoU , example_images



def run(args, model, criterion, optimizer, dataloader, fold, scheduler=None):
    best_mIoU_score = 0.0

    train_loader, val_loader = dataloader

    for epoch in range(args.EPOCHS):

        train_loss = train(
            args,
            epoch,
            args.EPOCHS,
            model,
            criterion,
            optimizer,
            train_loader,
            scheduler,
        )

        valid_loss, mIoU_score, example_images = evaluate(args, model, criterion, val_loader)

        if WANDB:
            wandb.log(
                {"train_loss": train_loss, "valid_loss": valid_loss, "mIoU": mIoU_score, "predictions" : example_images}
            )

        if args.CHECKPOINT and not ((epoch + 1) % args.CHECKPOINT):
            if args.KFOLD > 1:
                path = f"{args.CHECKPOINT_PATH}/{args.MODEL}_{args.ENCODER}_fold_{fold+1}_epoch_{epoch+1}_miou_{mIoU_score:.3f}.pt"
            else:
                path = f"{args.CHECKPOINT_PATH}/{args.MODEL}_{args.ENCODER}_epoch_{epoch+1}_miou_{mIoU_score:.3f}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "valid_loss": valid_loss,
                    "mIoU": mIoU_score,
                },
                path,
            )
            print(f"Save the checkpoint at {path}")

        print(
            f"epoch:{epoch+1}/{args.EPOCHS} train_loss: {train_loss:.4f} valid_loss: {valid_loss:.4f} mIoU: {mIoU_score:.4f}"
        )

        if mIoU_score > best_mIoU_score:
            print(f"Best performance at epoch: {epoch + 1}")
            print("Save model in", args.MODEL_PATH)
            best_mIoU_score = mIoU_score
            save_model(model, args.MODEL_PATH)


def main(args):
    seed_everything(21)
    load_dotenv()

    if WANDB:
        if args.ENCODER:
            run_name = args.MODEL + "_" + args.ENCODER
        else:
            run_name = args.MODEL

    if args.KFOLD > 1:
        # kfold index 생성
        fold_split = KFold(args.KFOLD, shuffle=True, random_state=21)
        index_gen = iter(fold_split.split(range(3272)))  # 전체 이미지수 3272개
        # pt 저장 폴더 생성
        path_pair = args.MODEL_PATH.split(".")
        os.makedirs(path_pair[0], exist_ok=True)
        # 재사용위해 args 복사
        args_origin = copy.deepcopy(args)

    for fold in range(args.KFOLD):
        # hold-out, kfold에 따라서 dataloader 다르게 설정
        if args.KFOLD > 1:
            # wandb
            if WANDB:
                args = copy.deepcopy(args_origin)
                path_pair = args_origin.MODEL_PATH.split(".")
                args.MODEL_PATH = (
                    path_pair[0] + f"/kfold_{fold+1}." + path_pair[1]
                )  # MODEL_PATH 변경
                wandb.init(
                    project=os.environ.get("WANDB_PROJECT_NAME"),
                    name=run_name + f"_k{fold+1}",
                    config=args,
                    reinit=True,
                )
                args = wandb.config
            # dataloader
            dataloader = get_dataloader(args.BATCH_SIZE, fold_index=next(index_gen))
            print(f"\nfold {fold+1} start")
        else:
            # wandb
            if WANDB:
                wandb.init(
                    project=os.environ.get("WANDB_PROJECT_NAME"),
                    name=run_name,
                    reinit=True,
                )
                wandb.config.update(args)
                args = wandb.config
            # dataloader
            dataloader = get_dataloader(args.BATCH_SIZE)
        print("Get loader")

        model = get_model(args.MODEL, args.ENCODER).to(args.device)
        print("Load model")

        if WANDB:
            wandb.watch(model)

        criterion = []
        if "+" in args.LOSS:
            criterion.append("+")
            criterion.append(create_criterion(args.LOSS.split("+")[0]))
            criterion.append(create_criterion(args.LOSS.split("+")[1]))
        elif "-" in args.LOSS:
            criterion.append("-")
            criterion.append(create_criterion(args.LOSS.split("-")[0]))
            criterion.append(create_criterion(args.LOSS.split("-")[1]))
        else:
            criterion.append("0")
            criterion.append(create_criterion(args.LOSS))
        optimizer = create_optimizer(args.OPTIMIZER, model, args.LEARNING_RATE)
        if args.SCHEDULER:
            scheduler = create_scheduler(args.SCHEDULER, optimizer)
        else:
            scheduler = None
        # optimizer = optim.Adam(params = model.parameters(), lr = args.LEARNING_RATE, weight_decay=1e-6)

        print("Run")
        run(args, model, criterion, optimizer, dataloader, fold, scheduler)


def get_lr(optimizer):
    """
        현재 learning rate 리턴
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


if __name__ == "__main__":
    args = get_args()
    torch.cuda.empty_cache()
    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (args.device))

    main(args)
