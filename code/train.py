import os
import random
import time

import wandb

import numpy as np
import torch
from torch import nn, optim


from config import get_args
from dataset import get_dataloader
from loss import create_criterion
from optimizer import create_optimizer
from scheduler import create_scheduler
from model import get_model
from utils import seed_everything, label_accuracy_score, add_hist, get_miou
from evaluation import save_model

WANDB = True

def train(args,epoch,num_epochs, model, criterion, optimizer, dataloader,scheduler=None):
    model.train()
    epoch_loss = 0
    # labels = torch.tensor([]).to(args.device)
    # preds = torch.tensor([]).to(args.device)

    for step,(images, masks, _) in enumerate(dataloader) :
        optimizer.zero_grad()

        images = torch.stack(images)       # (batch, channel, height, width)
        masks = torch.stack(masks).long()  # (batch, channel, height, width)
        
        images, masks = images.to(args.device), masks.to(args.device)
                
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()
        if (step + 1) % 25 == 0:
            current_lr = get_lr(optimizer)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} lr: {}'.format(
                    epoch+1, num_epochs, step+1, len(dataloader), loss.item(),current_lr))
        epoch_loss += loss
    if scheduler:
        scheduler.step()
    return (epoch_loss / len(dataloader))


def evaluate(args, model, criterion, dataloader):
    model.eval()
    epoch_loss = 0
    n_class = 12
    with torch.no_grad():
        hist = np.zeros((n_class, n_class))
        miou_all = []
        for images, masks, _ in dataloader:
            
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(args.device), masks.to(args.device)            

            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss
            
            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=n_class)

            # miou 저장
            miou_list = get_miou(masks.detach().cpu().numpy(), outputs, n_class=n_class)
            miou_all.extend(miou_list)

        acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)

        # TODO 아래 miou 사용 확정시 label_accuracy_score 수정 필요
        # 새로운 miou 사용
        mIou = np.nanmean(miou_all)

        print(f'acc:{acc:.4f}, acc_cls:{acc_cls:.4f}, fwavacc:{fwavacc:.4f}')
    return (epoch_loss / len(dataloader)), mIoU


def run(args, model, criterion, optimizer, dataloader,scheduler=None):
    best_valid_loss = float("inf")

    train_loader, val_loader = dataloader

    for epoch in range(args.EPOCHS):

        train_loss = train(args,epoch,args.EPOCHS, model, criterion, optimizer, train_loader,scheduler)

        valid_loss, mIoU_score = evaluate(args, model, criterion, val_loader)
        
        if WANDB:
            wandb.log({
                "train_loss": train_loss, 
                "valid_loss": valid_loss,
                "mIoU": mIoU_score
                })
            

        print(f"epoch:{epoch+1}/{args.EPOCHS} train_loss: {train_loss:.4f} valid_loss: {valid_loss:.4f} mIoU: {mIoU_score:.4f}")
        if valid_loss < best_valid_loss:
                print(f'Best performance at epoch: {epoch + 1}')
                print('Save model in', args.MODEL_PATH)
                best_valid_loss = valid_loss
                save_model(model, args.MODEL_PATH)
        

def main(args):
    seed_everything(21)
    if WANDB:
        wandb.init(project="stage-3", reinit=True)
        if args.ENCODER:
            wandb.run.name = args.MODEL +'_' +args.ENCODER
        else:
            wandb.run.name = args.MODEL
        wandb.config.update(args)

        args = wandb.config

    dataloader = get_dataloader(args.BATCH_SIZE)
    print("Get loader")

    model = get_model(args.MODEL,args.ENCODER).to(args.device)
    print("Load model")

    if WANDB:
        wandb.watch(model)

    criterion = create_criterion(args.LOSS)
    optimizer = create_optimizer(args.OPTIMIZER,model,args.LEARNING_RATE)
    if args.SCHEDULER:
        scheduler = create_scheduler(args.SCHEDULER,optimizer)
    else:
        scheduler = None
    # optimizer = optim.Adam(params = model.parameters(), lr = args.LEARNING_RATE, weight_decay=1e-6)
    
    print("Run")
    run(args, model, criterion, optimizer ,dataloader,scheduler)
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
if __name__ == "__main__":
    args = get_args()
    torch.cuda.empty_cache()
    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (args.device))

    main(args)
