import os
import random
import time

import wandb
import copy
import numpy as np
import torch
from torch import nn, optim


from config import get_args
from dataset import get_dataloader
from loss import create_criterion
from optimizer import create_optimizer
from scheduler import create_scheduler
from model import get_model
from utils import seed_everything, label_accuracy_score, add_hist
from evaluation import save_model
from sklearn.model_selection import KFold
WANDB = True
# WANDB = False

def train(args,epoch,num_epochs, model, criterions, optimizer, dataloader,scheduler=None):
    model.train()
    epoch_loss = 0
    # labels = torch.tensor([]).to(args.device)
    # preds = torch.tensor([]).to(args.device)

    for step,image_data in enumerate(dataloader) :
        optimizer.zero_grad()
        # image_data = dict.values 형태  
        # image_data[0]=image -> Tuple  len(image) = batch  Tuple[0] ->Tensor (channel,height,width)
        # image_data[1]=mask -> Tuple   len(mask) = batch  Tuple[0] ->Tensor (mask_length,height,width)
        # image_data[2]=bbox -> Tuple   len(bbox) = batch  Tuple[0] ->Tensor (bbox_length,height,width)
        images=image_data[0]
        images = torch.stack(images)       # (batch, channel, height, width)

        temp_masks = image_data[1]
        # masks = np.zeros(images.shape[2:])
        masks=[]
        # 배치 사이즈 만큼 for문 
        for temp_mask in temp_masks:
            temp= np.zeros(images.shape[2:]) # width, height 크기만큼의 0 array 생성
            # 클래스 개수(12) 만큼 for문
            for mask in temp_mask:
                temp=np.maximum(mask, temp)
            masks.append(temp)
        masks=torch.tensor(masks).long()
        # masks = torch.stack(masks).long()  # (batch, channel, height, width)
        
        images, masks = images.to(args.device), masks.to(args.device)
        
        # 확률에 따라 cutmix 사용 , cutmix에서는 로스 조합 x
        if np.random.rand(1)<0.5:
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(images.size()[0]).cuda()
            target_a = masks
            target_b = masks[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            # compute output
            outputs = model(images)
            loss = criterions[1](outputs, target_a) * lam + criterions[1](outputs, target_b) * (1. - lam)
        else:
            outputs = model(images)
            flag=criterions[0]
            if flag=='+':
                loss = criterions[1](outputs, masks)+ criterions[2](outputs, masks)
            elif flag=='-':
                loss = criterions[1](outputs, masks) - criterions[2](outputs, masks)
            else:
                loss = criterions[1](outputs, masks)
        # loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()
        if (step + 1) % 10 == 0:
            current_lr = get_lr(optimizer)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} lr: {}'.format(
                    epoch+1, num_epochs, step+1, len(dataloader), loss.item(),current_lr))
        epoch_loss += loss
    if scheduler:
        scheduler.step(epoch_loss)
    return (epoch_loss / len(dataloader))


def evaluate(args, model, criterions, dataloader):
    model.eval()
    epoch_loss = 0
    n_class = 12
    with torch.no_grad():
        hist = np.zeros((n_class, n_class))
        for image_data in dataloader:

            images=image_data[0]
            images = torch.stack(images)       # (batch, channel, height, width)

            temp_masks = image_data[1]
            # masks = np.zeros(images.shape[2:])
            masks=[]
            # 배치 사이즈 만큼 for문 
            for temp_mask in temp_masks:
                temp= np.zeros(images.shape[2:]) # width, height 크기만큼의 0 array 생성
                # 클래스 개수(12) 만큼 for문
                for mask in temp_mask:
                    temp=np.maximum(mask, temp)
                masks.append(temp)
            masks=torch.tensor(masks).long()

            # images = torch.stack(images)       # (batch, channel, height, width)
            # masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(args.device), masks.to(args.device)            

            outputs = model(images)
            flag=criterions[0]
            if flag=='+':
                loss = criterions[1](outputs, masks)+ criterions[2](outputs, masks)
            elif flag=='-':
                loss = criterions[1](outputs, masks)- criterions[2](outputs, masks)
            else:
                loss = criterions[1](outputs, masks)
            epoch_loss += loss
            
            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=n_class)

        acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
        print(f'acc:{acc:.4f}, acc_cls:{acc_cls:.4f}, fwavacc:{fwavacc:.4f}')
    return (epoch_loss / len(dataloader)), mIoU


def run(args, model, criterion, optimizer, dataloader,scheduler=None):
    best_mIoU_score = 0.0
    # best_valid_loss = float("inf")

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
        # if valid_loss < best_valid_loss:
        if mIoU_score > best_mIoU_score:
                print(f'Best performance at epoch: {epoch + 1}')
                print('Save model in', args.MODEL_PATH)
                # best_valid_loss = valid_loss
                best_mIoU_score = mIoU_score
                save_model(model, args.MODEL_PATH)
        

def main(args):
    seed_everything(21)
    # load_dotenv()

    if WANDB:
        if args.ENCODER:
            run_name = args.MODEL +'_' +args.ENCODER
        else:
            run_name = args.MODEL
            
    
    if args.KFOLD > 1:
        # kfold index 생성
        fold_split = KFold(args.KFOLD, shuffle=True, random_state=21)
        index_gen = iter(fold_split.split(range(3272))) # 전체 이미지수 3272개
        # pt 저장 폴더 생성
        path_pair = args.MODEL_PATH.split('.')
        os.makedirs(path_pair[0], exist_ok=True) 
        # 재사용위해 args 복사
        args_origin = copy.deepcopy(args) 
        
    for i in range(args.KFOLD):
        # hold-out, kfold에 따라서 dataloader 다르게 설정
        if args.KFOLD > 1:
            # wandb
            if WANDB:
                args = copy.deepcopy(args_origin)
                path_pair = args_origin.MODEL_PATH.split('.')
                args.MODEL_PATH = path_pair[0] + f'/kfold_{i+1}.' + path_pair[1] # MODEL_PATH 변경
                wandb.init(project=os.environ.get('WANDB_PROJECT_NAME'), name=run_name+f'_k{i+1}', config=args, reinit=True)
                args = wandb.config
            # dataloader
            dataloader = get_dataloader(args.BATCH_SIZE, fold_index=next(index_gen))
            print(f'\nfold {i+1} start')
        else:
            # wandb
            if WANDB:
                wandb.init(project=os.environ.get('WANDB_PROJECT_NAME'), name=run_name, reinit=True)
                wandb.config.update(args)
                args = wandb.config
            # dataloader
            dataloader = get_dataloader(args.BATCH_SIZE)
        print("Get loader")

        model = get_model(args.MODEL, args.ENCODER).to(args.device)
        print("Load model")

        if WANDB:
            wandb.watch(model)
    
        criterion=[]
        if '+' in args.LOSS :
            criterion.append('+')
            criterion.append(create_criterion(args.LOSS.split('+')[0]))
            criterion.append(create_criterion(args.LOSS.split('+')[1]))        
        elif '-' in args.LOSS:
            criterion.append('-')
            criterion.append(create_criterion(args.LOSS.split('-')[0]))
            criterion.append(create_criterion(args.LOSS.split('-')[1]))   
        else:
            criterion.append('0')
            criterion.append(create_criterion(args.LOSS))
            
        if i%2==0 :
            optimizer = create_optimizer('AdamP',model,args.LEARNING_RATE)
        else:
            optimizer = create_optimizer('SGDP',model,args.LEARNING_RATE)
        
        print(optimizer)
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

def rand_bbox(size,lam):
    H=size[2]
    W=size[3]
    cut_rat=np.sqrt(1.-lam)
    cut_w=np.int(W*cut_rat)
    cut_h=np.int(H*cut_rat)

    cx=np.random.randint(W)
    cy=np.random.randint(H)

    bbx1=np.clip(cx-cut_w//2,0,W)
    bby1=np.clip(cy-cut_h//2,0,H)
    bbx2=np.clip(cx+cut_w//2,0,W)
    bby2=np.clip(cy+cut_h//2,0,H)

    return bbx1, bby1,bbx2, bby2
if __name__ == "__main__":
    args = get_args()
    torch.cuda.empty_cache()
    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (args.device))

    main(args)
