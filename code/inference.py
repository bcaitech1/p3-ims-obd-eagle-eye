import os
from tqdm import tqdm
import argparse

import torch
import numpy as np
import pandas as pd
import albumentations as A

from dataset import get_testloader
from model import get_model

def load_model(args, device):
    model = get_model(args.model, args.encoder)
    model_path = args.model_dir

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded model:{args.model}")

    return model


def test(model, test_loader, device):
    size = 256
    transform = A.Compose([A.Resize(256, 256, interpolation=0)]) # maks용 interpolation 설정
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for  imgs, image_infos in tqdm(test_loader):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for mask in oms:
                transformed = transform(image=mask)
                mask = transformed['image']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


def kfold_test(args, test_loader, device):
    print('Start prediction.')
    size = 256
    transform = A.Compose([A.Resize(256, 256, interpolation=0)]) # maks용 interpolation 설정

    # 모든 모델 불러오기
    model_paths = os.listdir(args.kfold_dir)
    model_list = []
    for i, mp in enumerate(model_paths):
        args.model_dir = args.kfold_dir + f"/kfold_{i+1}.pt"
        model_list.append(load_model(args, device).to(device))

    # 
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    with torch.no_grad():
        for imgs, image_infos in tqdm(test_loader):
            test_imgs = imgs
            imgs = torch.stack(imgs).to(device)
            outs_size = list(imgs.size())
            outs_size[1] = 12 # 클래스 개수 12
            prob = np.zeros(outs_size, dtype=np.float)

            for model in model_list:
                model.eval()
                outs = model(imgs)
                outs = outs.detach().cpu().numpy()
                prob += outs

            pred = np.argmax(prob, axis=1) # soft voting

            # resize (256 x 256)
            temp = []
            for each in pred:
                pred = transform(image=each)['image']
                temp.append(pred)
            pred = np.array(temp)

            # reshape
            pred = pred.reshape([pred.shape[0], size*size]).astype(int)

            preds_array = np.vstack((preds_array, pred))
            file_name_list.append([i['file_name'] for i in image_infos])

    file_names = [y for x in file_name_list for y in x]
    print("End prediction.")

    return file_names, preds_array


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    ### 데이터 증강은 추후 dataset.py 수정등 할일이 있어 추후 update 예정
    # parser.add_argument('--augmentation', type=str, default=None, help='augmentation')
    parser.add_argument('--model', type=str, default='DeepLabV3Plus', help='model type (default: BaseModel)')
    parser.add_argument('--encoder', type=str, default='resnet18', help='model type (default: BaseModel)')
    parser.add_argument('--model_dir', type=str, default='/opt/ml/code/saved/DeepLabV3Plus_timm-efficientnet-b0_Segmentation_k1.pt')
    parser.add_argument('--kfold_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()

    print("Start")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    submission = pd.read_csv('/opt/ml/code/submission/sample_submission.csv', index_col=None)
    test_loader = get_testloader()

    if args.kfold_dir:
        file_names, preds = kfold_test(args, test_loader, device)

    else :
        # 예시
        # test_loader = get_testloader(augmentation=args.augmentation)
        model = load_model(args,device)
        model = model.to(device)
        file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({
            "image_id" : file_name, 
            "PredictionString" : ' '.join(str(e) for e in string.tolist()
            )}, ignore_index=True)

    os.makedirs(args.output_dir, exist_ok=True)
    # submission.to_csv(os.path.join(args.output_dir, f'output_{args.model}.csv'), index=False)
    submission.to_csv(os.path.join(args.output_dir, f'output.csv'), index=False)
    print("Finish")
