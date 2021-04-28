import os
import tqdm

import torch
import numpy as np
import pandas as pd
import albumentations as A

from dataset import get_testloader
from model import get_model


def load_model(model_name: str, device):
    base = model_name.split("_")[0]
    model = get_model(base)
    model_path = f"/opt/ml/code/saved/{model_name}"

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded model:{model_name}")

    return model


def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


if __name__ == "__main__":
    print("Start")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # sample_submisson.csv 열기
    submission = pd.read_csv('/opt/ml/code/submission/sample_submission.csv', index_col=None)

    test_loader = get_testloader()

    model_paths = os.listdir("/opt/ml/code/saved")

    for mp in model_paths:
        if mp.endswith(".pt"):
            model = load_model(mp, device)
            model.to(device)

            # test set에 대한 prediction
            file_names, preds = test(model, test_loader, device)

            # PredictionString 대입
            for file_name, string in zip(file_names, preds):
                submission = submission.append({
                    "image_id" : file_name, 
                    "PredictionString" : ' '.join(str(e) for e in string.tolist()
                    )}, ignore_index=True)

            submission_name = mp[:-3]
            submission.to_csv(f"/opt/ml/code/submission/{submission_name}.csv", index=False)
    print("Finish")