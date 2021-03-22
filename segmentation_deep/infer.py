from dataloader import PortraitDataloader, PortraitToInferloader, mean, std
import torch
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
import cv2

#base_path = Path(__file__).parent.parent
data_path = Path("../APDrawingDB/data/").resolve()

#df = pd.read_csv(data_path / "my_metadata.csv")

df = pd.DataFrame()
# location of original and mask image
img_fol = data_path / "test_imgs"
#mask_fol = data_path / "mytrain_masks_bw-128"
df['image_name'] = list(os.listdir(img_fol))


# test_dataloader = PortraitDataloader(df, img_fol, mask_fol, mean, std, "val", 1, 4)
test_dataloader = PortraitToInferloader(img_fol, mean, std, 1, 4)
ckpt_path =  "./model_office_384x384_effnet_2.pth"

device = torch.device("cuda")
#model = smp.Unet("resnext50_32x4d", encoder_weights=None, classes=1, activation='sigmoid')
model = smp.Unet("timm-efficientnet-b4", encoder_weights=None, classes=1, activation=None)
model.to(device)
model.eval()
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])

# start prediction
tta = 1

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

with torch.no_grad():
    #preds_ = []
    for i, batch in enumerate(test_dataloader):
        preds_ = []
        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
        #fig.suptitle("predicted_mask//original_mask")
        print("i", i)
        images, mask_target, name = batch
        for _ in range(tta):
            preds = model(images.to(device))
            #print(preds)
            preds_ += [preds.detach().cpu().numpy()]
        preds = np.mean(preds_, axis=0)
        preds = torch.Tensor(preds)
        #batch_preds = torch.sigmoid(preds)
        batch_preds = normPRED(preds)
        batch_preds = batch_preds.detach().cpu().numpy()
        #print(batch_preds)
        batch_preds = np.squeeze(batch_preds)
        #mask_target = np.squeeze(mask_target)

        #pred1 = np.where(batch_preds==1, mask_target[:,:,0], 0)
        #pred2 = np.where(batch_preds==1, mask_target[:,:,1], 0)
        #pred3 = np.where(batch_preds==1, mask_target[:,:,2], 0)
        #pred = np.stack([pred1, pred2, pred3], axis=2)
        #print(name)
        #ax1.imshow(batch_preds, cmap="gray")
        #ax1.imshow(pred) # if using own dataset
        #ax1.imshow(batch_preds, cmap="gray")
        batch_preds = np.where(batch_preds>0., batch_preds, 0)
        pred = 1. - np.float32(batch_preds)
        #print(pred.shape)
        #print(np.float32(pred))
        #print('../results/'+name[0])
        #plt.imshow(pred, cmap='gray')
        #plt.show()
        cv2.imwrite('../results/'+name[0], pred*255)
        #break
        #ax2.imshow(np.squeeze(mask_target), cmap="gray")
        #plt.show()
