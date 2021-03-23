import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import numpy as np
import glob

from data_loader import SalObjDataset
from pathlib import Path

from model import U2NET
from model import U2NETP
import os
import cv2
# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    #print('labels_v.shape', labels_v.shape)
    #print(d0.shape)
    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    #print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data,loss1.data,loss2.data,loss3.data,loss4.data,loss5.data,loss6.data))

    return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'

data_dir = os.path.join(os.getcwd(), 'APDrawingDB/data' + os.sep)
tra_image_dir = os.path.join('train_imgs' + os.sep)
tra_label_dir = os.path.join('train_masks' + os.sep)

val_image_dir = os.path.join('test_imgs' + os.sep)
val_label_dir = os.path.join('test_masks' + os.sep)

image_ext = '.png'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
Path(model_dir).mkdir(parents=True, exist_ok=True)


epoch_num = 200
batch_size_train = 6
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
tra_lbl_name_list = glob.glob(data_dir + tra_label_dir + '*' + image_ext)
val_img_name_list = glob.glob(data_dir + val_image_dir + '*' + image_ext)
val_lbl_name_list = glob.glob(data_dir + val_label_dir + '*' + image_ext)
#print(tra_img_name_list)
# tra_lbl_name_list = []
# for img_path in tra_img_name_list:
#     img_name = img_path.split(os.sep)[-1]
#     aaa = img_name.split(".")
#     bbb = aaa[0:-1]
#     imidx = bbb[0]
#     for i in range(1,len(bbb)):
#         imidx = imidx + "." + bbb[i]

#     tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)
    
print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("test images: ", len(val_img_name_list))
print("test labels: ", len(val_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)



import albumentations as A
from albumentations.pytorch import ToTensorV2
transforms = A.Compose([
                A.RandomResizedCrop(288,288, p=1.),
                #A.HorizontalFlip(p=0.5),
                #A.VerticalFlip(p=0.5),
                # A.RandomRotate90(),
                # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
                #                  border_mode=cv2.BORDER_REFLECT),
                # A.OneOf([
                #     A.OpticalDistortion(p=0.3),
                #     A.GridDistortion(p=.1),
                #     A.IAAPiecewiseAffine(p=0.3),
                # ], p=0.3),
                # A.OneOf([
                #     A.HueSaturationValue(10,15,10),
                #     A.CLAHE(clip_limit=2),
                #     A.RandomBrightnessContrast(),            
                # ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.),
                ToTensorV2()
            ], p=1.0)

transforms_val = A.Compose([
                A.Resize(512,512, p=1.),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.),
                ToTensorV2()
            ], p=1.0)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms)
    # transform=transforms.Compose([
    #     RescaleT(320),
    #     RandomCrop(288),
    #     ToTensorLab(flag=0)]))

salobj_dataset_val = SalObjDataset(
    img_name_list=val_img_name_list,
    lbl_name_list=val_lbl_name_list,
    transform=transforms_val)


salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4)
salobj_dataloader_val = DataLoader(salobj_dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=4)

# ------- 3. define model --------
# define the net
if(model_name=='u2net'):
    net = U2NET(3, 1)
    #net.load_state_dict(torch.load('./saved_models/u2net/u2net_bce_itr_10500_train_1.407865_tar_0.184744.pth'))

elif(model_name=='u2netp'):
    net = U2NETP(3,1)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-6)#, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6, last_epoch=-1)
# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 500 # save the model every 2000 iterations
best_loss = float("inf")


for epoch in range(0, epoch_num):
    net.train()
    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        #print(d0.shape, d1.shape, d2.shape, d3.shape, d4.shape, d5.shape, d6.shape)
        #print(labels_v.shape)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()
        
        # # print statistics
        running_loss += loss.data
        running_tar_loss += loss2.data

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss
        #print(scheduler.get_last_lr()[0])
        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, LR: %5f" % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val, scheduler.get_last_lr()[0]))
        # if ite_num % save_frq == 0:

        #     torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
        #     running_loss = 0.0
        #     running_tar_loss = 0.0
        #     net.train()  # resume train
        #     ite_num4val = 0
    net.eval()
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    for i, data in enumerate(salobj_dataloader_val):
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)

        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
        
        # # print statistics
        running_loss += loss.data
        running_tar_loss += loss2.data

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss
        #print(scheduler.get_last_lr()[0])
    print("[epoch: %3d/%3d] val loss: %3f, tar: %3f" % (
    epoch + 1, epoch_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

    if running_loss/ite_num4val < best_loss:
        print("******** optimal found, saving state ********")
        best_loss = running_loss/ite_num4val
        torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, best_loss, running_tar_loss / ite_num4val))
        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0

    scheduler.step()

