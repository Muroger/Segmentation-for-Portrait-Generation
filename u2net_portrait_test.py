import os
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
import glob

from data_loader import SalObjDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+'/'+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net_portrait'#u2netp


    image_dir = './test_data/test_portrait_images/your_portrait_im'
    prediction_dir = './test_data/test_portrait_images/your_portrait_results'
    if(not os.path.exists(prediction_dir)):
        os.mkdir(prediction_dir)

    model_dir = './saved_models/u2net_portrait/u2net_portrait.pth'

    #model_dir = './saved_models/u2net/u2net_bce_itr_8000_train_1.003468_tar_0.108501.pth'
    img_name_list = glob.glob(image_dir+'/*')
    print("Number of images: ", len(img_name_list))

    # --------- 2. dataloader ---------
    #1. dataloader
    transforms = A.Compose([
                    A.Resize(512,512),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()]
                    , p=1.0)
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms)

                              
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------

    print("...load U2NET---")
    net = U2NET(3,1)

    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        #pred = 1.0 - d1[:,0,:,:]#reverse black-white
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
