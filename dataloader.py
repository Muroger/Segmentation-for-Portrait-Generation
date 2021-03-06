import os
from pathlib import Path
import pandas as pd
import cv2
from torch.utils.data import DataLoader, Dataset, sampler
from sklearn.model_selection import train_test_split
import torch
# image augmantation library
from albumentations import (
    Resize,
    HorizontalFlip,
    Normalize,
    Compose,
    HueSaturationValue,
    RandomContrast,
    RandomBrightness,
    RGBShift,
    CenterCrop,
    RandomGamma,
    ShiftScaleRotate,
    OpticalDistortion,
    ElasticTransform,
    RandomResizedCrop
)

from albumentations.pytorch import ToTensorV2

# imagenet mean/std will be used as the resnet backbone is trained on imagenet stats
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

imsize = 512
def get_transform(phase: str, mean: float, std: float) -> list:
    """Make a list of transformations to be used."""
    list_trans = []
    if phase == "train":
        list_trans.extend(
            [   
                RandomResizedCrop(288,288),
                #Resize(512, 512),
                HorizontalFlip(p=0.5),
                RGBShift(
                    r_shift_limit=40, g_shift_limit=-150, b_shift_limit=100, p=0.5
                ),
                HueSaturationValue(
                    hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.5
                ),
                RandomContrast(limit=0.7, p=1),
                RandomBrightness(limit=1.1, p=0.5),
                #CenterCrop(height=64, width=64, p=0.5),
                ShiftScaleRotate(
                    shift_limit=0.65, scale_limit=1.2, rotate_limit=198, p=0.5
                ),
                OpticalDistortion(distort_limit=1, shift_limit=1, p=0.5),
                ElasticTransform(alpha=255, sigma=255, alpha_affine=255, p=0.5),
                Normalize(mean=mean, std=std, p=1),
                ToTensorV2()
            ]
        )
    elif phase == "val":
        list_trans.extend(
            [   
                Resize(512, 512, p=1.),
                Normalize(mean=mean, std=std, p=1.),
                ToTensorV2()
                # RGBShift(r_shift_limit=80, g_shift_limit=150, b_shift_limit=80, p=0.2),
                # HueSaturationValue(
                #     hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.2
                # ),
                # RandomBrightness(limit=1.1, p=0.2),
                # RandomContrast(limit=0.7, p=0.2),
            ]
        )

    #list_trans.extend(
    #    [Normalize(mean=mean, std=std, p=1), ToTensorV2(),]
    #)  # normalizing the data & then converting to tensors
    list_trans = Compose(list_trans)
    return list_trans


class PortraitDataset(Dataset):
    def __init__(self, df, img_fol, mask_fol, mean, std, phase):
        self.fname = df.iloc[:, 0].values.tolist()
        self.img_fol = img_fol
        self.mask_fol = mask_fol
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transform = get_transform(phase, mean, std)

    def __getitem__(self, idx):
        name = self.fname[idx]
        img_name_path = os.path.join(self.img_fol, name)# + "_rgb.png")
        # mask_name_path = (
        #     img_name_path.split(".")[0]
        #     .replace("train-256", "train_masks_bw-256")
        #     .replace("_rgb", "_label.png")
        # )
        # mask_name_path = (
        #     img_name_path.split(".")[0]
        #     .replace("train-384", "train_masks_bw-384")
        #     .replace("_rgb", "_label.png")
        # )
        #print('img_name_path', img_name_path)
        #print('img_name_path.split(".")', img_name_path.replace("train_imgs", "train_masks"))
        mask_name_path = img_name_path.replace("_imgs", "_masks")
        #print('img_name_path', img_name_path)
        #print('mask_name_path', mask_name_path)
        img = cv2.imread(img_name_path, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_name_path, cv2.IMREAD_GRAYSCALE)  # , cv2.COLOR_BGR2RGB)
        #print('img.shape', img.shape)
        #print('mask.shape', mask.shape)
        #mask = torch.unsqueeze(mask, dim=0)
        ## this is to change background to white
        # ret, thresh = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
        # img[thresh == 0] = 255

        ## change bg to another image
        # to add: choose random images
        # base_path = Path(__file__).parent.parent
        # data_path = Path(base_path / "data/").resolve()

        # bg_name_path = str(data_path / "mydata-256/picam.png")
        # bg = cv2.imread(bg_name_path, cv2.COLOR_BGR2RGB)
        # img[mask == 0] = bg[mask == 0]

        augmentation = self.transform(image=img, mask=mask/255)
        img_aug = augmentation["image"]  # [3,128,128] type:Tensor
        mask_aug = augmentation["mask"]  # [1,128,128] type:Tensor
        mask_aug = 1. - torch.unsqueeze(mask_aug, dim=0)

        #print(mask_aug[0])
        return img_aug, mask_aug

    def __len__(self):
        return len(self.fname)


def PortraitDataloader(df, img_fol, mask_fol, mean, std, phase, batch_size, num_workers):
    """divide data into train and val and return the dataloader depending upon train or val phase."""
    #df_train, df_valid = train_test_split(df, test_size=0.2, random_state=69)
    #df = df_train if phase == "train" else df_valid
    for_loader = PortraitDataset(df, img_fol, mask_fol, mean, std, phase)
    dataloader = DataLoader(
        for_loader, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    return dataloader


class PortraitToInfer(Dataset):
    def __init__(self, img_fol, mean, std):
        self.img_fol = img_fol
        self.mean = mean
        self.std = std
        self.transform = get_transform("val", mean, std)

    def __getitem__(self, idx):
        img_name_path = str(list(self.img_fol.iterdir())[idx])
        #print("img_name_path", img_name_path)
        name = os.path.basename(img_name_path)
        img_toinfer = cv2.imread(img_name_path, cv2.COLOR_BGR2RGB)
        img_original = cv2.imread(img_name_path, cv2.COLOR_BGR2RGB)
        augmentation = self.transform(image=img_toinfer, mask=img_original)
        img_aug = augmentation["image"]  # [3,128,128] type:Tensor
        return img_aug, img_original, name

    def __len__(self):
        return len(list(self.img_fol.iterdir()))


def PortraitToInferloader(img_fol, mean, std, batch_size, num_workers):
    for_loader = PortraitToInfer(img_fol, mean, std)
    dataloader = DataLoader(
        for_loader, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    return dataloader
