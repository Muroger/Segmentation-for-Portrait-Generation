# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2

#==========================dataset load==========================

class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,transform=None):
	
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):
		image = cv2.cvtColor(cv2.imread(self.image_name_list[idx]), cv2.COLOR_BGR2RGB)
		imname = self.image_name_list[idx]
		imidx = np.array([idx])

		if(len(self.label_name_list)==0):
			mask = np.zeros(image.shape)
			print('Warning: no masks was found')
		else:
			mask = cv2.imread(self.label_name_list[idx])#, cv2.IMREAD_GRAYSCALE)
			mask0 = mask[:,:,0]
			mask1 = mask[:,:,1]
			mask2 = mask[:,:,2]
			mask = np.mean([mask0, mask1, mask2], axis=0)
			#print(mask.shape)
			#plt.imshow(mask, cmap='gray')
			#plt.show()
			#mask = np.where(mask, mask>0, 0)
			
	
		if self.transform:
			sample = self.transform(image = image, mask = 1. - mask/255.)
			image, mask = sample['image'], sample['mask']

		mask = torch.unsqueeze(mask, 0)
		#mask = 1. - mask
		sample = {'imidx':imidx, 'image':image, 'label':mask}
	

		return sample
