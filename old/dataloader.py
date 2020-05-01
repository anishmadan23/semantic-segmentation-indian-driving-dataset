from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import scipy.misc as smi
import copy
import numpy as np
import skimage.io as io
from torch.utils import data
from PIL import Image
from sklearn.utils import shuffle
import cv2

g_seg_path = 'gt_labels/'
g_img_path = 'leftImg8bit_orig/'

class MyDataset(data.Dataset):
	def __init__(self, root, transforms, phase):
		self.labels = 39
		self.transforms = transforms
		img_path = g_img_path+phase
		seg_path = g_seg_path+phase
		self.greypath = os.path.join(root, img_path)
		self.segpath = os.path.join(root, seg_path)
		greyimg = os.listdir(self.greypath)
		segimg = os.listdir(self.segpath)
		self.greyimg, self.segimg = shuffle(greyimg, segimg, random_state=0)

	def __len__(self):
		return len(self.greyimg)

	def __getitem__(self, index):
		img = Image.open(os.path.join(self.greypath, self.greyimg[index]))
		img = self.transforms(img)
		# print(img.shape)
		temp = cv2.imread(os.path.join(self.segpath, self.segimg[index]))
		temp = cv2.resize(temp, (128,128), cv2.INTER_NEAREST)
		seg  = torch.from_numpy(temp)
		# seg = Image.open(os.path.join(self.segpath, self.segimg[index]))
		# seg = seg.resize((128,128),resample=1)
		# seg_tsor = torch.tensor(seg,requires_grad=False,dtype = torch.LongTensor)

		# print(seg.shape)
		seg = seg.unsqueeze(0)

		imglabel = np.unique(seg.cpu().numpy())
		print(imglabel)
		# seg = seg.unsqueeze(0)

		return img, seg
# 


