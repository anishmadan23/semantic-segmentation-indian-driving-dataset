import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
import scipy.misc as smi
from torchvision import transforms
import cv2


class MyDataset(data.Dataset):
	def __init__(self, X, Y, root, in_transforms, size, phase, num_classes = 39):
		seg_path = 'gt_labels/'
		img_path = 'leftImg8bit_orig/'

		self.phase = phase
		self.labels = num_classes
		self.size = size
		self.in_transforms = in_transforms
		
		# img_path = img_path+phase
		# seg_path = seg_path+phase
		
		self.colpath = os.path.join(root, img_path)
		self.segpath = os.path.join(root, seg_path)
		# self.colimg = os.listdir(self.colpath)
		# self.segimg = os.listdir(self.segpath)
		self.root = root
		self.X = X
		self.Y = Y
	def __len__(self):
		return len(self.X)

	def __getitem__(self, index):
		img = Image.open(os.path.join(self.colpath, self.phase,self.X[index]))
		img = self.in_transforms(img)

		seg = cv2.imread(os.path.join(self.segpath, self.phase,self.Y[index]))
		seg = seg[:,:,0]
		seg = cv2.resize(seg,self.size,interpolation = cv2.INTER_NEAREST)
		new_dims = (self.size[0],self.size[1],1)
		reshape_seg = seg.reshape(new_dims)

		seg_tsor =  torch.from_numpy(reshape_seg.transpose((2,0,1)))
	
		return img, seg_tsor