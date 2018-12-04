from PIL import Image
import os
import numpy as np 
import cv2

random_imgs = os.listdir('data/gt_labels/test/')

rand_img = cv2.imread(os.path.join('data/gt_labels/test',random_imgs[34]),cv2.IMREAD_COLOR)
# r_img = cv2.imread('gt_img.png')
print(np.unique(rand_img))