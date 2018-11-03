
# coding: utf-8

# In[8]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from pprint import pprint
import os


# In[10]:


data_path = 'Anue_dataset/'
labels_path = 'gtFine/'
imgs_path = 'leftImg8bit/'
modes = ['train']
for mode in modes:
    all_imgs = os.listdir(os.path.join(data_path,imgs_path,mode))
    some_imgs = sorted(all_imgs[:10])
    anno_arr = []
    for img_ in some_imgs:
        anno_arr.append(str(img_[:img_.find('_')])+str('_gtFine_polygons.json'))
    anno_arr = sorted(anno_arr)
    cur_base_img_path = os.path.join(data_path,imgs_path,mode)
    cur_base_anno_path = os.path.join(data_path,labels_path,mode)
    with open('label_to_color_map.txt') as f:
        p = json.load(f)
    
    print(anno_arr[0])
        
    for idx,anno in enumerate(anno_arr):
        anno_file = os.path.join(cur_base_anno_path,anno)
        
        with open(anno_file) as ff:
            data = json.load(ff)
            cur_img = cv2.imread(os.path.join(cur_base_img_path,some_imgs[idx]),cv2.IMREAD_UNCHANGED)
            for i, obj in enumerate(data['objects']):
                obj_label = obj['label']
                color_tuple = tuple(p[obj_label])
#                 print(color_tuple)
#                 print(obj['polygon'])
                cv2.fillPoly(cur_img,[np.array(obj['polygon']).astype(np.int32)],color=color_tuple)
            img_sve_name = str(idx)+str('.png')
            cv2.imwrite(img_sve_name, cur_img)
    

