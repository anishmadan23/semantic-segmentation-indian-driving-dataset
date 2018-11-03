
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from pprint import pprint
import os


# ### Generate class_id to label map

# In[46]:


with open('label_to_color_map.txt') as f:
    label_to_color_map = json.load(f)

label_arr = np.arange(len(label_to_color_map.keys()))
id_to_label_map = {}
label_to_id_map = {}
for i,class_id in enumerate(label_arr):
    label = list(label_to_color_map.keys())[i]
    id_to_label_map[str(class_id)] = label
    label_to_id_map[label] = str(class_id)

    
print(id_to_label_map)
with open('class_id_to_label_map.txt', 'w+') as ff:
    ff.write(json.dumps(id_to_label_map))
    
with open('label_to_id_map.txt','w+') as fff:
    fff.write(json.dumps(color_to_id_map))


# ### Make ground truth images coloured as per annotations and color mapping defined

# In[18]:


prev_data_path = 'Anue_dataset/'
new_data_path = 'labelled_Anue_dataset/'
lab_path = 'gtFine'
im_path = 'leftImg8bit'                     #(This folder was copied to new_data_path as it is)
modes = ['train','test','val'] 


#need to update gtFine in new data path by replacing json files from old dataset and extracting polygons and
#colorising the images. Then we will assign class_ids to pixel locations according to segmented class and 
# generate a map of HxWx1 size where H,W are height and width of images, thereby assigning each pixel(set of rgb)
# 1 class_id which corresponds to a particular class


for mode in modes:
    old_imgs_path = os.path.join(prev_data_path,im_path,mode)
    old_anno_path = os.path.join(prev_data_path,lab_path,mode)
    
    imgs = os.listdir(old_imgs_path)
    for img_ in imgs:
        anno_file_name = str(img_[:img_.find('_')])+str('_gtFine_polygons.json')
        anno_file = os.path.join(old_anno_path,anno_file_name)
        
        with open(anno_file) as ff:
            data = json.load(ff)
            cur_img = cv2.imread(os.path.join(old_imgs_path,img_),cv2.IMREAD_UNCHANGED)
            for i, obj in enumerate(data['objects']):
                obj_label = obj['label']
                color_tuple = tuple(label_to_color_map[obj_label])
                cv2.fillPoly(cur_img,[np.array(obj['polygon']).astype(np.int32)],color=color_tuple)
            save_path = os.path.join(new_data_path,im_path,mode,img_)
            cv2.imwrite(save_path, cur_img)
            
            


# In[87]:


labelled_gt = 'labelled_gt'
modes = ['train','test','val']
for mode in modes:
    imgs_path = os.path.join(new_data_path,im_path,mode)
    all_imgs = os.listdir(imgs_path)
    for i,img_ in enumerate(all_imgs):
            cur_img = cv2.imread(os.path.join(imgs_path,img_),cv2.IMREAD_COLOR)
            gt_img = cur_img.copy()
            for key in list(label_to_color_map.keys()):
                cur_color = np.array(label_to_color_map[key])[:3]      # removing alpha channel
                gt_img[np.where((gt_img==cur_color).all(axis=2))] = int(label_to_id_map[key])
            gt_img = gt_img[:,:,0]
#             print(gt_img.shape)
            save_name = str(img_[:img_.find('_')])+str('_id_gt.png')
            save_path = os.path.join(new_data_path,labelled_gt,mode,save_name)
            cv2.imwrite(save_path,gt_img)
        

