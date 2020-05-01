
# coding: utf-8

# In[26]:


import os
import shutil
import numpy as np


# In[22]:


new_datasetPath = 'Anue_dataset/'
fine_labels_ = 'gtFine'
leftImgsDir = 'leftImg8bit'
modes = ['train','val']
data_dirs = ['Dataset','Dataset_2','Dataset_3','Dataset_4','Dataset_5','Dataset_6','Dataset_7','Dataset_8']
# data_dirs = ['Dataset']
tmpAll_dir = os.path.join(new_datasetPath,'tmpAll')


# In[23]:


img_anno_arr = []


for data_dir in data_dirs:
    fine_label_path = os.path.join(data_dir,'anue',fine_labels_)
    leftImgPath = os.path.join(data_dir,'anue',leftImgsDir)
    
    for mode in modes:
        fine_label_path_mode = os.path.join(fine_label_path,mode)
        leftImgPath_mode = os.path.join(leftImgPath,mode)
        all_labels_dir = sorted(os.listdir(fine_label_path_mode))
        all_imgs_dir = sorted(os.listdir(leftImgPath_mode))
        common_dirs = sorted(list(set(all_labels_dir).intersection(all_imgs_dir)))
        
        for com_dir in common_dirs:
            all_fine_labels_path = os.path.join(fine_label_path_mode,com_dir)
            all_imgs_path = os.path.join(leftImgPath_mode,com_dir)
            print()
            print(com_dir)
            annots = os.listdir(all_fine_labels_path)
            imgss = os.listdir(all_imgs_path)
            
            img_arr = [img_name[:img_name.find('_')] for img_name in imgss]
            for anno in annots:
                if anno[:anno.find('_')] in img_arr:
                    img_anno_arr.append(anno[:anno.find('_')])
        
                    whole_anno_path = os.path.join(all_fine_labels_path,anno)
                    whole_img_name = str(anno[:anno.find('_')])+str('_leftImg8bit.png')
                    whole_img_path = os.path.join(all_imgs_path,whole_img_name)
                    
                    shutil.copy(whole_anno_path,tmpAll_dir)
                    shutil.copy(whole_img_path,tmpAll_dir)
                    
                    
                
                
                
                
                
                


# In[56]:


p = sorted(os.listdir(tmpAll_dir))
pk = [int(name[:name.find('_')]) for name in p]
pk = np.unique(pk)
kk = sorted(np.unique(np.array(img_anno_arr).astype(int)))
pk[pk==kk]=1
sum(pk)


# In[57]:


new_modes = ['train','val','test']
np.random.shuffle(img_anno_arr)
test_split_idx = int(0.8*len(img_anno_arr))
trainval_elemts = img_anno_arr[:test_split_idx]
test_elemts = img_anno_arr[test_split_idx:]
val_split_idx = int(0.8*len(trainval_elemts))
train_elemts = trainval_elemts[:val_split_idx]
val_elemts = trainval_elemts[val_split_idx:]

print(len(train_elemts),len(val_elemts),len(test_elemts))
dset = [train_elemts,val_elemts,test_elemts]
for idx,mode in enumerate(new_modes):
    new_fine_label_path = os.path.join(new_datasetPath,fine_labels_,mode)
    new_leftImg_path = os.path.join(new_datasetPath,leftImgsDir,mode)
    
    for elemt in dset[idx]:
        new_label = str(elemt)+str('_gtFine_polygons.json')
        new_img = str(elemt)+str('_leftImg8bit.png')
        new_lab_path = os.path.join(new_fine_label_path,new_label)
        new_img_path = os.path.join(new_leftImg_path,new_img)
        
        shutil.copy(os.path.join(tmpAll_dir,new_label),new_lab_path)
        shutil.copy(os.path.join(tmpAll_dir, new_img), new_leftImg_path)
    
    

