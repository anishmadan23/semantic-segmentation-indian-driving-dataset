
# coding: utf-8

# In[11]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from pprint import pprint
import os


# In[12]:


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


# In[13]:


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


# In[25]:


img = np.array(Image.open('052637_leftImg8bit.png'))
# colorize_mask(img)
plt.imshow(img)

img1 = cv2.imread('052637_leftImg8bit.png',cv2.IMREAD_UNCHANGED)


# In[27]:


lab_gen_dir = 'Anue_dataset/gtFine/train/'
all_jsons = os.listdir(lab_gen_dir)
all_labels = []
for json_i in all_jsons:
    json_path =  os.path.join(lab_gen_dir,json_i)
    with open(json_path) as f:
        data = json.load(f)
    
    for i,obj in enumerate(data['objects']):
        if(obj['label'] not in all_labels):
            all_labels.append(obj['label'])
    #     print(obj['label'])
    #     print(np.array(obj['polygon']))
#         c = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
#         cv2.fillPoly(img1,[np.array(obj['polygon']).astype(np.int32)],color=c)
# cv2.imwrite('img1.png',img1)
# print(len(all_labels))
# print(all_labels)
colours = []
for i in range(len(all_labels)):
    colours.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255),0))

label_to_color_dict = {}
for i in range(len(all_labels)):
    label_to_color_dict[all_labels[i]]  = colours[i]
# print(label_to_color_dict)
with open('label_to_color_map.txt','w+') as f:
     f.write(json.dumps(label_to_color_dict))

