# Semantic Segmentation on Indian Driving Dataset

## Team Members: Anish Madan, Apoorv Khattar

## About the project
We perform semantic segmentation using FCN8s and SegNet on the Indian Driving Dataset and compare the performance of the two models based on accuracy and IOU score.

## Dataset
The Indian Driving Dataset consists of 6906 and 979 high resolution images in the training and validation set. There are a total of 39 unique class labels. We work with a subset consisting of 1000 training and 100 test samples for our project.

## Data Preprocessing
For each image, we have a json file containing the number of different classes in the image and the polygon vertices for segmentation map of each class. We simplify this directory structure as follows:
  - img
    - train
    -val
  - seg
    - train
    -val
We create segmentation maps as .png files and store them in the seg directory. Refer to [Preprocessing.ipynb](https://github.com/anishmadan23/semantic-segmentation-indian-driving-dataset/blob/master/Preprocessing.ipynb) to understand how the preprocessing is done.

## Models
We refer to the implementaion by [zijundeng](https://github.com/zijundeng/pytorch-semantic-segmentation) for FCN8s and SegNet architectures. We use a pretrained VGG16 and pretrained VGG19 with batch normalization layers as a feature extractor for FCN8s and SegNet respectively.

The weights for the trained models are available [here]()

## Results
#### Accuracy And IOU Scores
The following summarises the accuracy and IOU for the trained networks on the validation set:
| Model  | Accuracy | IOU |
| ------------- | ------------- | ------------- |
| FCN8s  |   |   |
| SegNet |   |   |

#### Qualitative Results for FCN8s
The following are the visualizations of the output for the FCN8s (from left to right- input, ground truth, model output):

#### Qualitative Results for SegNet
The following are the visualizations of the output for the SegNet (from left to right- input, ground truth, model output):

## Acknowledgements
We would like to thank Dr. Saket Anand for providing us with the Indian Driving Dataset for this project. We would also like to thank Zijun Deng for making their repository on semantic segmentation publicly available which we have referred to for FCN and SegNet architectures.

###### This work was done as part of our project for CSE343: Machine Learning course at IIIT Delhi.
