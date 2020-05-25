## DataWhale街景字符编码识别--2.数据增强

## 1.Data Augmentation

Sometimes the data we use for training and testing are not always variable, hence the model we get will be bad in robustness and generalization ability which will result in a serious overfitting. In deep learning framework generally speaking, the more abundant the data are, the exacter predication we can obtain from the model. We will be closer to the rule of data then. 

To build useful deep learning models, the validation error must continue to decrease with the training error. Data augmentation is a very powerful method of achieving this. The augmented data will represent a more comprehensive set of possible data points, thus minimizing the distance between the training and validation set, as well as any future tesing data.

Here list some common techniques for data augmentation in img data applications. 

### a. gemetric transformations

### b. flipping

### c. color space

### d. cropping

### e. Ratation

### f. translation

### g. noise injection

### h. Color space transformation

### i. Kernel filter

These augmentations can be also combined together. In addition, there exists a potential problem called 'safety of augmentation methods' which refers to their likelihood of preserving the label post-transformation. 

## 

## 2. Pytorch Augmentation Techniques

In pytorch framework, the augmentations can be achieved via torchvision.transforms API, like function CenterCrop, ColorJitter, FiveCrop, Grayscale, Pad, RandomAffine, RandomApply, RandomCrop, RandomRotation,Resize etc. All of functions above can be viewed as two different cliques. 

### a. 'on the fly' 

This type like randomaffine, randomcrop means the amount of data of every epoch is unchanged after augmentation. But the data are different between every two epochs. Therefore, it can be viewed as increasing the data.

### b.  'normal'

This type like FiveCrop really increases the data amount.