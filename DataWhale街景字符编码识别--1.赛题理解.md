## DataWhale街景字符编码识别--1.赛题理解

任务场景源自于现实生活门牌号识别，数据集出自Google街景图像中的门牌号数据集（The Street View House Numbers Dataset, SVHN），并根据一定方式采样得到。



### 1.SVHN Dataset

SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images. For every img in train and validation dataset, we have a corresponding label which contains information of number sequence value, x and y value and width and height of digit zone bounding box.

For this competition, the dataset is a subset of SVHN which is sampled by a certain method.



### 2.Objective

Given an image from the dataset, we need to detect what numer sequence, the length of which is uncertain, like 4, 72, is in the context. Therefore, it is a OCR(Optical Character Recognition) problem. 

There are many available method to deal with it, such as CTC(Connectionist Temporal Classification), Attention mechanism, CRNN(convlutional Layers + recurrent Layers), SSD and YOLO.



### 3.Baseline

It is easy to find that the maximum length of number sequence is 6, hence if we add one supplement character (eg. X here) or more to the end of every number sequence to increase its length to 6, the problem can be simplified as a multi-classification problem.

For example, if the original number sequence is 72 whose length is less than 6, employing method above, we will obtain a new number sequence as 72XXXX. 

More specifically, for every digit in the new number sequence, it can be one of  11 possibilities from number 0 to 9 plus special character X. Therefore, baseline employs a typical img classification model ResNet-18 as a pre-trained model and add 6 x 11 classifier behind it to get results.

