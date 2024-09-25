
## Medical image recognition（Identify the presence of tumors, bleeding, or health based on ultrasound /CT/MR Images) 





## Explaination of the Repositories
MedVision Innovators_Anveshan.pdf - The first phase pdf for this hackathon.

all.ipynb - Having multiple models, VGC 16, resnet50, mobilenet, inception, densenet, efficient net, nasnet. This is for brain tumor detection.

best_rf_model.joblib - The model file for Random forest results.

breastcancer.ipynb- File for detecting breast cancer.

chest.ipynb -  File for detecting lung cancer.

newtumor.ipynb - CNN model for brain tumour detection.
 
resnet50.ipynb - RESNET, densenet, mobilenet among other models for brain tumor detection.

svm.ipynb - Support vector machine and randomforest for brain tumour detection.

vgc16.ipynb - VGC 16, logistic regresion, svc, CNN , among other models for brain tumour detection.

voicetotext.ipynb - This file provides voice to text comversion code. The plan for this was to help doctors in prescribing the  medicines to patients, just to have an additional feature in our project.

## DATASET
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data

Creates 4 classses, on different types of brain tumours.
Dictionary: {'glioma_tumor': 0, 'meningioma_tumor': 1, 'no_tumor': 2, 'pituitary_tumor': 3}
Class labels: ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

## Work plan
Phase 1: Initial Setup and Planning
Requirements Gathering
Identify all the functional and non-functional requirements.
Define the scope and objectives of the project.
Gather all necessary datasets and resources (images, articles, metadata).
Project Setup

Set up version control with a repository (e.g., GitHub).
Create a virtual environment and initialize the project.
Create the initial structure for the project.
Phase 2: Software Development
Perform various models for the project to get the best accuracy.
Implement database connection and setup functions.
Image Management and classification, segmentation.

Implement functionality to fetch and process images from the specified directories.

Phase 3:
Use Internet of things(IoT), analog microncontrollers to process the project to higher levels. Use it to store data, upload on cloud, and present in a formiddable way, by using substantial techniques.

## TF IDF Vector

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents. It is widely used in text mining and information retrieval. TF-IDF is composed of two main components: Term Frequency (TF) and Inverse Document Frequency (IDF). TF measures how frequently a term appears in a specific document, emphasizing terms that are more common within that document. IDF, on the other hand, assesses the importance of a term across the entire document corpus by reducing the weight of terms that appear in many documents and increasing the weight of terms that are rare. By combining these two metrics, TF-IDF helps to identify terms that are both significant in individual documents and unique across the corpus, making it an effective tool for various natural language processing tasks, such as keyword extraction, document similarity, and information retrieval.




## Cosine similarity


Cosine similarity is a metric used to measure the similarity between two vectors in a multi-dimensional space. It is widely used in text analysis and information retrieval to determine how similar two documents are, regardless of their size. Cosine similarity is calculated by taking the dot product of the vectors and dividing it by the product of their magnitudes. This results in a value between -1 and 1, where 1 indicates that the vectors are identical, 0 indicates no similarity, and -1 indicates that they are diametrically opposed. In the context of text analysis, cosine similarity can be used to compare documents represented as TF-IDF vectors, allowing for the identification of documents with similar content based on the angle between their vector representations, rather than their absolute differences in word counts. This makes it particularly useful for tasks such as document clustering, recommendation systems, and semantic search.

## VGC 16
The VGG16 model is a popular deep learning architecture known for its simplicity and effectiveness in image classification tasks. Introduced by the Visual Geometry Group (VGG) from the University of Oxford, VGG16 consists of 16 layers, including 13 convolutional layers and 3 fully connected layers, which make it deep yet manageable for modern computational resources. The architecture uses small 3x3 convolutional filters and focuses on increasing depth to capture more complex features while maintaining simplicity. Despite its relatively straightforward design, VGG16 has proven to be highly effective in tasks like object recognition and has been widely adopted in transfer learning applications. Its use of pre-trained weights allows it to generalize well across various domains, making it a go-to model for image-related deep learning tasks.
## resnet50 
ResNet-50 is a powerful deep learning model known for its innovative use of "residual learning" to address the vanishing gradient problem in very deep neural networks. Introduced by Microsoft Research, ResNet-50 consists of 50 layers, making it significantly deeper than traditional models like VGG16. What sets ResNet apart is its use of shortcut connections, or "skip connections," which allow the model to bypass certain layers and pass information directly to deeper layers. This design helps the network learn more complex features while avoiding issues that typically arise in deep networks, such as degradation of accuracy. ResNet-50 has become a widely used architecture for tasks like image classification, object detection, and feature extraction, known for its strong performance and ability to achieve high accuracy without the need for excessively large computational resources.







## mobilenet
MobileNet is a deep learning model designed specifically for mobile and embedded vision applications, where computational efficiency and low latency are crucial. Developed by Google, MobileNet is based on depthwise separable convolutions, which significantly reduce the number of parameters and computation compared to traditional convolutional neural networks (CNNs). This approach breaks down the standard convolution operation into two smaller tasks: a depthwise convolution, which filters each input channel separately, and a pointwise convolution, which combines these channels.

This lightweight architecture allows MobileNet to maintain a high level of accuracy while being faster and more efficient, making it ideal for real-time applications like object detection, face recognition, and mobile-based AI tasks. There are also variations like MobileNetV2 and MobileNetV3, which introduce further optimizations, making the model even more efficient for edge devices with limited resources.

## densenet
DenseNet (Dense Convolutional Network) is a deep learning architecture known for its dense connectivity pattern, which improves information flow between layers and mitigates the vanishing gradient problem. In DenseNet, each layer is directly connected to every other subsequent layer, meaning that the output of each layer is fed into all following layers as input. This approach allows DenseNet to reuse features, making the network more efficient and reducing the need for a large number of parameters compared to traditional deep networks like ResNet or VGG.

DenseNet excels at feature propagation and encourages feature reuse, leading to stronger gradients and more compact models. It also helps prevent overfitting while maintaining high accuracy, even with a relatively shallow number of parameters. DenseNet has proven effective in various computer vision tasks, including image classification, segmentation, and object detection, and is valued for its balance between performance and computational efficiency.

## Inception

The Inception model, also known as GoogLeNet, is a deep learning architecture designed to improve computational efficiency and accuracy in image classification tasks. Introduced by Google, Inception utilizes a novel module called the Inception module, which performs convolutions with multiple filter sizes (1x1, 3x3, and 5x5) in parallel within the same layer. This multi-scale feature extraction allows the model to capture information at different spatial scales, combining them for a richer representation of the input image.

Inception also includes 1x1 convolutions, which act as dimensionality reduction tools, reducing the number of feature maps and thereby lowering computational costs. As a result, the model can be deep without being computationally expensive, unlike traditional deep networks. The original GoogLeNet (Inception V1) has 22 layers, but later versions, such as Inception V3 and V4, introduced further optimizations and increased depth.

Inception models are widely used in tasks such as object detection, image classification, and transfer learning, offering a good balance between performance, accuracy, and computational efficiency. They are particularly valuable when training models on large datasets where resource constraints might be an issue.

## CNN
A Convolutional Neural Network (CNN) is a specialized type of deep learning architecture designed primarily for processing and analyzing visual data, such as images and videos. CNNs are particularly effective at capturing spatial hierarchies and patterns through convolution operations, which makes them highly suitable for tasks like image classification, object detection, and segmentation.

The core idea behind CNNs is the convolutional layer, where filters (or kernels) slide over the input image to detect specific features like edges, textures, or more complex structures. These filters allow CNNs to capture local patterns, and their learnable parameters enable the model to detect increasingly complex features as it goes deeper into the network.

CNNs typically consist of several key components:

Convolutional layers – Extract feature maps using filters.
Pooling layers – Downsample feature maps to reduce dimensions and computational complexity while retaining important information.
Activation functions (like ReLU) – Introduce non-linearity to allow the model to learn more complex patterns.
Fully connected layers – After feature extraction, these layers make predictions based on the learned features.
What makes CNNs stand out is their ability to learn hierarchical feature representations automatically. Early layers capture low-level features such as edges or textures, while deeper layers recognize more abstract concepts, like shapes or objects. CNNs are widely used in computer vision tasks, making them the backbone of applications like facial recognition, medical image analysis, and autonomous driving.
## Histogram Equalisation
Histogram Equalization is a technique in image processing used to improve the contrast of an image. It works by redistributing the intensities of the pixels so that the histogram of the output image is approximately uniform, meaning all intensity levels are used more evenly across the image.

Here’s how Histogram Equalization works:

Histogram Calculation: First, a histogram of the image is calculated, which shows the frequency of each pixel intensity level.
Cumulative Distribution Function (CDF): The CDF is computed from the histogram. This function determines the mapping of old intensity levels to new ones.
Intensity Mapping: Each pixel's intensity in the input image is mapped to a new intensity value based on the CDF, spreading the pixel values more evenly across the entire range.
The result is an image with enhanced contrast, especially useful in cases where the original image has poor lighting or is too dark or too bright. For example, in medical images or satellite images, Histogram Equalization helps to make hidden details more visible by stretching out the pixel intensity range. However, it can also amplify noise if present in the image.
## logistic regresion
Logistic Regression is a widely used statistical method for binary classification tasks, where the objective is to predict one of two possible outcomes based on input features. Despite its name, it is not a regression algorithm but rather a classification technique that applies the logistic function, or sigmoid function, to map predicted values to probabilities ranging from 0 to 1. The model operates by first forming a linear combination of the input features, expressed as the sigmoid; transforming the linear output into a probabilistic value that indicates the likelihood of the input belonging to a particular class. For instance, if the probability exceeds a threshold (commonly set at 0.5), the model classifies the input as belonging to class 1; otherwise, it is classified as class 0. Logistic Regression employs a loss function known as binary cross-entropy, which quantifies the difference between the predicted probabilities and the actual binary labels, guiding the optimization of model parameters via techniques such as gradient descent. One of the key advantages of logistic regression is its interpretability, as the coefficients provide insights into the influence of each feature on the likelihood of the outcome. Although it assumes a linear relationship between the features and the log-odds of the outcome, logistic regression is often effective in practice, especially when the classes are linearly separable. Its applications span various fields, including medicine for diagnosing diseases, finance for assessing credit risk, and marketing for predicting customer behavior, making it a fundamental tool in the data scientist's arsenal.

## random forest
Random Forest is an ensemble learning method primarily used for classification and regression tasks, which combines the predictions of multiple decision trees to improve accuracy and control overfitting. The core idea behind Random Forest is to build a collection of decision trees during training time and output the mode of their predictions (for classification) or the mean prediction (for regression) as the final result. Each tree in the forest is trained on a random subset of the training data, which is created through a process called bootstrapping, where samples are drawn with replacement. Additionally, during the construction of each tree, a random subset of features is selected for consideration at each split, introducing further randomness and ensuring that the individual trees are diverse. This diversity among the trees helps to reduce the model's variance, making Random Forest robust against overfitting, especially when dealing with high-dimensional data. The aggregation of predictions from multiple trees leads to a more stable and accurate model compared to a single decision tree. Random Forest is also highly interpretable, as feature importance scores can be derived to understand which features contribute most to the predictions. Its versatility allows it to handle both numerical and categorical data, making it applicable across various domains, including finance for credit scoring, healthcare for disease prediction, and ecology for species classification. Overall, Random Forest is a powerful and flexible tool in the machine learning toolkit, known for its strong performance across a wide range of applications.








## Support vector machine
Support Vector Machine (SVM) is a supervised machine learning algorithm primarily used for classification tasks, though it can also be applied to regression. The main idea behind SVM is to find the optimal hyperplane that best separates data points from different classes in a high-dimensional space. This hyperplane maximizes the margin, which is the distance between the closest points of the data classes, known as support vectors. By maximizing this margin, SVM aims to create a robust decision boundary that generalizes well to unseen data.

In cases where data is not linearly separable, SVM employs a technique called the kernel trick, which transforms the original feature space into a higher-dimensional space, making it possible to find a linear hyperplane. Common kernel functions include the linear, polynomial, and radial basis function (RBF) kernels, each providing a different way to map the input data. This flexibility allows SVM to effectively handle complex datasets with intricate patterns.

SVM is particularly powerful in high-dimensional spaces and is less prone to overfitting, especially when the number of features exceeds the number of samples. However, the choice of kernel and the regularization parameter can significantly impact the model's performance, requiring careful tuning. SVM has found applications across various domains, including text classification, image recognition, and bioinformatics, owing to its strong theoretical foundation and effectiveness in practice. Its ability to provide clear margins of separation and interpretability regarding support vectors makes SVM a popular choice among machine learning practitioners.

## Efficient Net
EfficientNet is a family of convolutional neural network architectures designed to optimize the trade-off between model accuracy and efficiency in image classification tasks. Developed by researchers at Google, EfficientNet introduces a systematic approach to scaling up networks, which focuses on three key dimensions: depth, width, and resolution. Unlike traditional scaling methods that independently adjust these dimensions, EfficientNet employs a compound scaling method that uniformly scales all three dimensions based on a fixed set of parameters, resulting in a more balanced and effective architecture.

The core architecture of EfficientNet is based on the use of Mobile Inverted Bottleneck Convolution (MBConv) blocks, which effectively reduce computational cost while maintaining performance. These blocks utilize depthwise separable convolutions, allowing for fewer parameters and less computation compared to standard convolutions. Furthermore, EfficientNet incorporates techniques such as squeeze-and-excitation optimization, which enhances the network's ability to model channel relationships and improve feature extraction.

EfficientNet achieves state-of-the-art performance on various benchmark datasets, including ImageNet, while being significantly smaller and faster than previous models like ResNet and DenseNet. For instance, EfficientNet-B7, the largest variant in the family, achieves remarkable accuracy with far fewer parameters than its predecessors. This efficiency makes EfficientNet particularly well-suited for applications in resource-constrained environments, such as mobile devices and edge computing, where computational power and memory are limited. Overall, EfficientNet represents a significant advancement in neural network design, combining high accuracy with computational efficiency, making it a preferred choice in modern computer vision tasks.

## nasnet
NASNet, or Neural Architecture Search Network, represents a significant advancement in deep learning for image classification, utilizing an automated approach to design its architecture through a process known as Neural Architecture Search (NAS). This method employs reinforcement learning to discover optimal network architectures, allowing it to outperform manually designed models. One of NASNet's key advantages is its scalability; it can be tailored to different computational resources and varying dataset sizes, making it versatile for various applications. The architecture consists of intricate building blocks, including normal and reduction cells, which enable it to capture complex patterns in data. With state-of-the-art performance on benchmarks like ImageNet, NASNet has set new standards in image classification tasks. Additionally, it supports transfer learning, allowing users to leverage pre-trained models for other tasks, significantly reducing training time and improving accuracy on new datasets. Implementing NASNet is straightforward with frameworks like TensorFlow and Keras, where users can load pre-trained models, freeze layers to prevent them from updating during training, and add custom layers for specific tasks. This adaptability and performance have made NASNet a preferred choice among researchers and practitioners in the field of computer vision.


## Local Binary Pattern
Local Binary Pattern (LBP) is a highly effective texture descriptor widely utilized in computer vision and image processing, particularly for tasks such as texture classification and face recognition. The method operates by examining each pixel in a grayscale image alongside its neighboring pixels within a defined radius, typically using a 3x3 grid for the immediate eight neighbors surrounding the central pixel. By comparing the intensity of the neighboring pixels to that of the central pixel, LBP generates a binary code: a neighbor is assigned a '1' if its value is greater than or equal to the center pixel's value and '0' otherwise. This results in a unique binary representation for each pixel that can be converted into a decimal value, effectively capturing the local texture information. A notable feature of LBP is its rotation invariance, achieved by considering the minimum binary value of the circular pattern, allowing it to remain robust against variations in orientation. After processing the entire image, a histogram of these LBP values is created, providing a compact and descriptive feature vector that summarizes the distribution of texture patterns in the image. LBP's applications are diverse, ranging from face recognition, where it excels in capturing the nuances of facial textures while being resilient to changes in lighting and expressions, to texture classification in materials and industrial inspection. Additionally, LBP plays a role in content-based image retrieval systems, helping to index and retrieve images based on texture features. Its computational efficiency and simple implementation make LBP a preferred choice in real-time applications, while its robustness to illumination changes adds to its practicality in real-world scenarios. Overall, Local Binary Pattern stands out as a powerful and efficient tool for extracting meaningful texture features from images, contributing significantly to advancements in various domains, including computer vision, medical imaging, and remote sensing.







## Libraries and Usage

```
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage import feature, exposure
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects, skeletonize
from skimage.feature import local_binary_pattern, hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
# from imblearn.over_sampling import SMOTE
from scipy.stats import skew, kurtosis
from joblib import dump
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np 
import pandas as pd 


import os



import pathlib
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils import to_categorical, image_dataset_from_directory
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D, Activation
from keras.optimizers import RMSprop
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from tensorflow.keras import applications, losses
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau


from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNetV2, DenseNet121


from itertools import chain

import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")






## Run Locally

Clone the project

```bash
  git clone https://github.com/Prayag-Chawla/Anveshan-Hackathon/tree/Main
```

Go to the project directory

```bash
  cd https://github.com/Prayag-Chawla/Anveshan-Hackathon/tree/Main
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```


## Used By
In the real world, this project is used a lot of Medical Research based companies, and utilised & acquired by hospitals. Collection fo datastes, and then implication of neural network tecnhiques is an important aspect of this ultimate project.


## Appendix

A very crucial project in the realm of image processing, data science and new age predictions domain using visualization techniques as well as deep learning modelling.






## Feedback

If you have any feedback, please reach out to us at chawlapc.619@gmail.com

