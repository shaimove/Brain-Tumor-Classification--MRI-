# Brain Tumor Classification (MRI)

## Introduction
The following project is very simple in image classification, for MRI brain scan. 
The objective is to classify the scan, regardless of their point of view (Sagittal, Coronal, Axial) or MRI type scan (T1, T2, FLAIR). 
The dataset can be download from the following Kaggle repo:
https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri

We have 4 classes: No Tumer, Glioma Tumer, Meningioma Tumor, and Pituitary Tumor. [0,1,2,3]
For this purpose, I designed a few CNN's, from basic VGG style (medium performance) to ResNet50 and ResNet72. I used PyTorch as a library for CNN design, and since we are dealing with grayscale imaging, I didn't find any pre-trained CNN for this task. 
I had a few thousand examples, and the dataset was pretty balanced. 
I used CrossEntropy loss function, with Adam optimizer, with different learning rates (0.0001,0.001.0.1,1). The evaluation was based on the confusion matrix of the results. 

## Dataset.py

For this project, I design MRIDataset, a subclass of Dataset PyTorch class. In this dataset, and initialization created a list of all file locations, and labels (one-hot-encoding). After I send the dataset to the data loader, the image was read using OpenCV, with a batch size of 256 (or lower, depends on image resolution). We can read the images in 256*256*1 resolution of 512*512*1 resolution. 

## Log.py

Also, I design a ClassificationLog class, collecting the results during training, validation, and testing (split of ~70-15-15) collect the labels in each batch and each epoch. every Batch I collected the output of the CNN, the output of the last linear layer (to estimate the weights and keep them relatively small), and the true labels. In every epoch, I collected the loss over of examples. In addition, I added basic Confusion Martix plotting method. 

## utils.py 

The model was designed in a separate script, containing Convolution block and Identity block classes (depends on architecture), and the main class of the network. 
I also added a utils script, with useful functions: Calculate the mean pixel value and standard deviation of a pixel value of the whole training set. These values are then used to normalize the images before training - CreateValidationSet. The second function created a validation set from the test set (the dataset originally include only training and validation sets) - CreateValidationSet, and a simple function to calculate the number of trainable parameters since PyTorch don't have this kind of function (Keras has this king of function) - count_parameters.

## Results
The results of the training are very poor, here for example a result for VGG style of architecture after 20 epochs:

![Image 1](https://github.com/shaimove/Brain-Tumor-Classification--MRI-/blob/main/Results/Version%200.2%20after%20100%20epochs.png)


When you visit the Kaggle repo of that dataset, everyone tried only to classify between Tumer and Non-Tumer, which might be an easier task. 

## License

Free for any use.

## Contact information
Developed by Sharon Haimov, Research Engineer at Lumenis.

Email: sharon.haimov@lumenis.com or shaimove@gmail.com




