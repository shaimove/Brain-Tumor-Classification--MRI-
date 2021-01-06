# #Main.py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision import transforms

import model
import utils
from Dataset import DatasetMRI
from Log import ClassificationLog


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Dataset
# The dataset with all views (axial, segittal,coronal), but different sizes.
# the dataaset is balanced in training set and testing set
# Define folders 
folder_training = '../Training/'
folder_validation = '../Validation'
folder_testing = '../Testing/'

# define classes (sub folders)
classes = ['no_tumor','glioma_tumor','meningioma_tumor','pituitary_tumor']
clasees_one_hot = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

#%% Run only onces! Prepare the dateset 
# create validation set from training set
#utils.CreateValidationSet(folder_training,folder_validation)

# compute the mean and std pixel value to normalize the dataset
#mean_pixel,std_pixel = utils.CalculateStats(folder_training)
# mean pixel value: 44.17, mean std value: 43.51
mean = 44.17 ; std = 43.51;

#%% Create dataset and data loader
# Define transformation and data augmentation
transform = transforms.Compose([transforms.ToTensor(),
    transforms.Resize((512,512)),
    transforms.Normalize(mean=[mean],std=[std])])

# define batch size
batch_size_train = 64
batch_size_validation = 32

# define dataset and dataloader for training
train_dataset = DatasetMRI(folder_training,classes,'Training',transform=transform)
train_loader = data.DataLoader(train_dataset,batch_size=batch_size_train,shuffle=True)

# define dataset and dataloader for validation
validation_dataset = DatasetMRI(folder_validation,classes,'Validation',transform=transform)
validation_loader = data.DataLoader(train_dataset,batch_size=batch_size_validation,shuffle=True)


#%% Define parameters
# number of epochs
num_epochs = 20

# load model
model = model.MRIModel().to(device)
utils.count_parameters(model)

# send parameters to optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# define loss function 
criterion = torch.nn.CrossEntropyLoss()

# initiate logs
trainLog = ClassificationLog()
validationLog = ClassificationLog()

#%% Training


for epoch in range(num_epochs):
    ##################
    ### TRAIN LOOP ###
    ##################
    # set the model to train mode
    model.train()
    
    # initiate training loss
    train_loss = 0
    
    for batch in train_loader:
        # get batch images and labels
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # clear the old gradients from optimizer
        optimizer.zero_grad()
        
        # forward pass: feed inputs to the model to get outputs
        linear_output,output = model(inputs)
        
        # calculate the training batch loss
        loss = criterion(output, torch.max(labels, 1)[1])
        
        # backward: perform gradient descent of the loss w.r. to the model params
        loss.backward()
        
        # update the model parameters by performing a single optimization step
        optimizer.step()
        
        # accumulate the training loss
        train_loss += loss.item()
        
        # update training log
        trainLog.BatchUpdate(epoch,output,linear_output,labels)

            
    #######################
    ### VALIDATION LOOP ###
    #######################
    # set the model to eval mode
    model.eval()
    
    # initiate validation loss
    valid_loss = 0
    
    # turn off gradients for validation
    with torch.no_grad():
        for batch in validation_loader:
            # get batch images and labels
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # forward pass
            linear_output,output = model(inputs)
            
            # validation batch loss
            loss = criterion(output, torch.max(labels, 1)[1]) 
            
            # accumulate the valid_loss
            valid_loss += loss.item()
            
            # update validation log
            validationLog.BatchUpdate(epoch,output,linear_output,labels)
                
    #########################
    ## PRINT EPOCH RESULTS ##
    #########################
    train_loss /= len(train_loader)
    valid_loss /= len(validation_loader)
    # update training and validation loss
    trainLog.EpochUpdate(epoch,train_loss)
    validationLog.EpochUpdate(epoch,valid_loss)
    # print results
    print('Epoch: %s/%s: Training loss: %.3f. Validation Loss: %.3f.'
          % (epoch+1,num_epochs,train_loss,valid_loss))
 
    
    
validationLog.PlotConfusionMatrix(classes)





#%% Testing Loop
# define dataset and dataloader for testing
batch_size_test = 128
test_dataset = DatasetMRI(folder_testing,classes,'Testing',transform=transform)
test_loader = data.DataLoader(train_dataset,batch_size=batch_size_test,shuffle=True)

# initiate test log
testLog = ClassificationLog()

# initiate validation loss
test_loss = 0

# set the model to eval mode
model.eval()

with torch.no_grad():
    for bacth in test_loader:        
        # get batch images and labels
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # forward pass
        linear_output,output = model(inputs)
        
        # validation batch loss
        loss = criterion(output, torch.max(labels, 1)[1])

        # accumulate the valid_loss
        test_loss += loss.item()
        
        # update log
        testLog.BatchUpdate(0,output,linear_output,labels)
    
    test_loss /= len(test_loader)
    testLog.EpochUpdate(0,test_loss)
    print('Test Loss: %.3f.' % test_loss)
        
testLog.PlotConfusionMatrix(classes)
    
        
    