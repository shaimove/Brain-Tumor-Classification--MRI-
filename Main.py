# #Main.py
import numpy as np
import matplotlib.pyplot as plt
import torch
import model
from Dataset import DatasetMRI
from torch.utils import data
from torchvision import transforms

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
# mean pixel value: 149.58, mean std value: 110.05
mean = 149.58 ; std = 110.05;

#%% Create dataset and data loader
# Define transformation and data augmentation
transform = transforms.Compose([transforms.ToTensor(),
    transforms.Resize((256,256)),
    transforms.Normalize(mean=[mean],std=[std])])

# define batch size
batch_size = 256

# define dataset and dataloader for training
train_dataset = DatasetMRI(folder_training,classes,'Training',transform=transform)
train_loader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

# define dataset and dataloader for validation
validation_dataset = DatasetMRI(folder_validation,classes,'Validation',transform=transform)
validation_loader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)


#%% Define parameters
# number of epochs
num_epochs = 10

# load model
model = model.MRIModel().to(device)

# send parameters to optimizer
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# define loss function 
criterion = torch.nn.CrossEntropyLoss()


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
        
        # clear the old gradients from optimized variables
        optimizer.zero_grad()
        
        # forward pass: feed inputs to the model to get outputs
        output = model(inputs)
        
        # calculate the training batch loss
        loss = criterion(output, torch.max(labels, 1)[1])
        
        # backward: perform gradient descent of the loss w.r. to the model params
        loss.backward()
        
        # update the model parameters by performing a single optimization step
        optimizer.step()
        
        # accumulate the training loss
        train_loss += loss.item()

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
            output = model(inputs)
            
            # validation batch loss
            loss = criterion(output, torch.max(labels, 1)[1]) 
            
            # accumulate the valid_loss
            valid_loss += loss.item()
            
    #########################
    ## PRINT EPOCH RESULTS ##
    #########################
    train_loss /= len(train_loader)
    valid_loss /= len(validation_loader)
    print(f'Epoch: {epoch+1}/{num_epochs}: Training loss: {train_loss}. Validation Loss: {valid_loss}.')
 
    
    
    




#%% Testing
# define dataset and dataloader for testing
train_dataset = DatasetMRI(folder_testing,classes,'Testing',transform=transform)
train_loader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)


#%%

