# model.py
import torch
from torch import nn as nn

#%% ResNet50

class MRIModel(nn.Module):
    def __init__(self, in_channels=1, first_conv_size=16,f=3,s=2):
        super().__init__()
        
        # Preprocess stage:
        # from (N*1*512*512) to (N*16*256*256)
        self.conv1 = nn.Conv2d(in_channels,first_conv_size,kernel_size=7,padding=3)
        self.bn1 = nn.BatchNorm2d(first_conv_size)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2,2)
        
        # Stage 1:
        # from (N*16*256*256) to (N*32*128*128)
        self.ConvBlock1 = ConvBlock(first_conv_size,[8,8,32],f,s)
        self.Identity_1_1 = IdentityBlock([8,8,32],f)
        self.Identity_1_2 = IdentityBlock([8,8,32],f)
        
        # Stage 2:
        # from (N*32*128*128) to (N*64*64*64)    
        self.ConvBlock2 = ConvBlock(2*first_conv_size,[16,16,64],f,s)
        self.Identity_2_1 = IdentityBlock([16,16,64],f)
        self.Identity_2_2 = IdentityBlock([16,16,64],f)
        self.Identity_2_3 = IdentityBlock([16,16,64],f)
            
        # Stage 3:
        # from (N*64*64*64) to (N*128*32*32) 
        self.ConvBlock3 = ConvBlock(4*first_conv_size,[32,32,128],f,s)
        self.Identity_3_1 = IdentityBlock([32,32,128],f)
        self.Identity_3_2 = IdentityBlock([32,32,128],f)
        self.Identity_3_3 = IdentityBlock([32,32,128],f)
        self.Identity_3_4 = IdentityBlock([32,32,128],f)
       
        # Stage 4:
        # from (N*128*32*32)  to (N*256*16*16)
        self.ConvBlock4 = ConvBlock(8*first_conv_size,[64,64,256],f,s)
        self.Identity_4_1 = IdentityBlock([64,64,256],f)
        self.Identity_4_2 = IdentityBlock([64,64,256],f)
        self.Identity_4_3 = IdentityBlock([64,64,256],f)
        self.Identity_4_4 = IdentityBlock([64,64,256],f)
        self.Identity_4_5 = IdentityBlock([64,64,256],f)
        
        # Stage 5:
        # from (N*256*16*16) to (N*512*8*8)
        self.ConvBlock5 = ConvBlock(16*first_conv_size,[128,128,512],f,s)
        self.Identity_5_1 = IdentityBlock([128,128,512],f)
        self.Identity_5_2 = IdentityBlock([128,128,512],f)
        self.Identity_5_3 = IdentityBlock([128,128,512],f)
        self.Identity_5_4 = IdentityBlock([128,128,512],f)
        
        # Stage 6:
        # from (N*512*8*8) to (N*1024*4*4)
        self.ConvBlock6 = ConvBlock(32*first_conv_size,[256,256,1024],f,s)
        self.Identity_6_1 = IdentityBlock([256,256,1024],f)
        self.Identity_6_2 = IdentityBlock([256,256,1024],f)
        self.Identity_6_3 = IdentityBlock([256,256,1024],f)
        
        # Stage 7:
        # from (N*1024*4*4) to (N*2048*2*2)
        self.ConvBlock7 = ConvBlock(64*first_conv_size,[512,512,2048],f,s)
        self.Identity_7_1 = IdentityBlock([512,512,2048],f)
        self.Identity_7_2 = IdentityBlock([512,512,2048],f)
        
        
        # Linear stage
        # from (N*2048*2*2) to (N*2048*1*1)
        self.avrg_pool = nn.AvgPool2d(2,2)
        
        # from (N*4096) to (N*64)
        self.linear1 = nn.Linear(2048, 64)
        
        # TanH layer after first linear
        self.tanh = nn.Tanh()
        
        # from (N*64) to (N*4)
        self.linear2 = nn.Linear(64, 4)
        
        # softmax layer
        self.softmax = nn.Softmax(dim=1)
        
        # initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # initiate with Xavier initialization
        for m in self.modules():
            if type(m) in {nn.Conv2d,nn.Linear}:
                # Weight of layers
                nn.init.xavier_normal_(m.weight)
                # if we have bias
                if m.bias is not None:
                    m.bias.data.fill_(0.01)  
                    
            if type(m) in {nn.BatchNorm2d}:
                # Weight of layers
                nn.init.normal_(m.weight)
                # if we have bias
                if m.bias is not None:
                    m.bias.data.fill_(0.01) 
    
    def forward(self, X):
        
        # Preprocess stage:
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.maxpool(X)
        
        # Stage 1:
        X = self.ConvBlock1(X)
        X = self.Identity_1_1(X)
        X = self.Identity_1_2(X)
        
        # Stage 2:
        X = self.ConvBlock2(X)  
        X = self.Identity_2_1(X)
        X = self.Identity_2_2(X)
        X = self.Identity_2_3(X)
        
        # Stage 3: 
        X = self.ConvBlock3(X)   
        X = self.Identity_3_1(X)
        X = self.Identity_3_2(X)
        X = self.Identity_3_3(X)
        X = self.Identity_3_4(X)
        
        # Stage 4:
        X = self.ConvBlock4(X)     
        X = self.Identity_4_1(X)
        X = self.Identity_4_2(X)
        X = self.Identity_4_3(X)
        X = self.Identity_4_4(X)
        X = self.Identity_4_5(X)
        
        # Stage 5:
        X = self.ConvBlock5(X)     
        X = self.Identity_5_1(X)
        X = self.Identity_5_2(X)
        X = self.Identity_5_3(X)
        X = self.Identity_5_4(X)
        
        # Stage 6:
        X = self.ConvBlock6(X)     
        X = self.Identity_6_1(X)
        X = self.Identity_6_2(X)
        X = self.Identity_6_3(X)
        
        # Stage 7:
        X = self.ConvBlock7(X)     
        X = self.Identity_7_1(X)
        X = self.Identity_7_2(X)
        
        # Linear stage:
        X = self.avrg_pool(X) # average pooling
        X = X.view(X.size(0),-1,) # flatten to 1D vector
        X = self.linear1(X)
        X = self.tanh(X)
        linear = self.linear2(X)
        output = self.softmax(X)
        
        return linear,output
        


#%% Identity Block

class IdentityBlock(nn.Module):
    # from (N*F3*H*W) to (N*F3,H,W)
    def __init__(self,filters,f):
        super().__init__()
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Define First Main Path
        self.conv1 = nn.Conv2d(F3,F1,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(F1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Define second Main Path
        self.conv2 = nn.Conv2d(F1,F2,kernel_size=f,padding=f//2)
        self.bn2 = nn.BatchNorm2d(F2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Define Third Main Path
        self.conv3 = nn.Conv2d(F2,F3,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(F3)
        self.relu3 = nn.ReLU(inplace=True)
        
    def forward(self,X):
        # define shortcut for future
        X_shortcut = X
        
        # First Main Path
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu1(X)
        
        # Second Main Path
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu2(X)
        
        # Third Main Path
        X = self.conv3(X)
        X = self.bn3(X)
        
        # Add and Activation
        X = torch.add(X, X_shortcut)
        X = self.relu3(X)
        
        return X

#%% Convolutional  Block


class ConvBlock(nn.Module):
    # from (N*input_size*H*W) to (N*F3,H/2,W/2)
    def __init__(self,input_size,filters,f,s):
        super().__init__()
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Define First Main Path
        self.conv1 = nn.Conv2d(input_size,F1,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(F1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Define second Main Path
        self.conv2 = nn.Conv2d(F1,F2,kernel_size=f,padding=f//2,stride=s)
        self.bn2 = nn.BatchNorm2d(F2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Define Third Main Path
        self.conv3 = nn.Conv2d(F2,F3,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(F3)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Define Parrallel Path
        self.conv4 = nn.Conv2d(input_size,F3,kernel_size=1,stride=s)
        self.bn4 = nn.BatchNorm2d(F3)
    
    
    def forward(self,X):
        # define shortcut for future
        X_shortcut = X
        
        # First Main Path
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu1(X)
        
        # Second Main Path
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu2(X)
        
        # Third Main Path
        X = self.conv3(X)
        X = self.bn3(X)
        
        # Parrallel Path
        X_shortcut = self.conv4(X_shortcut)
        X_shortcut = self.bn4(X_shortcut)
        
        # Add and Activation
        X = torch.add(X, X_shortcut)
        X = self.relu3(X)
        
        return X
        
        
        
        
        
        
        
        
        
        
        
        
        