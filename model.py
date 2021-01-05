# model.py

import torch
from torch import nn as nn

class MRIModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=4):
        super().__init__()
        
        # from (N*1*256*256) to (N*4*128*128)
        self.block1 = ConvBlock(in_channels, conv_channels)
        
        # from (N*4*128*128) to (N*8*64*64)
        self.block2 = ConvBlock(conv_channels, conv_channels * 2)
        
        # from (N*8*64*64) to (N*16*32*32)
        self.block3 = ConvBlock(conv_channels * 2, conv_channels * 4)
        
        # from (N*16*32*32) to (N*32*16*16)
        self.block4 = ConvBlock(conv_channels * 4, conv_channels * 8)
        
        # from (N*32*16*16) to (N*64*8*8)
        self.block5 = ConvBlock(conv_channels * 8, conv_channels * 16)
        
        # Linear to 64-output
        self.linear1 = nn.Linear(64*8*8, 64)
        
        # TanH layer after first linear
        self.tanh = nn.Tanh()
        
        # Linear to 4-output
        self.linear2 = nn.Linear(64, 4)
        
        # softmax layer
        self.softmax = nn.Softmax(dim=1)
        
        # initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # initiate with Xavier initialization
        for m in self.modules():
            if type(m) in {nn.Linear,nn.Conv2d,nn.BatchNorm2d}:
                # Weight of layers
                nn.init.normal_(m.weight)
                # if we have bias
                if m.bias is not None:
                    m.bias.data.fill_(0.01)  
    
    
    def forward(self, input_batch):
        
        # run throw all blocks
        block_out = self.block1(input_batch)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        block_out = self.block5(block_out)
        
        # flatten to 1D vector
        conv_flat = block_out.view(block_out.size(0),-1,)
            
        # linear and softmax
        linear_output = self.linear1(conv_flat)
        linear_output = self.tanh(linear_output)
        linear_output = self.linear2(linear_output)
        output = self.softmax(linear_output)
        
        return linear_output,output
        
        

class ConvBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        
        # two blocks of Cond2D->ReLU->BatchNorm2D ans than MaxPool2D
        self.conv1 = nn.Conv2d(in_channels, conv_channels, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(conv_channels)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(conv_channels)
        
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.bn1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)
        block_out = self.bn2(block_out)

        return self.maxpool(block_out)
