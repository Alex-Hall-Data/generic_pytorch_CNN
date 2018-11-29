# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:03:38 2018

class to create a simple cnn with 3 convolutional layers 
will handle images of 120x120x3

@author: ahall
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class SimpleCNN(torch.nn.Module ):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self, output_classes):
        super(SimpleCNN, self).__init__()
        self.output_classes = output_classes
        
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(9, 27, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(27, 81, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #batch normalisation layers
        self.conv1_bn = nn.BatchNorm2d(9)
        self.conv2_bn = nn.BatchNorm2d(27)
        self.conv3_bn = nn.BatchNorm2d(81)
        
        #dropout layers
        self.conv_dropout = nn.Dropout2d(p=0.5)
        self.dropout = nn.Dropout(p=0.5)
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(81 * 15 * 15, 1024)
        self.fc2 = torch.nn.Linear(1024,64)
        self.fc3 = torch.nn.Linear(64, output_classes)
        
        #randomise initial weights
        torch.nn.init.xavier_uniform(self.conv1.weight)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)
        
        #nn.init.xavier_uniform(self.conv1.bias)
        #nn.init.xavier_uniform(self.conv2.bias)
        #nn.init.xavier_uniform(self.conv3.bias)
        #nn.init.xavier_uniform(self.fc1.bias)
        #nn.init.xavier_uniform(self.fc2.bias)
        #nn.init.xavier_uniform(self.fc3.bias)
        
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 120, 120) to (6, 120, 120)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x=self.conv_dropout(x)
        
        #Size changes from (6, 120, 120) to (6, 60, 60)
        x = self.pool(x)
        
        #second convolutional layer - (6,60,60) to (12 , 60 , 60)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x=self.conv_dropout(x)
        
        #pooling layer = (12,60,60) to (12, 30 ,30)
        x = self.pool(x)
        
        #third convolution - (12, 30 ,30) to (24 , 30 , 30)
        x = F.relu(self.conv3(x))
        #x=self.conv_dropout(x)
        
        #pooling layer  - (24 , 30 ,30) to (24,15,15)
        x = self.pool(x)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 81 * 15 *15)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))
        x=self.dropout(x)
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        #x=self.dropout(x)
        
        x= F.relu(self.fc3(x))
        return(x)