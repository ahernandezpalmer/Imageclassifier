
import time
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from tqdm import tqdm_notebook as tqdm
import time
import os, random

import argparse

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=bool,default=False,help='whether to use gpu')
    parser.add_argument('--data_dir',help='directory for data:',default='/flowers')
    parser.add_argument('--arch',type=str,default=models.vgg16(pretrained=True),help='Architecture type vgg16 or vgg19')
    parser.add_argument('--save_dir',dest='save_dir',type=str,default='./Checkpoint.pth',help='Folder when model is saved, default is current:')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='Learning rate:')
    parser.add_argument('--input_size',type=int,action='store',default=25088,dest='input_size',help='Input size')
    parser.add_argument('--hidden_layer1',type=int,action='store',default=1024,dest='hidden_layer1',help='Number of hidden units layer 1:')
    parser.add_argument('--hidden_layer2',type=int,action='store',default=512,dest='hidden_layer2',help='Number of hidden unitslayer 2:')
    parser.add_argument('--output_size',type=int,action='store',default=102,dest='output_size',help='Output size')
    parser.add_argument('--epochs',type=int,help='Number of epochs:',default=3)
    
    args = parser.parse_args()
    return args
args=main()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(15),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(250),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms= transforms.Compose([transforms.Resize(250),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = args.arch

# turn off gradients for the model

for param in model.parameters():
    param.requires_grad = False
    
 # define our new classifier
from collections import OrderedDict
classifier = nn.Sequential(nn.Linear(args.input_size, args.hidden_layer1),           
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(args.hidden_layer1,args.hidden_layer2),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(args.hidden_layer2 ,args.output_size),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier

# define the criterion and optimizer and move the model to any device avaiable 

criterion = nn.NLLLoss()      # define the loss
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)    # optimizer

model.to(device);
#validation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 42
model.to(device)
for epoch in range(epochs):
    for images, labels in trainloader:
        steps += 1
        # Move images and label tensors to the default device
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            valid_loss = 0
            model.eval()
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    valid_loss += batch_loss.item()
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                     
                   
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"Valid accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
            
            model.class_to_idx = train_data.class_to_idx
            
            checkpoint = {'input_size': 25088,
              'output_size': 102,
              'hidden_layer1': 1024,
              'hidden_layer2':512,
              'learning rate':0.001,
              'classifier':classifier,
              'epochs':4,
              'optimizer':optimizer.state_dict,
              'class_to_idx':model.class_to_idx,
              'state_dict': model.state_dict()}

optimizer.state_dict
torch.save(checkpoint, args.save_dir)
