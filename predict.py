
import matplotlib.pyplot as plt
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

import copy
import argparse

from PIL import Image

def main():
    parser=argparse.ArgumentParser(description='Process image and Predict')
    parser.add_argument('--gpu',type=bool,default=False,help='whether to use GPU')
    parser.add_argument('--image_path',type=str,help='path of image to be predicted')
    parser.add_argument('--cat_to_name',type=str,default='cat_to_name.json',help='path to category to flower name mapping json')
    parser.add_argument('--saved_model_path',type=str,default='flower98_checkpoint.pth',help='path of your saved mode')
    parser.add_argument('--topk',type=int,default=5,help='display top k probabilities')

    args = parser.parse_args()
def process_image(im):
    
   
    im = Image.open(im)
    width, height = im.size
    
    if width > height:
        ratio =float(width)/float(height)
        im.thumbnail((ratio*256,256))
        
    elif height>width:
        ratio =float(width)/float(height)
        im.thumbnail((256,ratio*256))
        
    new_width, new_height = im.size  #size of resized image
    
    left = (new_width-224)/2
    top = (new_height-224)/2
    right = (new_width +224)/2
    bottom = (new_height+224)/2
    im=im.crop((left, top, right, bottom))
    
    np_image=np.array(im)
    np_image=np_image/255
    
    means=np.array([0.485,0.456,0.406])
    std=np.array([0.229,0.224,0.225])
    
    np_image=(np_image-means)/std
    np_image=np_image.transpose((2,0,1))
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.array(image).transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    if title:
        ax.set_title(title)
    
    return ax
                        
    ### Predict the class (or classes) of an image using a trained deep learning model.
   
    
    # TODO: Implement the code to predict the class from an image file

    
def predict(image_path, model, topk=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.eval()
    
    image=process_image(image_path)
    image=torch.from_numpy(np.array([image])).float()
    
    image=Variable(image)
    image=image.to(device)
    output=model.forward(image)
     
    probability = torch.exp(output).data
    probs = torch.topk(probability,topk)[0].tolist()[0]
    indices = torch.topk(probability,topk)[1].tolist()[0]
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])
        
    label = []
    for i in range(5):
        label.append(ind[indices[i]])

    return probs, label
       
img = random.choice(os.listdir('flowers/test/98/'))
img_path= 'flowers/test/98/'+ img
with Image.open(img_path) as image:
        plt.imshow(image)
        
probs, classes = predict(img_path,model)
print(probs)
print(classes)
print([cat_to_name[x] for x in classes])
     