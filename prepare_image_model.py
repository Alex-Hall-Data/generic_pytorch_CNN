#this script takes a labelled image dataset (image labels in json format) and arranges it into test, valid and train folders with each class having its own folder

import os
import shutil
import json
import random
import numpy as np
from matplotlib import pyplot as plt

from PIL import Image, ImageOps
import torch
from torchvision import transforms, datasets

HIERARCHY_DEPTH = 2 #3
VALIDATE_PROP = 0.2
TEST_PROP=0.2
TRAIN_PROP=0.6

def prepare_imagefolder():
    """Images should be categorized into subdirectories corresponding to labels.
    Finding out how narrowly we can classify the taxonomy will be trial and error
    We make a copy of them, sorted into directories for use with ImageLoader.
    The directory names serve as image labels.

    HIERARCHY_DEPTH indicates how narrow we want our taxonomic classification to be.
    If it's a larger value there will be more directories and more labels.
    (Default should be 2 really but our hierarchy data includes the mikrotax 'module')
    TODO remove module from the classification list if we ever re-scrape the data.
    """

    for filename in os.listdir('./data'):
       #TODO - replace trycatch with something less general
        
        with open(os.path.join(os.getcwd(), 'data', filename)) as json_data:
            data = json.load(json_data)
            

            hierarchy = data['hierarchy']
            if len(hierarchy) < HIERARCHY_DEPTH:
                # images should be duplicated with more specific taxonomic names anyway
                continue

            dirs = {}
            for directory in ['train', 'validate' , 'test']:
                dirs[directory] = os.path.join(os.getcwd(), directory, hierarchy[HIERARCHY_DEPTH - 1])
                if not os.path.isdir(dirs[directory]): os.makedirs(dirs[directory])


            for images in data['samples']:
                
                rand = random.uniform(0,1)
                
                for thumbnail in images['thumbs']:
                    thumbnail = thumbnail.split('/')[-1]
                    if not thumbnail: continue

                    # sample 1 in every VALIDATE images to use to assess model quality
                    if rand < TRAIN_PROP:

                        shutil.copy(os.path.join(os.getcwd(), 'images', thumbnail),
                                    os.path.join(dirs['train'], thumbnail))


                    elif TRAIN_PROP + VALIDATE_PROP > rand >= TRAIN_PROP:
                        shutil.copy(os.path.join(os.getcwd(), 'images', thumbnail),
                                    os.path.join(dirs['validate'], thumbnail))

                        
                    else:
                        shutil.copy(os.path.join(os.getcwd(), 'images', thumbnail),
                                    os.path.join(dirs['test'], thumbnail))                         

        


def create_imagefolder(directory):
    """Use the structure of the folder created above, with generic ImageLoader, as per
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#afterword-torchvision
    """

    # These Normalize values are boilerplate everywhere, what do they signify?
    # The 224 size is to coerce ResNet into working, but sources are all 120 - ONLY NECESSARY FOR PRETRAINED NETWORKS
    data_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    

    
    coccoliths = datasets.ImageFolder(root=directory,
                                      transform=data_transform)

    return coccoliths

#preprocess the images to make them usable for training
#consider adding noise and filtering images here
    #LA for grayscale '1' for binary
def process_images(directory , grayscale = 'LA' ,rotate=False):
    for root, dirs, files in os.walk(directory):
        for file in files:
            im = Image.open(os.path.join(root,file))
            
            #convert to grayscale
            im=im.convert(grayscale).convert('RGB')###
            
            if(rotate== True):
                #rotate the images and make a copy at each rotation
                for theta in [0 , 90,180,270]:
                    imr = im.rotate(theta)
                    imr.save(os.path.join(root, str(file[:-4] + str(theta) + ".jpg")))
            else:
                im.save(os.path.join(root, str(file[:-4] + "processed" + ".jpg")))
                
            #delete the original    
            os.remove(os.path.join(root,file))
        
        for dct in dirs:
            file_count = sum([len(f) for r, d, f in os.walk(os.path.join(directory,dct))])
            if(file_count == 0):
                os.rmdir(os.path.join(directory,dct))
            
        
                          
      

def create_dataloader(imagefolder):

    """Separate interface as we get the classnames from this interface"""
    dataset_loader = torch.utils.data.DataLoader(imagefolder,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=0) #must be 0 for windows

    return dataset_loader


#prints an image to console for data checking
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated




if __name__ == '__main__':
    prepare_imagefolder()
    process_images('validate','LA')
    process_images('train','LA')
   # process_images('test','LA')





