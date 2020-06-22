import os
import numpy as np
import cv2

def masker(image):
    #create a simple mask, and make everything else 0
    mask = np.zeros((image.shape[0],image.shape[1]))
    temp_blue = image[:,:,1].copy()
    temp_blue[temp_blue==0] = 1
    mask[image[:,:,4]/temp_blue >= 1] = 1
    mask[image[:,:,4] >= 40] = 1
    mask[:5,:5] = 0
    image[mask==0] = 0
    return image, mask

path = '/zhome/27/c/138037/eyefoss-project-blobs/'
classes = {'Barley':0, 'Broken':1, 'Oat':2, 'Rye':3, 'Wheat':4}

data = np.zeros(5).reshape(1,5)
for category in list(classes.keys()):
    #print(category)
    file_paths = next(os.walk(path  + category))[2]
    for file in file_paths:
        #print(file)
        image, mask = masker(np.load(os.path.join(path + category + '/' + file)))
        intensity = np.mean(image[np.ma.where(image)])
        size = len(np.where(mask)[0]) / (image.shape[0]*image.shape[1]) 
        min_x = np.min(np.where(mask)[1])
        max_x = np.max(np.where(mask)[1])
        min_y = np.min(np.where(mask)[0])
        max_y = np.max(np.where(mask)[0])
        height = max_y-min_y
        width = max_x-min_x
        depth = np.max(image[:,:,6])
        aspect =  image.shape[1] / image.shape[0]
        weighted_aspect = aspect /size
        ldiag = len(np.ma.where(np.diag(image[:,:,0]))
        rdiag = len(np.ma.where(np.diag(np.fliplr(image[:,:,0])))
        
        data = np.append(data, [[height, width, depth, aspect, weighted_aspect, ldiag, rdiag, intensity, size, depth, classes[category]]], axis=0)
        
np.save('/zhome/27/c/138037/share/src/features_manual.npy', data)
