import os
import numpy as np

cwd = '/zhome/27/c/138037/share/data/train/Broken'

def mask(img):
	image = img[:,:,:7]

	#create a simple mask, and make everything else 0
	mask = image[:,:,4].copy()        
	mask[image[:,:,4] < 35] = 0 
	mask[image[:,:,4] >= 35] = 1
	mask[:5,:5] = 0
	image[mask==0] = 0
	return image
	

path = os.getcwd() + '/data/train/'
classes = ['Broken'] #[Barley, Broken, Oat, Rye, Wheat]

counter = 0
channel_min = np.zeros((7,1))
channel_max = np.zeros((7,1))

for category in classes:
	file_paths = next(os.walk(path + '/' + category))[2]	

	for file in file_paths:			
		counter += 1		
		image = mask(np.load(os.path.join(path + '/' + category + '/' + file)))
		for channel in range(6):
			if np.min(image[:,:,channel].any()) < channel_min[channel].any():
				channel_min[channel] = np.min(image[:,:,channel]) 
			if np.max(image[:,:,channel].any()) > channel_max[channel].any():
				channel_max[channel] = np.max(image[:,:,channel]) 

print([f'{category} min={channel_min[channel]} max={channel_max[channel]}' for category in classes for channel in range(6)])

    
    
