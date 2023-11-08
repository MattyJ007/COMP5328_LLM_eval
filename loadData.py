import numpy as np
from skimage.color import rgb2gray

# Default to CIFAR for testing
def getData(file_path):
  dataset = np.load(file_path)
  xtr_val = dataset['Xtr']
  str_val = dataset['Str']
  xts = dataset['Xts']
  yts = dataset['Yts']

  #convert to greyscale
  if xtr_val.shape[-1] == 3:
        xtr_val = rgb2gray(xtr_val)
        xts = rgb2gray(xts)

  #flatten the image into 1D array
  xtr_val = xtr_val.reshape(xtr_val.shape[0], -1)
  xts = xts.reshape(xts.shape[0], -1)

  #normalize the data
  xtr_val = xtr_val.astype('float32') / 255.
  xts = xts.astype('float32') / 255.

  return xtr_val, str_val, xts, yts
