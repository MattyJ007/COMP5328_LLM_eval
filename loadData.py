import numpy as np

# Default to CIFAR for testing
def getData(file_path='./data/CIFAR.npz'):
  dataset = np.load(file_path)
  xtr = dataset['Xtr']
  str = dataset['Str']
  xts = dataset['Xts']
  yts = dataset['Yts']
  return xtr, str, xts, yts