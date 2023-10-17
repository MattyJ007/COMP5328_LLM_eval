import numpy as np

# Default to CIFAR for testing
def getData(file_path='./data/CIFAR.npz'):
  dataset = np.load(file_path)
  xtr = dataset['Xtr']
  str = dataset['Str']
  xts = dataset['Xts']
  yts = dataset['Yts']
  print(xtr.shape)
  print(str.shape)
  print(xts.shape)
  print(yts.shape)
  return xtr, str, xts, yts