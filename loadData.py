import numpy as np

# Default to CIFAR for testing
def getData(file_path):
  dataset = np.load(file_path)
  xtr_val = dataset['Xtr']
  str_val = dataset['Str']
  xts = dataset['Xts']
  yts = dataset['Yts']
  return xtr_val, str_val, xts, yts