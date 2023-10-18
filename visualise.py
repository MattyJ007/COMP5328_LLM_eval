import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

from loadData import getData

EXAMPLES=10
# An ordered list of the CIFAR class names
CIFAR_CLASS_NAMES = ["0. Plane", "1. Car", "2. Cat"]
CIFAR_CATEGORY_LABELS = dict(zip(map(str, list(range(3))), CIFAR_CLASS_NAMES))

FASHION_CLASS_NAMES = ["0. T-shirt", "1. Pants", "2. Dress"]
FASHION_CATEGORY_LABELS = dict(zip(map(str, list(range(3))), FASHION_CLASS_NAMES))

cifar_xtr, cifar_str, cifar_xts, cifar_yts = getData('./data/CIFAR.npz')
# print('CIFAR\n---------------')
# print('Training samples:\n', pd.DataFrame(cifar_str).value_counts())
# print('Test samples:\n', pd.DataFrame(cifar_yts).value_counts())

fashion5_xtr, fashion5_str, fashion5_xts, fashion5_yts = getData('./data/FashionMNIST0.5.npz')
# print('Fashion 0.5\n---------------')
# print('Training samples:\n', pd.DataFrame(fashion5_str).value_counts())
# print('Test samples:\n', pd.DataFrame(fashion5_yts).value_counts())

fashion6_xtr, fashion6_str, fashion6_xts, fashion6_yts = getData('./data/FashionMNIST0.6.npz')
# print('Fashion 0.6\n---------------')
# print('Training samples:\n', pd.DataFrame(fashion6_str).value_counts())
# print('Test samples:\n', pd.DataFrame(fashion6_yts).value_counts())

def plot_examples(title, data_set, data_noisy_labels, categories, examples, category_labels):
  fig = plt.figure(figsize=(examples, categories))  # Added a figure instance with a specified size
  count = 1
  for i in range(categories):
    categoryIndeces = np.where(data_noisy_labels == i)
    for j in range(examples):
      plt.subplot(categories, examples, count),
      plt.imshow(data_set[categoryIndeces[0][j]], cmap = 'binary')
      plt.title(category_labels[str(data_noisy_labels[categoryIndeces[0][j]])]), plt.xticks([]), plt.yticks([])
      count += 1
  
  fig.suptitle(title, fontsize=16)
  plt.tight_layout()
  plt.show()
  plt.close()

plot_examples('CIFAR Training Data', cifar_xtr, cifar_str, len(CIFAR_CLASS_NAMES), EXAMPLES, CIFAR_CATEGORY_LABELS)
plot_examples('CIFAR Test Data', cifar_xts, cifar_yts, len(CIFAR_CLASS_NAMES), EXAMPLES, CIFAR_CATEGORY_LABELS)

plot_examples('Fashion 0.5 Training Data', fashion5_xtr, fashion5_str, len(FASHION_CLASS_NAMES), EXAMPLES, FASHION_CATEGORY_LABELS)
plot_examples('Fashion 0.5 Test Data', fashion5_xts, fashion5_yts, len(FASHION_CLASS_NAMES), EXAMPLES, FASHION_CATEGORY_LABELS)

plot_examples('Fashion 0.6 Training Data', fashion6_xtr, fashion6_str, len(FASHION_CLASS_NAMES), EXAMPLES, FASHION_CATEGORY_LABELS)
plot_examples('Fashion 0.6 Test Data', fashion6_xts, fashion6_yts, len(FASHION_CLASS_NAMES), EXAMPLES, FASHION_CATEGORY_LABELS)
