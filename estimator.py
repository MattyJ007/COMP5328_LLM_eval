# Implement an estimator to estimate the transition matrix. (Note that you can use the
# provided transition matrices of the first two datasets to validate the effectiveness of your
# transition matrix estimator.) Then use the estimated transition matrix for classification.
# You need to include your estimated transition matrix in the final report. You also need to
# report the mean and the standard derivation of the test accuracy for each class ifier.
from loadData import getData

def estimate(filename):
  print(filename)
  xtr, str, xts, yts = getData(filename)