from loadData import getData

def classify(filename, transition_matrix, experiments=10):
  print(filename)
  xtr, str, xts, yts = getData(filename)
