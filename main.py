import sys
from classifier import classify
from estimator import estimate

FASHION_5_TRANSITION_MATRIX=[[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
FASHION_6_TRANSITION_MATRIX=[[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]]
CIFAR_TRANSITION_MATRIX=[[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]]


def help():
  print("Available flags:")
  print("- fashion5: Classify FashionMNIST0.5 dataset")
  print("- fashion6: Classify FashionMNIST0.6 dataset")
  print("- cifar: Classify CIFAR dataset")
  print("- all: Classify all datasets")
  print("- estimate: Estimate Transition Matrix from CIFAR dataset")
  print("- help: Show this help message")

def main():
  actions = {
      "fashion5": lambda: classify('./data/FashionMNIST0.5.npz', FASHION_5_TRANSITION_MATRIX),
      "fashion6": lambda: classify('./data/FashionMNIST0.6.npz', FASHION_6_TRANSITION_MATRIX),
      "cifar": lambda: classify('./data/CIFAR.npz', CIFAR_TRANSITION_MATRIX),
      "all": lambda: [
        classify('./data/FashionMNIST0.5.npz', FASHION_5_TRANSITION_MATRIX),
        classify('./data/FashionMNIST0.6.npz', FASHION_6_TRANSITION_MATRIX),
        classify('./data/CIFAR.npz', CIFAR_TRANSITION_MATRIX)],
      "estimate": lambda: estimate('./data/CIFAR.npz'),
      "help": help
  }
  
  if len(sys.argv) != 2:
      print("Usage: python main.py <flag>")
      help()
      sys.exit(1)

  flag = sys.argv[1]
  if flag in actions:
      actions[flag]()
  else:
      print("Invalid flag. Please use one of: fashion5, fashion6, cifar, all, estimate, help")
      sys.exit(1)

if __name__ == "__main__":
  main()
