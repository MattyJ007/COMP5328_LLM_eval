from loadData import getData

print('CIFAR')
cifar_xtr, cifar_str, cifar_xts, cifar_yts = getData('./data/CIFAR.npz')
print('Fashion 0.5')
fashion5_xtr, fashion5_str, fashion5_xts, fashion5_yts = getData('./data/FashionMNIST0.5.npz')
print('Fashion 0.6')
fashion6_xtr, fashion6_str, fashion6_xts, fashion6_yts = getData('./data/FashionMNIST0.6.npz')
