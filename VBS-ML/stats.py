import numpy
import torch
from numpy import *
a=numpy.loadtxt('./result/L1.txt')
print(numpy.mean((a)))
# 10000  0.07922   0.07915
# 1      0.07383   0.07381
#     epoch 9500 0.07377  9000 0.07374  8000 0.07400  7000  0.0738  6000 0.07388
#           5000 0.0737        4000 0.07418     3000 0.0738    2000 0.0757

#  256  128 64 1  -2  epoch 2000  0.0739

#  128 64 32 1    epoch 2000  0.07313           5000  0.07328    9000 0.08347
#  noise 100      epoch 2000  0.08407           4000  0.081968  6000 0.08169
#TANH  0.0780


#  64  32 16 1    epoch 2000  0.07331

#  32  16 8 1     epoch 2000  0.07473


#  128 128 64 64 32 32 1   epoch 10000    0.07318  9000 0.07321  8000  0.0733  7000 0.0735



import torch
pthfile = '/home/yuhang/PycharmProjects/AGT/Challenge2/checkpoint/best_epoch.pt'
net = torch.load(pthfile)
print(net)