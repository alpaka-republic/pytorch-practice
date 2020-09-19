import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
from pytorch_memlab import profile
import time

class Profiler(nn.Module):
  def __init__(self):
    self.device = torch.device('cuda:7')
    # self.device = torch.device('cpu')
    # self.device = torch.cuda.set_device(7)

  @profile
  def profile(self):
    x = torch.Tensor(50000, 3000)
    x = x.to(self.device)
    y = torch.rand(50000, 3000)
    y = y.to(self.device)
    z = x + y
    print(z)
    
for _ in range(10):
  profiler = Profiler()
  start = time.time()
  profiler.profile()
  print(time.time() - start)
