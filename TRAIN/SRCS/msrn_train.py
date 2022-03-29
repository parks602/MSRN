import torch
from progress_bar import progress_bar
import math
import numpy as np
import os, sys
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
from msrn_model import MSRN
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim


def MSRNTrainer(data_conf, train_conf, model_conf, use_cuda, var, utc, ftime, geomap, gismap, train_loader, valid_loader):
  sdate       = data_conf['sdate']
  edate       = data_conf['edate']
  lr          = train_conf['lr']
  epoch       = train_conf['epochs']
  step        = train_conf['step']
  model_path  = model_conf['path']
  model_name  = model_conf['save_name']
  patience_num= train_conf['patienct_num']
  models_name = "%s/%s" %(model_path, model_name %(var.lower(), str(ftime*3).zfill(2), utc, sdate, edate))
  early_stopping = EarlyStopping(patience = patience_num, verbose=True)
  model      = MSRN()
  criterion = nn.L1Loss(size_average = True)
  if use_cuda:
    cuda = torch.device('cuda')
    model = nn.DataParallel(model, device_ids=[0,1]).cuda()
    criterion = criterion.cuda()
  else:
    model = model.cpu()
    cuda  = torch.device('cpu')
  optimizer = optim.Adam(model.parameters(), lr = lr)

  train_loss_list, valid_loss_list = [], []
  train_loss, valid_loss = np.Inf, np.Inf
  for epoch in range(1,epoch):
    train_loss, valid_loss = train(train_loader, valid_loader, optimizer, model, cuda, criterion, epoch, lr, step, geomap, gismap)
    train_loss, valid_loss = round(train_loss, 4), round(train_loss, 4)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    early_stopping(valid_loss, model, epoch, models_name)
    if early_stopping.early_stop:
      print("Early stopping")
      break
  print(model_name)


def train(training_data_loader, valid_data_loader, optimizer, model, cuda, criterion, epoch, lr, step, geomap, gismap):
  geomap = Variable(geomap).to(cuda)
  gismap = Variable(gismap).to(cuda)
  lr = adjust_learning_rate(lr, step, optimizer, epoch - 1)
  for param_group in optimizer.param_groups:
    param_group["lr"] = lr
  print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
  val_loss_summary = 0.0
  train_loss = 0
  for iteration, batch in enumerate(training_data_loader, 1):
    input, label = Variable(batch[0]).to(cuda), Variable(batch[1].to(cuda), requires_grad = False)
    sr = model(input, geomap, gismap)
    loss = criterion(sr, label)
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if iteration == len(training_data_loader):
      val_loss_summary=0.0
      with torch.no_grad():
        for pp, val in enumerate(valid_data_loader, 1):
          val_x, val_y = Variable(val[0]).to(cuda), Variable(val[1].to(cuda), requires_grad = False)
          val_sr = model(val_x, geomap, gismap)
          val_loss = criterion(val_sr, val_y)
          val_loss_summary += val_loss.item()
          progress_bar(iteration-1, len(training_data_loader), 'Train_Loss: %.4f / Valid_Loss: %.4f' \
                       %(math.sqrt(train_loss/len(training_data_loader)), math.sqrt(val_loss_summary/(pp+1))))
  print("Train Average Loss:{:.4f} / Valid Averange Loss:{:.4f}".format(math.sqrt(train_loss/ \
         len(training_data_loader)), math.sqrt(val_loss_summary/len(valid_data_loader))))
  return math.sqrt(train_loss/len(training_data_loader)), math.sqrt(val_loss_summary/len(valid_data_loader))


def adjust_learning_rate(lr, step,optimizer, epoch):
    lr = lr * (0.5 ** (epoch // step))
    return lr


def embedding(data):
    embed = nn.Embedding(18,64)
    data = torch.from_numpy(data)
    data = data.type(torch.LongTensor)
    embed_data = embed(data)
    return(embed_data)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=100, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch,  model_name):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, model_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch,  model_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, model_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...%s'%(model_name))
        state = {'epoch': epoch, 'model': model}
        torch.save(model.state_dict(), model_name)
        self.val_loss_min = val_loss



#minmax scale defenition
def minmax(array, min_value, max_value):
    minmax_array = (array-min_value) / (max_value - min_value)
    return(minmax_array)

