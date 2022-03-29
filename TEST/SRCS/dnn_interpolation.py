from __future__ import print_function
import torch
import torch.utils.data as data_utils
import os
import numpy as np
from utils import normData

def SR_dataset_maker(var, utc, ftime, dnn_model, device, gis_5, grid_5, inputs):
  test       = inputs
  re_test    = np.reshape(test, (5,(test.shape[1]*test.shape[2])))
  new_data   = np.zeros((re_test.shape[1], 8))
  new_gis    = np.reshape(gis_5,(1,149*253))
  new_grid   = np.reshape(grid_5,(149*253,2))
  norm_data  = normData(new_data, re_test, var, new_gis, new_grid)
  test_x     = torch.from_numpy(norm_data[:,4:]).float()
  trn_loader = data_utils.DataLoader(test_x, batch_size = 5000, shuffle =False)
  for k, p in enumerate(trn_loader):
    x          = p.to(device)
    out        = dnn_model(x)
    out        = out.to('cpu')
    out        = out.data.numpy()
    if k == 0:
      save_out = out
    else:
      save_out = np.hstack((save_out, out))
  save_npy    = np.zeros((test.shape[1],test.shape[2]))
  for i in range(149):
    for j in range(253):
      save_npy[i,j]=save_out[(i*253)+j]
  return (save_npy)

