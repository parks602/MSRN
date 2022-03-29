import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import minmaxScaler, minmaxNorm, geo2torch, gis2torch, datasets3d

def SuperResolution(inter_data, msrn_model, device, geo_1, gis_1, var):
  geo_map   = geo2torch(geo_1)
  gis_map   = gis2torch(gis_1)
  test_x    = np.asarray(inter_data)
  if var    == 'T3H':
    min_x, max_x = -50, 50
  elif var  == 'REH':
    min_x, max_x = 0, 100
  test_x    = minmaxScaler(test_x, min_x, max_x)
  data_x    = []
  data_x.append(test_x)
  test_x    = datasets3d(data_x)
  x_data    = DataLoader(dataset = test_x, batch_size = 1, shuffle=False)
  for i in x_data:
    x       = i.to(device)
    out     = msrn_model(i, geo_map, gis_map)
    out     = out.to('cpu')
    out     = out[0][0].data.numpy()
    out     = minmaxNorm(out, min_x, max_x)
  return(out)

