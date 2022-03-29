import sys, os
import numpy as np
import torch
import skimage.transform as trans
from torch.utils.data import Dataset
from datetime import datetime, timedelta
import random
import torch.nn as nn
from torch.utils.data import DataLoader


def RandomDate(sdate, edate):
  fmt = "%Y%m%d%H"
  dt_sdate = datetime.strptime(sdate, fmt)  ### str -> datetime
  dt_edate = datetime.strptime(edate, fmt)
  day_list = []
  now = dt_sdate
  while now<=dt_edate:
    ex_sdate = now.strftime(fmt)
    day_list.append(ex_sdate)
    now = now + timedelta(days=1)
  train_list = sorted(random.sample(day_list, int(len(day_list)*8//10)))
  for i in range(len(train_list)):
    day_list.remove(train_list[i])
    valid_list = day_list
  return (train_list, valid_list)

def FileExists(path):
  if not os.path.exists(path):
    print("Can't Find : %s" %(path))
    return False
  else:
    return True

class ReadPPM:
  def __init__(self,fname):
    if not os.path.exists(fname):
      print("Can't find %s" %(fname))
      sys.exit(1)

    data, dset = {}, []
    with open(fname,'r') as f:
      while True:
        line = f.readline().strip('\n')
        if not line : break

        if line[0] != ' ':      # Read Anal & Forecast Time
          if dset:
            if ftim not in data:
              data[ftim] = {}
            data[ftim]['analtim'] = analtim
            data[ftim]['lstim'] = lstim
            data[ftim]['data'] = dset
            dset  = []
          analtim = line[0:10]
          ftim    = line[11:14]
          lstim   = line[23:33]
          #print(analtim, ftim, lstim)
        else:                  # Read Data
          temp = map(float,line.split())
          dset.extend(temp)
      ### Read Last Data
      if dset:
        if ftim not in data:
          data[ftim] = {}
        data[ftim]['analtim'] = analtim
        data[ftim]['lstim'] = lstim
        data[ftim]['data'] = dset

    self.data = data

  def GetData(self):
    return self.data

def CalculatePadsForUpscale(nn, upscale):
  res = nn%upscale
  if res % 2 == 0:
    pads = [ res // 2, res // 2 ]
  else:
    pads = [ res - res // 2, res // 2 ]
  return pads

### Interpolate Raw Data -> Down Scaled Data
def ExtrAndResize(fname, nx, ny, upscale, ftime,order):
  ppm = ReadPPM(fname)
  dic_data = ppm.GetData()
  time_list = list(dic_data.keys())
  xpads = CalculatePadsForUpscale(nx,upscale)
  ypads = CalculatePadsForUpscale(ny,upscale)

  xdata, ydata = [], []
  for i, ftim in enumerate(time_list):
    tdata = np.asarray(dic_data[ftim]['data'])
    tdata = np.transpose(np.reshape(tdata, (ny,nx)))
    tdata = tdata[xpads[0]:nx-xpads[1],ypads[0]:ny-ypads[1]] ### Remove paddings for upscaling
    resized_data = trans.resize(tdata,(nx // upscale, ny // upscale), order=order)
    #resized_data = tdata[:nx//upscale, :ny//upscale]
    xdata.append(resized_data)
    ydata.append(tdata)
  xdata = np.transpose(np.asarray(xdata),(1,2,0))
  ydata = np.transpose(np.asarray(ydata),(1,2,0))
  xdata = xdata[:,:,ftime]
  ydata = ydata[:,:,ftime]
  return xdata, ydata

def MakeMSRNDataset(conf_inputs, var, utc, ftime, date_list):
  fmt = "%Y%m%d%H"
  #=== Config
  nx             = conf_inputs['nx']
  ny             = conf_inputs['ny']
  xipath         = conf_inputs['in_path']
  xprefix        = conf_inputs['in_prefix']
  upscale_factor = conf_inputs['upscale_factor']
  order          = conf_inputs['interpol_order']
  #=== Read Data
  xdata, ydata = [],[]
  for date in date_list:
    fname = "%s/%s_5km_%s.%s" %(xipath, xprefix, var, date)
    if not FileExists(fname):
      continue
    xdat, ydat = ExtrAndResize(fname, nx, ny, upscale_factor, ftime, order)
    if var == "T3H":
      xdat = MinMaxscaler(-50, 50, xdat)
      ydat = MinMaxscaler(-50, 50, ydat)
    elif var == "REH":
      xdat = MinMaxscaler(0, 100, xdat)
      ydat = MinMaxscaler(0, 100, ydat)
    xdata.append(xdat)
    ydata.append(ydat)
  xdata = np.asarray(xdata)
  ydata = np.asarray(ydata)
  return(xdata, ydata)
'''
  if len(valid_x)//batch_size == 0:
    pass
  else:
    valid_x = valid_x[0:(len(valid_x)-(len(valid_x)%batch_size))]
    valid_y = valid_y[0:(len(valid_x)-(len(valid_x)%batch_size))]

  return np.asarray(train_x), np.asarray(train_y), np.asarray(valid_x), np.asarray(valid_y)
'''
class datasets3d(Dataset):
  def __init__(self, x, y):
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    self.x = torch.tensor(x, dtype=torch.float)
    self.y = torch.tensor(y, dtype=torch.float)
    if x.shape[0] == y.shape[0]:
      self.rows = x.shape[0]
    else:
      print("x & y nsamples are not matched")
      sys.exit(-1)
  def __len__(self):
    return self.rows

  def __getitem__(self, idx):
    xx = torch.tensor(self.x[idx], dtype=torch.float)
    yy = torch.tensor(self.y[idx], dtype=torch.float)
    return (xx, yy)

def MinMaxscaler(Min, Max, data):
  minmax = (data - Min)/(Max - Min)
  return(minmax) 

def embedding(data):
  embed = nn.Embedding(18,64)
  data = torch.from_numpy(data)
  data = data.type(torch.LongTensor)
  embed_data = embed(data)
  return(embed_data)

def geomap_make(geo_dir, batchSize):
  geo_map = np.load(geo_dir)[2:147,2:252,2]
  geo_map = embedding(geo_map)
  geo_map = geo_map.detach()
  geo_map = geo_map.numpy()
  geo_map = MinMaxscaler(np.min(geo_map),np.max(geo_map),geo_map)
  geo_map = np.transpose(geo_map, (2,0,1))
  geo_map = torch.from_numpy(geo_map)
  geo_map = geo_map.type(torch.LongTensor)
  geomap = torch.rand(batchSize,64,145,250)
  for i in range(batchSize):
    geomap[i] = geo_map
  return(geomap)

def gismap_make(gis_dir, batchSize):
  gis = np.load(gis_dir)[2:147,2:252]
  gis = MinMaxscaler(0,3000,gis)
  gis = torch.from_numpy(gis)
  gis = gis.type(torch.LongTensor)
  gismap = torch.rand(batchSize,64,145,250)
  for i in range(batchSize):
    for j in range(64):
      gismap[i][j] = gis
  return(gismap)
                          

def DatasetMaker(conf_inputs, var, utc, ftime, batchSize):
  sdate = conf_inputs['sdate']+utc
  edate = conf_inputs['edate']+utc

  train_list, valid_list, = RandomDate(sdate, edate)
  train_x, train_y        = MakeMSRNDataset(conf_inputs, var, utc, ftime, train_list)
  valid_x, valid_y        = MakeMSRNDataset(conf_inputs, var, utc, ftime, valid_list)
  train_dataset           = datasets3d(train_x, train_y)
  train_loader            = DataLoader(dataset = train_dataset, batch_size = batchSize, shuffle = True)
  valid_dataset           = datasets3d(valid_x, valid_y)
  valid_loader            = DataLoader(dataset = valid_dataset, batch_size = batchSize, shuffle = False)
  return(train_loader, valid_loader)

