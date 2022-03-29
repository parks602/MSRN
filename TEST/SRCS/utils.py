import numpy as np
from datetime import datetime, timedelta
import sys, os
import json
import torch.nn as nn
from torch.utils.data import Dataset
import torch

def readConfig(fconfig):
  with open(fconfig,'r') as f:
    confg = json.load(f)
    opti_conf = confg['option']
    dnnm_conf = confg['dnnm']
    msrn_conf = confg['msrn']
    inpu_conf = confg['input']
    outp_conf = confg['output']
    proj_conf = confg['proj']
    return opti_conf, dnnm_conf, msrn_conf, inpu_conf, outp_conf, proj_conf

def normData(new_dt, dt, val, gis, grid):
  if val == 'T3H':
    new_dt[:,0] = minmaxScaler(dt[0,:], 800, 1200)
    new_dt[:,1] = minmaxScaler(dt[1,:], -50, 50)
    new_dt[:,2] = minmaxScaler(dt[2,:], -50, 50)
    new_dt[:,3] = minmaxScaler(dt[3,:],   0,100)
    new_dt[:,4] = minmaxScaler(dt[4,:], -50, 50)
    new_dt[:,5] = minmaxScaler(gis, 0, 3000)
    new_dt[:,6] = minmaxScaler(grid[:,1], 30, 45)
    new_dt[:,7] = minmaxScaler(grid[:,0], 120, 135)
  if val == 'REH':
    new_dt[:,0] = minmaxScaler(dt[0,:], 800, 1200)
    new_dt[:,1] = minmaxScaler(dt[1,:], -50, 50)
    new_dt[:,2] = minmaxScaler(dt[2,:], -50, 50)
    new_dt[:,4] = minmaxScaler(dt[3,:],   0,100)
    new_dt[:,3] = minmaxScaler(dt[4,:], -50, 50)
    new_dt[:,5] = minmaxScaler(gis, 0, 3000)
    new_dt[:,6] = minmaxScaler(grid[:,1], 30, 45)
    new_dt[:,7] = minmaxScaler(grid[:,0], 120, 135)
  return new_dt

def minmaxScaler(dt, minval, maxval):
  scaled_dt = (dt - minval) / (maxval - minval)
  return scaled_dt

def minmaxNorm(dt, minval, maxval):
  normed_dt = dt * (maxval - minval) + minval
  return normed_dt

def embedding(data):
  embed      = nn.Embedding(18,64)
  data       = torch.from_numpy(data)
  data       = data.type(torch.LongTensor)
  embed_data = embed(data)
  return(embed_data)

def geo2torch(data):
  em_data    = embedding(data)
  em_data    = em_data.detach()
  em_data    = em_data.numpy()
  norm_data  = minmaxScaler(em_data, np.min(em_data), np.max(em_data))
  norm_data  = np.transpose(norm_data, (2,0,1))
  norm_data  = torch.from_numpy(norm_data)
  norm_data  = norm_data.type(torch.LongTensor)
  geo_map    = torch.rand(1, 64, 745, 1265)
  geo_map[0] = norm_data
  return(geo_map)

def gis2torch(data):
  gis_data   = minmaxScaler(data, 0, 3000)
  gis_data   = torch.from_numpy(gis_data)
  gis_data   = gis_data.type(torch.LongTensor)
  gis_map    = torch.rand(1, 64, 745, 1265)
  for i in range(64):
    gis_map[0][i] = gis_data
  return(gis_map)

class datasets3d(Dataset):
  def __init__(self, x):
    x = np.expand_dims(x,axis=1)
    self.x = torch.tensor(x, dtype=torch.float)
    self.rows = x.shape[0]

  def __len__(self):
    return self.rows

  def __getitem__(self, idx):
    xx = torch.tensor(self.x[idx], dtype=torch.float)
    #xx = self.x[idx].clone().detach()
    return (xx)


def MakeDir(path):
  if not os.path.exists(path):
    os.makedirs(path)
    print("Make Directories : %s" %(path))

class ARR2TXT:
  def __init__(self, dt, anltim):
    """
     fname : file (format npy)
     anltim: yyyymmddhh (str)
     nx    : 149 (nrow)
     ny    : 253 (ncol)
     nt    : forecast timeseries (interval : 3h)
    """
    self.anltim = anltim
    self.nx     = 149
    self.ny     = 253
    self.nt     = 28
    iy, ix, it = dt.shape
    if ix != self.nx or iy != self.ny:
      dt = dt.transpose([1,0,2])
    if it > 28:
      pad = it - 28
      dt = dt[:,:,pad:] ## start 6h ~ 87h
    self.data   = dt

  def _LocalTime(self, ftim):
    fmt = '%Y%m%d%H'
    lst = datetime.strptime(self.anltim,fmt) + timedelta(hours=9) + timedelta(hours=ftim)
    return datetime.strftime(lst,fmt)

  def _output_header(self, utc):
    lst = self._LocalTime(utc)
    line = '%s+%03dHOUR     %sLST\n' %(self.anltim, utc, lst)
    return line

  def toTXT(self, opath, prefix, o_in_line=10):
    MakeDir(opath)
    ofile = '%s/%s.%s' %(opath, prefix, self.anltim)
    with open(ofile, 'w') as f:
      for i in range(self.nt):
        temp = self.data[:,:,i].flatten()
        q = int(temp.size / o_in_line) + 1
        utc = i * 3 + 6
        f.write(self._output_header(utc))
        for j in range(q):
          idx = o_in_line * j
          eidx = o_in_line * (j+1)
          if temp.size < eidx: eidx = temp.size
          for k in range(o_in_line):
            if eidx <= idx + k: break
            f.write('%7.1f' %(temp[idx+k:idx+k+1]) )
          if j != (q-1): f.write('\n')
        if (temp.size % o_in_line) != 0 and i != (self.nt - 1): f.write('\n')

