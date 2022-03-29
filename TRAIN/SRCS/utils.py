import numpy as np
import pandas as pd
import os, sys
import json
from scipy.io import FortranFile

def readConfig(fconfig):
  with open(fconfig,'r') as f:
    confg = json.load(f)
    data_conf = confg['data']
    train_conf = confg['train']
    model_conf = confg['model']
    proj_conf = confg['proj']
    return data_conf, train_conf, model_conf, proj_conf


def getnearpos(array_lat, array_long, value_lat, value_long):
  idx = (np.abs(((array_lat - value_lat) ** 2 + (array_long - value_long) ** 2)) ** 0.5).argmin()
  a = idx // 253
  b = idx % 253
  pos = (a,b)
  return (pos)


def Fortran2npy(fname):
  with FortranFile(fname, 'r') as f:
    info    = f.read_ints(np.int32)
    stnlist = f.read_ints(np.int32)
    data    = f.read_reals(np.float32)
    data    = np.reshape(data, info[:7:-1])
  data = np.transpose(data)
  data = np.where(data == -999., np.nan, data)
  return (info, stnlist, data)



def col_obs(targs, data_dir, val):
  ta_name = targs['names']
  for i in range(len(ta_name)):
    dt = targs['names'][i]
    stn_name = targs[dt]['info']
    fname = '%s%s' %(data_dir, targs[dt]['data'] %(val))
    info, stnlist, data = Fortran2npy(fname)
    data = data[:,:,:,:,:24]
    stn_csv = pd.read_csv('%s/%s'%(data_dir, stn_name))
    stn_npy = np.array((stn_csv))
    for j, ak in enumerate(stnlist):
      if i==0 and j==0:
        new_data = np.expand_dims(data[j], axis=0)
        new_stn = stnlist[j]
        new_pos_dict = stn_npy[np.where(stn_npy[:,0]==ak)]
      else:
        if not len(stn_npy[np.where(stn_npy[:,0]==ak)])== 0:
          new_data = np.vstack((new_data, np.expand_dims(data[j], axis=0)))
          new_stn = np.vstack((new_stn, stnlist[j]))
          new_pos_dict = np.vstack((new_pos_dict, stn_npy[np.where(stn_npy[:,0]==ak)]))
  if new_data.shape[0] == new_stn.shape[0] and new_stn.shape[0] == new_pos_dict.shape[0]:
      print("all dataset, shape is same")
  else :
      print('ERROR : dataset shape is not same')
      print(new_data.shape, new_stn.shape, new_pos_dict.shape)
      sys.exit()
  np.save('%s%s_obs_data.npy' %(obs_dir, var), new_data)
  np.save('%s%s_obs_stn.npy' %(obs_dir, var), new_stn)
  np.save('%s%s_obs_pos_dict.npy' %(obs_dir, var), new_pos_dict)
  return(new_data, new_stn, new_pos_dict)
