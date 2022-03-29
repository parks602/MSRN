import sys, os
import argparse
import torch
import numpy as np
from utils import readConfig
from msrn_dataset import DatasetMaker, geomap_make, gismap_make
from msrn_train import MSRNTrainer
import warnings
#########################################################################################

def main():
  warnings.filterwarnings(action='ignore')
  parser = argparse.ArgumentParser(description='Pytorch DNN TRAINER')
  parser.add_argument('--cuda',action='store_true', default=False, help='Enables CUDA')
  parser.add_argument('--config',type=str, help='Configure File', required=True)
  args        = parser.parse_args()
  fconfig     = args.config
  enable_cuda = args.cuda
  use_cuda    = enable_cuda and torch.cuda.is_available()
  data_conf, train_conf, model_conf, proj_conf = readConfig(fconfig)
  batchSize   = train_conf['batch_size']
  vars        = data_conf['var']
  utcs        = data_conf['utc']
  ftimes      = range(2,3)
  if use_cuda:
    cuda = torch.device('cuda')
  else:
    cuda =  torch.device('cpu')
  geomap = geomap_make(data_conf['geo_dir'], batchSize)
  gismap = gismap_make(data_conf['gis_dir'], batchSize)
  for var in vars:
    for utc in utcs:
      for ftime in ftimes:
        train_loader, valid_loader = DatasetMaker(data_conf, var,utc,ftime,batchSize)
        print("dataload finish")
        print("VAR : %s, UTC : %s, FTIME : %s MSRN model Run"%(var, utc, ftime))
        MSRNTrainer(data_conf, train_conf, model_conf, use_cuda, var, utc, ftime, geomap, gismap, train_loader, valid_loader)

if __name__ == "__main__":
  main()
