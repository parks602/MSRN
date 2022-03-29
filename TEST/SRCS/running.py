from datetime import datetime, timedelta
import numpy as np
import torch
from dnn_interpolation import SR_dataset_maker
from msrn_super_resolution import SuperResolution
from utils import readConfig, ARR2TXT
from dnn_model import MLPRegressor
from msrn_model import MSRN
from datasets import MakeDataset
from collections import OrderedDict
import sys
#########################################################################################

def Run(fconfig, enable_cuda):
  use_cuda = enable_cuda and torch.cuda.is_available()
  option, dnnm, msrn, inputs, outputs, porj_conf= readConfig(fconfig)
  utcs          = inputs['utcs']
  vars          = inputs['vars']
  ftimes        = range(2,30)
  sdate, edate  = inputs['sdate'], inputs['edate']
  fmt           = '%Y%m%d%H'
  print("=============================================")
  print("> inputs")
  print(inputs)
  print("> outputs")
  print(outputs)
  print("=============================================")
  gis_5         = np.load(inputs['gis5_name'])
  grid_5        = np.load(inputs['grid5_name'])
  gis_1         = np.load(inputs['gis1_name'])
  geo_1         = np.load(inputs['geo1_name'])

  dnn_md_info   = dnnm['dnnm_model_path']
  dnn_md_sdate  = dnn_md_info['sdate']
  dnn_md_edate  = dnn_md_info['edate']

  msrn_md_info  = msrn['msrn_model_path']
  msrn_md_sdate = msrn_md_info['sdate']
  msrn_md_edate = msrn_md_info['edate']

  device        = torch.device('cuda' if use_cuda else 'cpu')
  print(device)

  #=== Run
  if option == '0':
    for utc in utcs:
      for var in vars:
        xdata, datelist  = MakeDataset(inputs, porj_conf, gis_5, utc, var)
        print('Correction Start - Var : %s, UTC : %s'%(var, utc))
        for k, dt in enumerate(xdata):
          save_data = np.zeros((149, 253, 28))
          for ftime in ftimes:
            #print(utc, var, ftime*3)
            dnn_name   = dnn_md_info['model_name']%(var.lower(),str(ftime*3).zfill(2), utc, \
                                                    dnn_md_sdate, dnn_md_edate)
            load_name  = "%s%s"%(dnn_md_info['path'],dnn_name)
            dnn_model  = MLPRegressor()
            state_dict = torch.load(load_name, map_location = device)
            ### For Remove module
            new_state_dict = OrderedDict()
            for key, val in state_dict.items():
                if 'module' in key:
                    name = key[7:] # remove 'module'
                else: 
                    name = key
                new_state_dict[name] = val
            #dnn_model.load_state_dict(state_dict, strict=False)
            dnn_model.load_state_dict(new_state_dict)
            dnn_model.eval()
            #=== Preprocessing
            inter_data = SR_dataset_maker(var, utc, ftime, dnn_model, device, gis_5, grid_5, dt[:,:,:,ftime-2])
            save_data[:,:,ftime-2] = inter_data
          print('Correction with DNNM',datelist[k],'is finished')
          ppm = ARR2TXT(save_data,datelist[k])
          save_name = '%s_%s'%(outputs['dnnm_prefix'], var)
          ppm.toTXT(outputs['dnnm_opath'], save_name)


  if option == '1':
    for utc in utcs:
      for var in vars:
        xdata, datelist  = MakeDataset(inputs, porj_conf, gis_5, utc, var)
        print('Correction & Super-Resolution Start - Var : %s, UTC : %s'%(var, utc))
        for k, dt in enumerate(xdata):
          dnn_save_data = np.zeros((149, 253, 28))
          save_data = np.zeros((745, 1265, 28))
          for ftime in ftimes:
            dnn_name        = dnn_md_info['model_name']%(var.lower(),str(ftime*3).zfill(2), utc, \
                                                    dnn_md_sdate, dnn_md_edate)
            dnn_load_name   = "%s%s"%(dnn_md_info['path'],dnn_name)
            dnn_model       = MLPRegressor()
            dnn_state_dict  = torch.load(dnn_load_name, map_location = device)
            ### For Remove module
            new_state_dict = OrderedDict()
            for key, val in dnn_state_dict.items():
                if 'module' in key:
                    name = key[7:] # remove 'module'
                else:
                    name = key
                new_state_dict[name] = val
            dnn_model.load_state_dict(new_state_dict)
            #dnn_model.load_state_dict(dnn_state_dict,strict=False)
            dnn_model.eval()
            inter_data      = SR_dataset_maker(var, utc, ftime, dnn_model, device, gis_5, grid_5, dt[:,:,:,ftime-2])
            dnn_save_data[:,:,ftime-2] = inter_data
            msrn_name       = msrn_md_info['model_name']%(var.lower(),str(ftime*3).zfill(2), utc, \
                                                     str(msrn_md_sdate)+utc, str(msrn_md_edate)+utc)
            msrn_load_name  = "%s%s"%(msrn_md_info['path'],msrn_name)
            msrn_model = MSRN()
            msrn_state_dict  = torch.load(msrn_load_name, map_location = device)
            ### For Remove module
            new_state_dict = OrderedDict()
            for key, val in msrn_state_dict.items():
                if 'module' in key:
                    name = key[7:] # remove 'module'
                else:
                    name = key
                new_state_dict[name] = val
            msrn_model.load_state_dict(new_state_dict)
            #msrn_model.load_state_dict(msrn_state_dict,strict=False)
            msrn_model.eval()
            sr_data         = SuperResolution(inter_data, msrn_model, device, geo_1, gis_1, var)  

            save_data[:,:,ftime-2] = sr_data
          print('Super Resolution with MSRN',datelist[k],'is finished')
          ppm = ARR2TXT(dnn_save_data,datelist[k])
          save_name = '%s_%s'%(outputs['dnnm_prefix'], var)
          ppm.toTXT(outputs['dnnm_opath'], save_name)

          ppm = ARR2TXT(save_data,datelist[k])
          save_name = '%s_%s'%(outputs['msrn_prefix'], var)
          ppm.toTXT(outputs['msrn_opath'], save_name)

