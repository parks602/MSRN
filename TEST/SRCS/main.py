import sys, os
import argparse
import warnings
from running import Run
#########################################################################################

def main():
  warnings.filterwarnings(action='ignore')
  parser = argparse.ArgumentParser(description='Pytorch DNN & MSRN')
  parser.add_argument('--cuda',action='store_true', default=False, help='Enables CUDA')
  parser.add_argument('--config',type=str, help='Configure File', required=True)
  args        = parser.parse_args()
  fconfig     = args.config
  enable_cuda = args.cuda
  
  Run(fconfig, enable_cuda)

if __name__ == "__main__":
  main()

