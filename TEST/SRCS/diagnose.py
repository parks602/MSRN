########################################################################
### Author      : KOAST
### Date        : 2019.5.17
### Description : PostProcessing For Variables
### Update info :
########################################################################
import numpy as np
import math
import sys
########################################################################
def diagnose_3h_temperature(t2m_dfs, ter_dfs, ter_5km_dfs, lapse_rate, kel2cel):
  ''' 
    convert the temperature(C) at 2M to air temperature for human. 
  '''
  t3h = t2m_dfs[:,:,:]

  nx, ny, nt = t3h.shape
  for i in range(nx):
    for j in range(ny):
      t3h[i,j,:] = t3h[i,j,:] - kel2cel + lapse_rate * (ter_dfs[i,j] - ter_5km_dfs[i,j])
  return t3h 

def diagnose_pressure(ps_dfs, ttz_dfs, ter_dfs, ter_5km_dfs):
  ### ttz_dfs :: (lon, lat, ftim)  using 0 index level (1000hpa)
  nx, ny, nt = ps_dfs.shape
  g = 9.81
  R = 287.04
  psfc = ps_dfs / 100.0  ### Pa to hPa

  dz = ter_dfs[:,:] - ter_5km_dfs[:,:]
  for t in range(nt):
    psfc[:,:,t] = psfc[:,:,t] + psfc[:,:,t] * (np.exp(g*dz[:,:]/R/(ttz_dfs[:,:,t] + 2.0 + 0.0025*dz)) - 1)
  return psfc

def cntl_element_value(data, element, null=-999.0):
  if element in ['T3H','TMX','TMN','UUU','VVV']:
    maximum = 50.0
    minimum = -50.0
    precision = 1
  elif element in ['WSP']:
    maximum = 75.0
    minimum = 0.0
    precision = 1
  elif element in ['REH','POP']:
    maximum = 100.0
    minimum = 0.0
    precision = 0
  elif element in ['SKY']:
    maximum = 4.0
    minimum = 1.0
    precision = 0
  elif element in ['PTY']:
    maximum = 3.0
    minimum = 0.0
    precision = 0
  elif element in ['RN3','SN3','PRC']:
    maximum = 800.0
    minimum = 1.0
    precision = 1
  elif element in ['SNW']:
    maximum = 200.0
    minimum = 0.0
    precision = 1
  elif element in ['WAV']:
    maximum = 30.0
    minimum = 0.0
    precision = 1
  else:
    sys.exit('element error')
  
  #print(maximum, minimum, precision)
  ### Check whether out of range
  double = 2.0 * maximum
  data[ data > double ] = null
  data[ (data > maximum) & (data <= double) ] = maximum
  data[ data < null ] = null
  data[ (data >= null) & (data < minimum) ] = minimum
  ### Check precision
  data = np.round(data, precision)
  return data
