############################################################################################
### Author      : KOAST
### Date        : 2019.12.02
### Description : Parsing Program For ERA5 & PPM in ascii format
### Update info :
############################################################################################
import numpy as np
import os, sys
import struct
from datetime import datetime, timedelta
from eccodes import codes_grib_new_from_file, codes_get, codes_release, \
                    codes_get_array
############################################################################################
from proj import F_PROJ
from diagnose import diagnose_3h_temperature, cntl_element_value, \
                     diagnose_pressure
############################################################################################
class ReadNWPD:
  def __init__(self,fname, nullvalue=-999.):
    if not os.path.exists(fname):
      print("Can't find : %s" %(fname))
      sys.exit(1)

    self._null = nullvalue
    self._paramId = nullvalue
    self._paramName = nullvalue
    self._shortName = nullvalue
    self._dataDate = nullvalue
    self._dataTime = nullvalue
    self._valDate  = nullvalue
    self._valTime  = nullvalue
    self._gridType = nullvalue
    self._levType  = nullvalue
    self._npnt     = nullvalue
    self._lats     = nullvalue
    self._lons     = nullvalue
    self._values   = nullvalue
    ### Read File
    self._fIn = open(fname,'r')
    self._gid = codes_grib_new_from_file(self._fIn)
    self._isClosed=False # File Close Check
    self._isEOF= False   # End of File
    if self._gid is None:
      sys.stderr.write("Error, Check File\n")
      sys.exit(1)

  def _GoToNext(self):
    codes_release(self._gid)
    self._gid = codes_grib_new_from_file(self._fIn)

    if self._gid is None:
      if self._isEOF is False:
        self.CloseFile()
        self._isEOF = True
      return False
    return True

  def CloseFile(self):
    if not self._isClosed:
      self._isClosed=True
      self._fIn.close()
      #print("File Closed")

  def GetValueByParamId(self,Id,paramLevel=None):
    if self._isEOF is True:
      #print("File Ends")
      return [self._null] * 4
    level=-1
    while True:
      paramId = codes_get(self._gid,'paramId')
      #print(paramId, codes_get(self._gid,'name'), codes_get(self._gid,'level'),int(codes_get(self._gid,'validityTime') / 100))
      ### Check Parameter's Id
      if str(paramId) == str(Id):
        self._paramId  = str(paramId)
        self._paramName= codes_get(self._gid,'name')
        self._shortName= codes_get(self._gid,'shortName')
        self._dataDate = str(codes_get(self._gid,'dataDate'))
        self._dataTime = int(codes_get(self._gid,'dataTime') / 100)
        self._valDate  = str(codes_get(self._gid,'validityDate'))
        self._valTime  = int(codes_get(self._gid,'validityTime') / 100)
        self._gridType = codes_get(self._gid,'gridType')
        self._levType  = codes_get(self._gid,'levtype')  # sfc, pl, ml
        self._npnt     = int(codes_get(self._gid,'numberOfPoints'))
        ### Calculate Forecast time
        analtim = datetime.strptime(self._dataDate,'%Y%m%d').date()
        valtim  = datetime.strptime(self._valDate, '%Y%m%d').date()
        hdiff   = (valtim - analtim).days * 24    # days to hours
        self._ftim = hdiff + self._valTime
        ### Check Parameter's Level
        if not paramLevel is None:
          level = codes_get(self._gid,'level')
          if int(level) != int(paramLevel):
            if not self._GoToNext():  # Go to the Next Parameter
              break
            continue
        Ni, Nj = codes_get(self._gid,'Ni'), codes_get(self._gid,'Nj')  ## nlon, nlat
        lat1d  = codes_get_array(self._gid,'latitudes')
        lat2d  = np.array(lat1d).reshape((Nj,Ni))
        lon1d  = codes_get_array(self._gid,'longitudes')
        lon2d  = np.array(lon1d).reshape((Nj,Ni))
        data1d = codes_get_array(self._gid, "values")
        values = np.array(data1d).reshape((Nj,Ni))

        self.lats = lat2d[::-1,:]     ### desc -> asc
        self.lons = lon2d[::-1,:]     ### desc -> asc
        self.values = values[::-1,:]  ### desc -> asc

        #self.OutputParamInfo()
        self._GoToNext()
        return self.lats, self.lons, self.values, self._ftim

      if not self._GoToNext():
        break
    return [self._null] * 4

  def OutputParamInfo(self):
    print("paramId : %s"  %(self._paramId))
    print("paramName : %s (%s)" %(self._paramName,self._shortName))
    print("DateTime : %s (%d)" %(self._dataDate, self._dataTime))
    print("ValidityDateTime : %s (%d)" %(self._valDate, self._valTime))
    print("ForecastTime : %d" %(self._ftim))
    print("GridType : %s" %(self._gridType))
    print("Level : %s" %(self._levType))

class ReadLandsea:
  def __init__(self, fname, nx, ny):
    if not os.path.exists(fname):
      print("Can't find %s" %(fname))
      sys.exit(1)

    self.nx = nx
    self.ny = ny

    length = nx * ny

    dat = []
    with open(fname,'r') as f:
      lines = f.readlines()
      for line in lines:
        data = line.split()
        data = map(int,data)
        dat.extend(data)

    self.data = np.transpose(np.asarray(dat).reshape(ny,nx))  # order : lat,lon -> lon,lat

  def getData(self):
    return self.data


def ExtractValueFromGrib(dtime, paramId, ipath):
  '''
    dtime : [datetime] Datetime of the dataset you want to create
    paramId  : [dict] parameter id for extracting values
    ipath    : [str] input path
  '''
  if dtime < datetime(2016,6,1):
    solution = "N512"
    prefix= "umgl_%s_ph_093" %(solution.lower())
  elif dtime < datetime(2018,6,1):
    solution = "N768"
    prefix= "umgl_%s_pgg_093" %(solution.lower())
  elif dtime >= datetime(2018,6,1):
    solution = "N128"
    prefix= "umgl_%s_pgg_093" %(solution.lower())
  else:
    print("Check datetime : ", dtime)
    sys.exit()
  data = {}
  uuu, vvv, prs, hum, tmp, ztmp = [], [], [], [], [], []
  is_hgt = False
  fmt = "%Y%m%d%H"
  dt = dtime.strftime(fmt)
  print(solution)
  ipath = ipath %(solution)
  fname = "%s/%s/%s/%s.%s.gb2" %(ipath,dt[:6], dt[6:8], prefix, dt)
  print(fname)
  if not FileExists(fname): return None
  grib = ReadNWPD(fname)
  if solution == "N512":
    fname = "%s/%s/%s/%s.%s.gb2" %(ipath,dt[:6], dt[6:8], prefix.replace('ph','pi',1), dt)
    if not FileExists(fname):
      grib.CloseFile()
      return None
    grib2 = ReadNWPD(fname)
  else:
    grib2 = ReadNWPD(fname)
  while True:
    lat, lon, val, tim = grib.GetValueByParamId(paramId['uwind'],10)
    if tim < -990: break
    uuu.append(val)
    lat, lon, val, tim = grib.GetValueByParamId(paramId['vwind'],10)
    if tim < -990: break
    vvv.append(val)
    lat, lon, val, tim = grib.GetValueByParamId(paramId['t'],2)
    if tim < -990: break
    tmp.append(val)
    lat, lon, val, tim = grib.GetValueByParamId(paramId['r'],2)
    if tim < -990: break
    hum.append(val)
    lat, lon, val, tim = grib.GetValueByParamId(paramId['mslp'],0)
    if tim < -990: break
    prs.append(val)
    if not is_hgt:
      lat, lon, val, tim = grib.GetValueByParamId(paramId['h'],0)
      if tim < -990: break
      hgt = val[::-1,:] ### lat asc -> desc
      is_hgt = True
  while True:
    lat, lon, val, tim = grib2.GetValueByParamId(paramId['t'],1000)
    if tim < -990: break
    ztmp.append(val)
  grib.CloseFile()
  grib2.CloseFile()
  minlen = min(map(len,([uuu,vvv,tmp,hum,prs,ztmp])))
  maxlen = max(map(len,([uuu,vvv,tmp,hum,prs,ztmp])))
  if minlen != maxlen: return None
  data['u'] = np.transpose(np.asarray(uuu))
  data['v'] = np.transpose(np.asarray(vvv))
  data['t'] = np.transpose(np.asarray(tmp))
  data['r'] = np.transpose(np.asarray(hum))
  data['p'] = np.transpose(np.asarray(prs))
  data['zt'] = np.transpose(np.asarray(ztmp))
  data['h'] = np.transpose(np.asarray(hgt))
  return data, solution

def dfs_shrt_gdps_nppm_job(conf, data, gis_data, sol, nftim=30):
  '''
    conf      : [dict] configure data
    data      : [dict] variable data (u, v, t, r, p)
    gis_data  : [array] 5km gis data
    sol       : [str] solution (N128, N768, N512)
  '''
  ter_5km_dfs = gis_data
  ter_um = np.where(data['h'] < 0.1,0.0,data['h'])
  landsea = ReadLandsea(conf['landsea'],conf['nx'],conf['ny']).getData()
  landsea[66,161] = 1.0

  temp = data['t']
  rhum = data['r']
  uuuu = data['u']
  vvvv = data['v']
  pres = data['p']
  zt   = data['zt']

  ### Init Projection
  pp = F_PROJ(conf['slat1'],conf['slat2'],conf['olat'],conf['olon'],conf['xo'],conf['yo'],conf['dd'],
              conf['pi'],conf['r'] )
  ter_dfs = pp.um2dfs(conf['nx'],conf['ny'],1,ter_um,conf[sol]['lat_um'],conf[sol]['lon_um'],conf[sol]['dx'],conf[sol]['dy'])
  t2m_dfs = pp.um2dfs_land(conf['nx'],conf['ny'],nftim,temp,conf[sol]['lat_um'],conf[sol]['lon_um'],conf[sol]['dx'],conf[sol]['dy'],ter_um,landsea,1)
  h2m_dfs = pp.um2dfs_land(conf['nx'],conf['ny'],nftim,rhum,conf[sol]['lat_um'],conf[sol]['lon_um'],conf[sol]['dx'],conf[sol]['dy'],ter_um,landsea,1)
  uuu_dfs = pp.um2dfs(conf['nx'],conf['ny'],nftim,uuuu,conf[sol]['lat_um'],conf[sol]['lon_um'],conf[sol]['dx'],conf[sol]['dy'])
  vvv_dfs = pp.um2dfs(conf['nx'],conf['ny'],nftim,vvvv,conf[sol]['lat_um'],conf[sol]['lon_um'],conf[sol]['dx'],conf[sol]['dy'])
  prs_dfs = pp.um2dfs(conf['nx'],conf['ny'],nftim,pres,conf[sol]['lat_um'],conf[sol]['lon_um'],conf[sol]['dx'],conf[sol]['dy'])
  zt_dfs = pp.um2dfs(conf['nx'],conf['ny'],nftim,zt[:,:,:],conf[sol]['lat_um'],conf[sol]['lon_um'],conf[sol]['dx'],conf[sol]['dy'])

  ter_dfs = np.where(ter_dfs < 0.1, 0.0, ter_dfs)

  ### Diagnose
  t3h = diagnose_3h_temperature(t2m_dfs, ter_dfs[:,:,0], ter_5km_dfs, conf[sol]['lapse_rate'], conf['kel2cel'])
  prs = diagnose_pressure(prs_dfs, zt_dfs, ter_dfs[:,:,0], ter_5km_dfs)
  uuu = uuu_dfs
  vvv = vvv_dfs
  rh  = h2m_dfs
  ### Check meteorological range
  t3h = cntl_element_value(t3h,'T3H')
  uuu = cntl_element_value(uuu,'UUU')
  vvv = cntl_element_value(vvv,'VVV')
  rh  = cntl_element_value(rh, 'REH')

  res = np.zeros((5, conf['nx'], conf['ny'], nftim))
  res[0,:,:,:] = prs[:,:,:]
  res[1,:,:,:] = uuu[:,:,:]
  res[2,:,:,:] = vvv[:,:,:]
  res[3,:,:,:] = rh[:,:,:]
  res[4,:,:,:] = t3h[:,:,:]
  return res

def MakeXData(conf_projs, dtime, ipath, gis_data):
  '''
    conf_projs: [dict] configure data for projection
    dtime     : [datetime] Datetime of the dataset you want to create
    ipath     : [str] path where file is exists
    gis_data  : [array] 5km gis data
  '''
  paramId = { 'ncpcp' : 260009, 'snol' : 260012,  'uwind' : 165, 'vwind': 166,
              't' : 130, 'r' : 157, 'vis' : 3020, 'acpcp' : 3063, 'snoc': 260011,
              'mslp' : 260074, 'h' : 3008}

  data, solution = ExtractValueFromGrib(dtime, paramId, ipath)
  if data is None: return None
  data = dfs_shrt_gdps_nppm_job(conf_projs, data, gis_data, solution)
  return data


def MakeDataset(conf_inputs, conf_projs, gis_data, utc, var, intv_hour=24):
  fmt = "%Y%m%d%H"
  #=== Config
  xipath     = conf_inputs['dnn_input_dir']
  nx         = conf_projs['nx']
  ny         = conf_projs['ny']
  sdate      = conf_inputs['sdate']
  edate      = conf_inputs['edate']
  #=== Read Data
  print("Make Dataset : %s - %s\n" %(sdate,edate))
  dt_sdate = datetime.strptime(sdate+str(utc).zfill(2),fmt)  ### str -> datetime
  dt_edate = datetime.strptime(edate+str(utc).zfill(2),fmt)  ### str -> datetime
  if dt_sdate > dt_edate:
    print("Check Input date : %s, %s" %(sdate, edate))
    sys.exit(1)
  cur = dt_sdate
  xdata, ydata, datelist = [], [], []
  while cur <= dt_edate:
    str_cur = datetime.strftime(cur,fmt)
    xdat = MakeXData(conf_projs, cur, xipath, gis_data)
    if xdat is None:
      cur += timedelta(hours=intv_hour)
      continue
    datelist.append(str_cur)
    xdata.append(xdat)
    cur += timedelta(hours=intv_hour)
  xdata = np.asarray(xdata)
  return xdata, datelist
def FileExists(path):
  if not os.path.exists(path):
    print("Can't Find : %s" %(path))
    return False
  else:
    return True
