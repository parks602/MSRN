# -*- coding: utf-8 -*-  
########################################################################
### Author      : KOAST 
### Date        : 2019.5.17
### Description : Lamber Conformal Conic Projection.
###               It is written by referring to the KMA code in Fortran.
### Update info : 
########################################################################
import math
import sys
import numpy as np
import fortran_proj as fp # lamcinit, lamcproj

class F_PROJ:
  def __init__(self, slat1, slat2, olat, olon, xo, yo, dd, PI, R):
    fp.lamcinit(PI,R,slat1,slat2,olat,olon,xo,yo,dd)

  def um2dfs(self, nx, ny, nz, um, lat_um, lon_um, dx, dy):
    ''' Interpolate UM to DFS '''
    if len(um.shape) == 2: um = np.reshape(um,(um.shape[0],um.shape[1],1))
    dfs = np.zeros((nx,ny,nz))

    for i in range(nx):
      for j in range(ny):
        xi = i + 1
        xj = j + 1
        lat, lon = fp.lamcproj(xi,xj)
        ii = int((lon - lon_um) / dx) 
        jj = int((lat - lat_um) / dy) 

        idx_degree = lon_um + ii*dx
        jdx_degree = lat_um + jj*dy
        #print(ii+1,jj+1)

        w00 = abs(idx_degree + dx - lon) * abs(jdx_degree + dy - lat)
        w10 = abs(lon - idx_degree     ) * abs(jdx_degree + dy - lat)
        w01 = abs(idx_degree + dx - lon) * abs(lat - jdx_degree     )
        w11 = abs(lon - idx_degree     ) * abs(lat - jdx_degree     )
        winv = 1. / (w00 + w01 + w10 + w11)

        #print(ii+1,jj+1,um[ii,jj,0], um[ii+1,jj,0  ], um[ii,jj+1,0], um[ii+1,jj+1,0])
        for k in range(nz):
          dfs[i,j,k] = winv * ( um[ii,jj,k  ] * w00 + um[ii+1,jj,k  ] * w10 + \
                                um[ii,jj+1,k] * w01 + um[ii+1,jj+1,k] * w11 )
    return dfs

  def um2dfs_land(self, nx, ny, nz, um, lat_um, lon_um, dx, dy, ter_um, land_dfs, opt):
    ''' Interpolate UM to DFS 
        opt :: 0 normal OA, 1 OA with only UM terrain and DFS land
    '''
    if len(um.shape) == 2: um = np.reshape(um,(um.shape[0],um.shape[1],1))
    dfs = np.zeros((nx,ny,nz))

    for i in range(nx):
      for j in range(ny):
        xi = i + 1
        xj = j + 1
        lat, lon = fp.lamcproj(xi,xj)
        ii = int((lon - lon_um) / dx)
        jj = int((lat - lat_um) / dy)

        idx_degree = lon_um + ii*dx
        jdx_degree = lat_um + jj*dy
        #print(ii+1,jj+1)

        w00 = abs(idx_degree + dx - lon) * abs(jdx_degree + dy - lat)
        w10 = abs(lon - idx_degree     ) * abs(jdx_degree + dy - lat)
        w01 = abs(idx_degree + dx - lon) * abs(lat - jdx_degree     )
        w11 = abs(lon - idx_degree     ) * abs(lat - jdx_degree     )
        ### land_um -> ter_um
        rate = 0
        for k in range(ii,ii+2):
          for l in range(jj,jj+2):
            if ter_um[k,l] > 0: rate += 1

        if opt==1 and land_dfs[i,j] == 1.:
        ### at leat one grid is ocean
          if rate >= 1 and rate <=3:
            if int(ter_um[ii  ,jj  ]) == 0: w00 = 0.0
            if int(ter_um[ii+1,jj  ]) == 0: w10 = 0.0
            if int(ter_um[ii  ,jj+1]) == 0: w01 = 0.0
            if int(ter_um[ii+1,jj+1]) == 0: w11 = 0.0
          elif rate == 0:
            xx = ((lon - lon_um) / dx)  
            yy = ((lat - lat_um) / dy) 
            i0 = int(round((lon - lon_um) / dx,0))
            j0 = int(round((lat - lat_um) / dy,0))
              
            min_distance = 999.9
            for iii in range(i0-5, i0+6):
              for jjj in range(j0-5, j0+6):
                distance = np.sqrt((xx-iii)**2. + (yy-jjj)**2.)
                if distance < min_distance and ter_um[iii,jjj] > 0: # land -> ter
                  w00, w01, w10, w11 = 1., 0., 0., 0.
                  ii, jj = iii, jjj
                  min_distance = distance

        try:
          winv = 1 / (w00 + w01 + w10 + w11)
        except ZeroDivisionError as e:
          w00 = abs(idx_degree + dx - lon) * abs(jdx_degree + dy - lat)
          w10 = abs(lon - idx_degree     ) * abs(jdx_degree + dy - lat)
          w01 = abs(idx_degree + dx - lon) * abs(lat - jdx_degree     )
          w11 = abs(lon - idx_degree     ) * abs(lat - jdx_degree     )
          winv = 1 / (w00 + w01 + w10 + w11)
           
        for k in range(nz):
          dfs[i,j,k] = winv * ( um[ii,jj,k  ] * w00 + um[ii+1,jj,k  ] * w10 + \
                                um[ii,jj+1,k] * w01 + um[ii+1,jj+1,k] * w11 )
    return dfs
