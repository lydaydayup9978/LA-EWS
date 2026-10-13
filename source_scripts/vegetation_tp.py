# %%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import fathon
from fathon import fathonUtils as fu
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline
from statsmodels.tsa.seasonal import STL
from scipy.stats import kendalltau

# %%
from osgeo import gdal
from os import listdir

# %%
# ndvi数据路径
tp_ndvi_path = 'D:/GEE学习/EWS空间数据处理/NOAA-tp-rename/'
# 掩膜路径
tpfe_tifmask_path = 'D:/GEE学习/EWS空间数据处理/TP草地掩膜/tpfe_landcover.tif'
# 掩膜路径
tpgl_tifmask_path = 'D:/GEE学习/EWS空间数据处理/TP草地掩膜/tpgl_landcover.tif'
# 读取tiff文件
tpfe_mask = gdal.Open(tpfe_tifmask_path)
tpfe_maskarray = tpfe_mask.ReadAsArray()
tpfe_maskarray = np.where(tpfe_maskarray,tpfe_maskarray,np.nan)

tpgl_mask = gdal.Open(tpgl_tifmask_path)
tpgl_maskarray = tpgl_mask.ReadAsArray()
tpgl_maskarray = np.where(tpgl_maskarray,tpgl_maskarray,np.nan)

# %%
tpgl_maskarray.shape

# %%
tpfe_ndvi_mean = np.zeros(468)
tpgl_ndvi_mean = np.zeros(468)
# 路径
path = tp_ndvi_path
files = listdir(path) # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
num = 0
for i in files:
    temp_input = path+i
    # 读取tiff文件
    tp_ndvi = gdal.Open(temp_input)
    tp_ndvi_array = tp_ndvi.ReadAsArray()
    tpgl_ndvi_masked = tpgl_maskarray*tp_ndvi_array
    tpfe_ndvi_masked = tpfe_maskarray*tp_ndvi_array
    tpgl_ndvi_mean[num] = np.nanmean(tpgl_ndvi_masked)
    tpfe_ndvi_mean[num] = np.nanmean(tpfe_ndvi_masked)
    
    num = num+1

# %%
# 使用mask可以给nan加回来
mask = np.isnan(tpgl_ndvi_mean)
tpgl_ndvi_mean[mask] = np.nanmean(tpgl_ndvi_mean)
mask2 = np.isnan(tpfe_ndvi_mean)
tpfe_ndvi_mean[mask2] = np.nanmean(tpfe_ndvi_mean)

# %%
res_tpgl_ndvi = STL(tpgl_ndvi_mean, period=12).fit()
res_tpgl_ndvi_resid = res_tpgl_ndvi.resid
res_tpfe_ndvi = STL(tpfe_ndvi_mean, period=12).fit()
res_tpfe_ndvi_resid = res_tpfe_ndvi.resid

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(res_tpgl_ndvi_resid,'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='NDVI res',
       title='NDVI res in 1982-2019')
#ax1.set_ylim(-0.05,0.6)
ax1.grid()

plt.show()

# %%
tpgl_ndvi_DFA_alpha = np.zeros(348)
for i in range(348):
    temp = 120+i
    pydfa = fathon.DFA(res_tpgl_ndvi_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    tpgl_ndvi_DFA_alpha[i] = H
    
tpfe_ndvi_DFA_alpha = np.zeros(348)
for i in range(348):
    temp = 120+i
    pydfa = fathon.DFA(res_tpfe_ndvi_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    tpfe_ndvi_DFA_alpha[i] = H

# %%
date1 = pd.date_range('19920101','20210101',freq='M')
date2 = pd.date_range('19820101','20210101',freq='M')

# %%
# Data for plotting
xlabel = date1
fig = plt.figure(figsize=(10,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,tpgl_ndvi_DFA_alpha,c='#dade48',label='grassland')
ax1.plot(xlabel,tpfe_ndvi_DFA_alpha,c='#05450a',label='forest')
ax1.set(xlabel='time series (monthly)', ylabel='DFA',
       title='TP DFA in 1992-2021 10 year window')
ax1.grid()
ax1.legend()
ax1.set_xlim(date2[0],date2[467])

plt.show()

# %%
cor_tpgl_ndvi = np.zeros(len(res_tpgl_ndvi_resid)-120)
for i in range(len(res_tpgl_ndvi_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_tpgl_ndvi[i] = np.corrcoef(res_tpgl_ndvi_resid[month_start:month_end-1],res_tpgl_ndvi_resid[month_start+1:month_end])[0,1]
    
cor_tpfe_ndvi = np.zeros(len(res_tpfe_ndvi_resid)-120)
for i in range(len(res_tpfe_ndvi_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_tpfe_ndvi[i] = np.corrcoef(res_tpfe_ndvi_resid[month_start:month_end-1],res_tpfe_ndvi_resid[month_start+1:month_end])[0,1]

# %%
# Data for plotting
xlabel = date1
fig = plt.figure(figsize=(10,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,cor_tpgl_ndvi,c='#dade48',label='grassland')
ax1.plot(xlabel,cor_tpfe_ndvi,c='#05450a',label='forest')
ax1.set(xlabel='time series (monthly)', ylabel='AR1',
       title='TP AR1 in 1992-2021 10 year window')
ax1.grid()
ax1.legend()
ax1.set_xlim(date2[0],date2[467])

plt.show()

# %%
np.save('cor_tpgl_ndvi.npy',cor_tpgl_ndvi)
np.save('tpgl_ndvi_DFA_alpha.npy',tpgl_ndvi_DFA_alpha)

# %%

