# %%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline

# %%
from statsmodels.tsa.seasonal import STL

# %%
from os import listdir

# %%
import fathon
from fathon import fathonUtils as fu

# %%
from scipy.stats import kendalltau

# %%
import netCDF4 as nc
import tifffile as tf

# %%
nb_mask_path = 'D:/GEE学习/EWS空间数据处理/NB针叶林掩膜/nb_evimask1.tif'
# 读取tiff文件
nb_mask = tf.imread(nb_mask_path)
print(nb_mask.shape)
nb_mask_path2 = 'D:/GEE学习/EWS空间数据处理/NB针叶林掩膜/nb_boreal.tif'
# 读取tiff文件
nb_mask2 = tf.imread(nb_mask_path2)
print(nb_mask2.shape)

# %%
nb_mask_array = np.array(nb_mask)
nb_mask_array = np.where(nb_mask_array,nb_mask_array,np.nan)
nb_mask_array[np.where(nb_mask_array==3)] = np.nan

# %%
nb_mask_array2 = np.array(nb_mask2)
nb_mask_array2 = np.where(nb_mask_array2,nb_mask_array2,np.nan)
nb_mask_array2[np.where(nb_mask_array2==3)] = np.nan

# %%
plt.imshow(nb_mask_array2)

# %%
nb_ndvi_mean2 = np.zeros(468)
# 路径
path = 'D:/GEE学习/EWS空间数据处理/NOAA-nb-rename/'
files = listdir(path) # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
num = 0
for i in files:
    temp_input = path+i
    # 读取tiff文件
    nb_ndvi = tf.imread(temp_input)
    nb_ndvi_array = np.array(nb_ndvi)
    nb_ndvi_masked = nb_mask_array2*nb_ndvi_array
    nb_ndvi_mean2[num] = np.nanmean(nb_ndvi_masked)
    num = num+1

# %%
nb_ndvi_mean = np.zeros(468)
# 路径
path = 'D:/GEE学习/EWS空间数据处理/NOAA-zyl-rename/'
files = listdir(path) # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
num = 0
for i in files:
    temp_input = path+i
    # 读取tiff文件
    nb_ndvi = tf.imread(temp_input)
    nb_ndvi_array = np.array(nb_ndvi)
    nb_ndvi_masked = nb_mask_array*nb_ndvi_array
    nb_ndvi_mean[num] = np.nanmean(nb_ndvi_masked)
    num = num+1

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(nb_ndvi_mean2,'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='NDVI',
       title='NDVI in 1982-2020')
ax1.grid()

plt.show()

# %%
# 使用mask可以给nan加回来
mask = np.isnan(nb_ndvi_mean)
nb_ndvi_mean[mask] = np.nanmean(nb_ndvi_mean)
mask2 = np.isnan(nb_ndvi_mean2)
nb_ndvi_mean2[mask2] = np.nanmean(nb_ndvi_mean2)
#am_ndvi_monthly_mask = am_ndvi_monthly[~mask]

# %%
res_nb_ndvi = STL(nb_ndvi_mean, period=12).fit()
res_nb_ndvi.plot()
plt.show()
res_nb_ndvi_resid = res_nb_ndvi.resid

# %%
res_nb_ndvi2 = STL(nb_ndvi_mean2, period=12).fit()
res_nb_ndvi2.plot()
plt.show()
res_nb_ndvi_resid2 = res_nb_ndvi2.resid

# %%
nb_ndvi_DFA_alpha = np.zeros(348)
for i in range(348):
    temp = 60+i
    pydfa = fathon.DFA(res_nb_ndvi_resid[i:temp])
    winSizes = fu.linRangeByStep(10,60)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    nb_ndvi_DFA_alpha[i] = H
    
nb_ndvi_DFA_alpha2 = np.zeros(348)
for i in range(348):
    temp = 120+i
    pydfa = fathon.DFA(res_nb_ndvi_resid2[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    nb_ndvi_DFA_alpha2[i] = H

# %%
np.save('ca_ndvi_DFA_alpha2.npy',nb_ndvi_DFA_alpha)

# %%
ca_ndvi_DFA_alpha = np.load('ca_ndvi_DFA_alpha.npy') 

# %%
date1 = pd.date_range('19820101','20210101',freq='M')
date2 = pd.date_range('19920101','20210101',freq='M')

# %%
date_evi = pd.date_range('20110101','20240101',freq='M')
date_vod = pd.date_range('19970801','20180101',freq='M')

# %%
ca_evi_DFA_alpha = np.load('nb_evi_DFA_alpha.npy') 
ca_vod_DFA_alpha = np.load('nb_vod_DFA_alpha.npy')
nb_ndvi_DFA_alpha = np.load('nb_ndvi_DFA_alpha.npy')

# %%
# 输出为图像
xlabel = date2
fig = plt.figure(figsize=(12,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
#ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
ax1.plot(xlabel,nb_ndvi_DFA_alpha,'royalblue',linewidth=1)
#ax2.plot(xlabel[0:336],ca_ndvi_DFA_alpha,'royalblue',linewidth=1)
ax1.plot(xlabel,nb_ndvi_DFA_alpha2,c='indianred')
ax1.set(xlabel='time series (monthly)', ylabel='DFA',
       title='NDVI DFA in 1992-2021 10 year window')
ax1.set_xlim(date1[0],date1[467])
ax1.legend(['blue area','red area'],loc='upper left',shadow=True,fancybox=True)
#ax1.set_ylim(-0.05,0.6)
ax1.grid()

plt.show()

# %%
# Data for plotting
xlabel = date2
fig = plt.figure(figsize=(10,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
#ax1.plot(xlabel[0:36],am_ndvi_DFA_alpha[0:36],ls='--',c='lightgray')
ax1.plot(xlabel[0:36],nb_ndvi_DFA_alpha[0:36],ls='--',c='royalblue')
ax1.plot(xlabel[36:348],nb_ndvi_DFA_alpha[36:348],c='indianred')
ax2.plot(date_evi,ca_evi_DFA_alpha,c='dimgray')
ax2.plot(date_vod,ca_vod_DFA_alpha,c='darkgray')
#ax1.plot(xlabel[290:345],am_ndvi_DFA_alpha[290:345],ls='--',c='#000000')
ax1.set(xlabel='time series', ylabel='DFA')
ax1.set_xlim(date1[0],date1[467])
ax1.axvline(x='19950101',color='black')
ax1.legend(['NDVI before tipping','NDVI after tipping'],loc='upper left',shadow=True,fancybox=True)
ax2.legend(['EVI','VOD'],loc='upper right',shadow=True,fancybox=True)
ax1.grid()

plt.show()

# %%
kendalltau(ca_ndvi_DFA_alpha[0:120],ca_ndvi_DFA_alpha[120:240])

# %%
cor_nb_ndvi = np.zeros(len(res_nb_ndvi_resid)-120)
for i in range(len(res_nb_ndvi_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_nb_ndvi[i] = np.corrcoef(res_nb_ndvi_resid[month_start:month_end-1],res_nb_ndvi_resid[month_start+1:month_end])[0,1]

# %%
cor_nb_ndvi2 = np.zeros(len(res_nb_ndvi_resid2)-120)
for i in range(len(res_nb_ndvi_resid2)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_nb_ndvi2[i] = np.corrcoef(res_nb_ndvi_resid2[month_start:month_end-1],res_nb_ndvi_resid2[month_start+1:month_end])[0,1]

# %%
cor_nb_evi = np.load('cor_nb_evi.npy') 
cor_nb_vod = np.load('cor_nb_vod.npy')

# %%
# 输出为图像
xlabel = date2
fig = plt.figure(figsize=(12,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
#ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
ax1.plot(xlabel,cor_nb_ndvi,'royalblue',linewidth=1)
ax1.plot(xlabel,cor_nb_ndvi2,'indianred',linewidth=1)
#ax2.plot(xlabel[0:336],ca_ndvi_DFA_alpha,'royalblue',linewidth=1)
#ax2.plot(date_vod,ca_vod_DFA_alpha,c='darkgray')
ax1.set(xlabel='time series (monthly)', ylabel='AR1',
       title='NDVI DFA in 1992-2021 10 year window')
ax1.set_xlim(date1[0],date1[467])
ax1.legend(['blue area','red area'],loc='upper left',shadow=True,fancybox=True)
#ax1.set_ylim(-0.05,0.6)
ax1.grid()

plt.show()

# %%
# Data for plotting
xlabel = date2
fig = plt.figure(figsize=(10,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
#ax1.plot(xlabel[0:36],am_ndvi_DFA_alpha[0:36],ls='--',c='lightgray')
#ax1.plot(xlabel[0:96],cor_nb_ndvi[0:96],ls='--',c='royalblue')
ax1.plot(xlabel[0:348],cor_nb_ndvi[0:348],c='indianred')
ax2.plot(date_evi,cor_nb_evi,c='dimgray')
ax2.plot(date_vod,cor_nb_vod,c='darkgray')
#ax1.plot(xlabel[290:345],am_ndvi_DFA_alpha[290:345],ls='--',c='#000000')
ax1.set(xlabel='time series', ylabel='AR1')
ax1.set_xlim(date1[0],date1[467])
ax1.axvline(x='19920101',color='black')
ax1.legend(['NDVI after tipping'],loc='upper left',shadow=True,fancybox=True)
ax2.legend(['EVI','VOD'],loc='upper right',shadow=True,fancybox=True)
ax1.grid()

plt.show()

# %%
kendalltau(cor_nb_ndvi[0:96],cor_nb_ndvi[96:192])

# %%
var_ca_ndvi = np.zeros(len(res_ca_ndvi_resid)-120)
for i in range(len(res_ca_ndvi_resid)-120):
    temp = 120+i
    var_ca_ndvi[i] = np.var(res_ca_ndvi_resid[i:temp]*100)

# %%
np.save('cor_nb_ndvi.npy',cor_nb_ndvi)

# %%
# Data for plotting
xlabel = date2
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,var_ca_ndvi,'#000000',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='VAR',
       title='VAR(ndvi in ca) in 1981-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.axvline(x='19920101',color='r')
ax1.grid()
plt.show()

# %%
kendalltau(var_ca_ndvi[0:120],var_ca_ndvi[120:240])

# %%
