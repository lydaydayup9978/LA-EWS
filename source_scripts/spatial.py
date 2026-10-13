# %%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import netCDF4 as nc
from fathon import fathonUtils as fu
from scipy.stats import kendalltau
from statsmodels.tsa.seasonal import STL
import fathon

# %%
from osgeo import gdal
from os import listdir

# %%
from tqdm import tqdm
import time

# %%
am_mask_path = 'D:/GEE学习/临界要素数据处理/研究区域掩膜tiff/am_mask.tif'
am_mask_raster = gdal.Open(am_mask_path)
am_mask_array = am_mask_raster.ReadAsArray()
am_mask_array = np.where(am_mask_array,am_mask_array,np.nan)

# %%
x = range(624)

# %%
# 一共62年的数据
# 存储62年的数据
am_pai_62 = np.zeros([744,301,401])
for i in range(62):
    # 导入am地区陆气耦合数据
    # 陆气耦合为三维数据npy格式文件
    am_pai_path = 'F:/ERA5/陆气耦合重算/am_pai/'
    year_num = i+1960
    year = str(year_num)
    # 打开npy文件
    data3 = np.load(am_pai_path+ year + '.npy')
    am_pai = data3
    am_pai[np.where(am_pai==-32767)] = np.nan
    # 当场转化为月平均数据
    monthly = np.zeros([12,301,401])
    for k in range(12):
        if (k+1)*30<len(am_pai):
            monthly[k,:,:] = np.nanmean(am_pai[k*30:(k+1)*30,:,:],0)
        else:
            monthly[k,:,:] = np.nanmean(am_pai[k*30:len(am_paimean_array)-1,:,:])
    am_pai_62[i*12:(i+1)*12,:,:] = monthly

# %%
am_pai_62 = am_pai_62*am_mask_array

# %%
am_raw = 301
am_col = 401
num = 744
am_pai_tau = np.zeros([am_raw,am_col])
am_pai_p = np.zeros([am_raw,am_col])
error_pixel = 0
# 初始化AR1存储矩阵
AR1_am = np.zeros([am_raw,am_col])
for i in tqdm(range(am_raw)):
    for j in range(am_col):
        temp_pai = am_pai_62[:,i,j]
        # 判断向量是否为nan，如果是则跳过
        if np.isnan(temp_pai).all():
            continue
        # 对向量的nan值进行掩膜
        mask = np.isnan(temp_pai)
        temp_pai_mask = temp_pai[~mask]
        # 掩膜之后，判断向量长度，如果长度小于720（24个月空缺），则该条数据舍去
        if len(temp_pai_mask)<720:
            error_pixel = error_pixel+1
            continue
        # 对向量进行季节性分解
        res_temp_pai = STL(temp_pai_mask, period=12).fit()
        res_temp_pai_resid = res_temp_pai.resid
        # 以下为DFA的H值计算
        cor_temp_pai = np.zeros(len(res_temp_pai_resid)-120)
        for n in range(len(res_temp_pai_resid)-120):
            temp = 120+n
            month_start = n
            month_end = temp
            cor_temp_pai[n] = np.corrcoef(res_temp_pai_resid[month_start:month_end-1],res_temp_pai_resid[month_start+1:month_end])[0,1]
        tau = np.zeros([2,1])
        for k in range(1):
            tau[:,k] = kendalltau(x[528+k:624+k],cor_temp_pai[528+k:624+k])
        tau_min = np.min(tau[0,:])
        p_min = tau[1,np.where(tau[0,:]==tau_min)]
        am_pai_tau[i,j] = tau_min
        am_pai_p[i,j] = np.min(p_min)
    time.sleep(0.01)



# %%
np.save('am_pai_ar1tau_future.npy',am_pai_tau)
np.save('am_pai_ar1p_future.npy',am_pai_p)

# %%
plt.imshow(am_pai_tau*am_mask_array)

# %%
# 一共62年的数据
# 存储62年的数据
am_tet_62 = np.zeros([624,301,401])
for i in range(52):
    # 导入am地区陆气耦合数据
    # 陆气耦合为三维数据nc格式文件
    #am_tet_path = 'F:/ERA5/landatmos_coupl/am_tet/'
    am_tet_path = 'F:/ERA5/陆气耦合重算/am_tet/'
    year_num = i+1970
    year = str(year_num)
    # 打开npy文件
    data3 = np.load(am_tet_path+ year + '.npy')
    am_tet = data3
    # 求均值（数据本身为月平均数据）
    am_tet_62[i*12:(i+1)*12] = am_tet

# %%
am_tet_62 = am_tet_62*am_mask_array

# %%
am_raw = 301
am_col = 401
num = 624
am_pai_tau = np.zeros([am_raw,am_col])
am_pai_p = np.zeros([am_raw,am_col])
error_pixel = 0
# 初始化AR1存储矩阵
AR1_am = np.zeros([am_raw,am_col])
for i in tqdm(range(am_raw)):
    for j in range(am_col):
        temp_pai = am_tet_62[:,i,j]
        # 判断向量是否为nan，如果是则跳过
        if np.isnan(temp_pai).all():
            continue
        # 对向量的nan值进行掩膜
        mask = np.isnan(temp_pai)
        temp_pai_mask = temp_pai[~mask]
        # 掩膜之后，判断向量长度，如果长度小于720（24个月空缺），则该条数据舍去
        if len(temp_pai_mask)<600:
            error_pixel = error_pixel+1
            continue
        # 对向量进行季节性分解
        res_temp_pai = STL(temp_pai_mask, period=12).fit()
        res_temp_pai_resid = res_temp_pai.resid
        # 以下为DFA的H值计算
        cor_temp_pai = np.zeros(len(res_temp_pai_resid)-120)
        for n in range(len(res_temp_pai_resid)-120):
            temp = 120+n
            month_start = n
            month_end = temp
            cor_temp_pai[n] = np.corrcoef(res_temp_pai_resid[month_start:month_end-1],res_temp_pai_resid[month_start+1:month_end])[0,1]
        tau = np.zeros([2,1])
        for k in range(1):
            tau[:,k] = kendalltau(x[408+k:504+k],cor_temp_pai[408+k:504+k])
        tau_min = np.min(tau[0,:])
        p_min = tau[1,np.where(tau[0,:]==tau_min)]
        am_pai_tau[i,j] = tau_min
        am_pai_p[i,j] = np.min(p_min)
    time.sleep(0.01)



# %%
np.save('am_tet_ar1tau_future.npy',am_pai_tau*am_mask_array)
np.save('am_tet_ar1p_future.npy',am_pai_p*am_mask_array)

# %%
am_pai_tau

# %%
# 一共62年的数据
# 存储62年的数据
am_tci_62 = np.zeros([624,301,401])
for i in range(52):
    # 导入am地区陆气耦合数据
    # 陆气耦合为三维数据nc格式文件
    #am_tet_path = 'F:/ERA5/landatmos_coupl/am_tet/'
    am_tci_path = 'F:/ERA5/陆气耦合重算/am_tci/'
    year_num = i+1970
    year = str(year_num)
    # 打开npy文件
    data3 = np.load(am_tci_path+ year + '.npy')
    am_tci = data3
    # 求均值（数据本身为月平均数据）
    am_tci_62[i*12:(i+1)*12] = am_tci

# %%
am_tci_62 = am_tci_62*am_mask_array

# %%
am_raw = 301
am_col = 401
num = 624
am_pai_tau = np.zeros([am_raw,am_col])
am_pai_p = np.zeros([am_raw,am_col])
error_pixel = 0
# 初始化AR1存储矩阵
AR1_am = np.zeros([am_raw,am_col])
for i in tqdm(range(am_raw)):
    for j in range(am_col):
        temp_pai = am_tci_62[:,i,j]
        # 判断向量是否为nan，如果是则跳过
        if np.isnan(temp_pai).all():
            continue
        # 对向量的nan值进行掩膜
        mask = np.isnan(temp_pai)
        temp_pai_mask = temp_pai[~mask]
        # 掩膜之后，判断向量长度，如果长度小于720（24个月空缺），则该条数据舍去
        if len(temp_pai_mask)<600:
            error_pixel = error_pixel+1
            continue
        # 对向量进行季节性分解
        res_temp_pai = STL(temp_pai_mask, period=12).fit()
        res_temp_pai_resid = res_temp_pai.resid
        # 以下为DFA的H值计算
        cor_temp_pai = np.zeros(len(res_temp_pai_resid)-120)
        for n in range(len(res_temp_pai_resid)-120):
            temp = 120+n
            month_start = n
            month_end = temp
            cor_temp_pai[n] = np.corrcoef(res_temp_pai_resid[month_start:month_end-1],res_temp_pai_resid[month_start+1:month_end])[0,1]
        tau = np.zeros([2,1])
        for k in range(1):
            tau[:,k] = kendalltau(x[408+k:504+k],cor_temp_pai[408+k:504+k])
        tau_min = np.min(tau[0,:])
        p_min = tau[1,np.where(tau[0,:]==tau_min)]
        am_pai_tau[i,j] = tau_min
        am_pai_p[i,j] = np.min(p_min)
    time.sleep(0.01)

# %%
np.save('am_tci_ar1tau_future.npy',am_pai_tau*am_mask_array)
np.save('am_tci_ar1p_future.npy',am_pai_p*am_mask_array)

# %%
am_la = (am_tci_62+am_tet_62+am_pai_62[120:744,:,:])/3

# %%
am_raw = 301
am_col = 401
num = 624
am_pai_tau = np.zeros([am_raw,am_col])
am_pai_p = np.zeros([am_raw,am_col])
error_pixel = 0
# 初始化AR1存储矩阵
AR1_am = np.zeros([am_raw,am_col])
for i in tqdm(range(am_raw)):
    for j in range(am_col):
        temp_pai = am_la[:,i,j]
        # 判断向量是否为nan，如果是则跳过
        if np.isnan(temp_pai).all():
            continue
        # 对向量的nan值进行掩膜
        mask = np.isnan(temp_pai)
        temp_pai_mask = temp_pai[~mask]
        # 掩膜之后，判断向量长度，如果长度小于720（24个月空缺），则该条数据舍去
        if len(temp_pai_mask)<600:
            error_pixel = error_pixel+1
            continue
        # 对向量进行季节性分解
        res_temp_pai = STL(temp_pai_mask, period=12).fit()
        res_temp_pai_resid = res_temp_pai.resid
        # 以下为DFA的H值计算
        cor_temp_pai = np.zeros(len(res_temp_pai_resid)-120)
        for n in range(len(res_temp_pai_resid)-120):
            temp = 120+n
            month_start = n
            month_end = temp
            cor_temp_pai[n] = np.corrcoef(res_temp_pai_resid[month_start:month_end-1],res_temp_pai_resid[month_start+1:month_end])[0,1]
        tau = np.zeros([2,1])
        for k in range(1):
            tau[:,k] = kendalltau(x[378+k:504+k],cor_temp_pai[378+k:504+k])
        tau_min = np.min(tau[0,:])
        p_min = tau[1,np.where(tau[0,:]==tau_min)]
        am_pai_tau[i,j] = tau_min
        am_pai_p[i,j] = np.min(p_min)
    time.sleep(0.01)

# %%
np.save('am_la_ar1tau_0502_late.npy',am_pai_tau*am_mask_array)
np.save('am_la_ar1p_0502_late.npy',am_pai_p*am_mask_array)

# %%

