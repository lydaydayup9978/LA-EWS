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
import netCDF4 as nc
import tifffile as tf

# %%
# 导入npy格式数据
am_tci_path = 'D:/GEE学习/陆气耦合计算/am_tci/'
am_tet_path = 'D:/GEE学习/陆气耦合计算/am_tet/'
year_num = 1970
year = str(year_num)
# 打开npy文件
am_tci = np.load(am_tci_path+ year + '.npy')
am_tet = np.load(am_tet_path+ year + '.npy')

# %%
np.shape(am_tci)

# %%
am_mask_path = 'D:/GEE学习/临界要素数据处理/研究区域掩膜shp/am_mask_ndvi025_region4.tif'
# 读取tiff文件
am_mask = tf.imread(am_mask_path)
print(am_mask.shape)

# %%
am_mask_array2 = np.array(am_mask)
am_mask_array = am_mask_array2[::-1, :]

# %%
temp = am_tci[7,:,:]
plt.imshow(temp)

# %%
temp = am_tet[5,:,:]
plt.imshow(temp[::-1, :])

# %%
plt.imshow(am_mask_array)

# %%
am_mask_array = np.where(am_mask_array,am_mask_array,np.nan)
am_mask_array2 = np.where(am_mask_array2,am_mask_array2,np.nan)

# %%
plt.imshow(am_mask_array2)

# %%
date1 = pd.date_range('19600101','20220101',freq='M')
date2 = pd.date_range('19700101','20220101',freq='M')
date3 = pd.date_range('19800101','20081001',freq='M')
date31 = pd.date_range('19920101','20201001',freq='M')
date4 = pd.date_range('19800101','20220101',freq='M')
date6 = pd.date_range('19920101','20190301',freq='M')
date5 = pd.date_range('19800601','20070801',freq='M')
date51 = pd.date_range('19910701','20180901',freq='M')

# %%
# 一共62年的数据
# 存储62年的数据
am_tci_62 = np.zeros(624)
for i in range(52):
    # 导入pe地区积雪深度数据
    # 积雪深度为三维数据nc格式文件
    #am_tci_path = 'F:/ERA5/landatmos_coupl/am_TCI/'
    am_tci_path = 'F:/ERA5/陆气耦合重算/am_TCI/'
    #am_tci_path = 'D:/GEE学习/陆气耦合计算/am_TCI/'
    year_num = i+1970
    year = str(year_num)
    # 打开npy文件
    data2 = np.load(am_tci_path+ year + '.npy')
    # mask与数据矩阵对应相乘
    am_tci = data2*am_mask_array2
    # 求均值（数据本身为月平均数据）
    am_tcimean_array = np.zeros(len(data2[:,1,1]))
    yearnum = len(data2[:,1,1])
    for j in range(yearnum):
        am_tcimean_array[j] = np.nanmean(am_tci[j,:,:])
    am_tci_62[i*12:(i+1)*12] = am_tcimean_array

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(am_tci_62,'#000000',linewidth=1,label='am_tci')
ax1.set(xlabel='time series (month)', ylabel='TCI',
       title='AM TCI in 1960-2021')
#ax1.set_ylim(-0.05,0.6)
ax1.legend()
ax1.grid()

plt.show()

# %%
res_am_tci = STL(am_tci_62, period=12,robust=True).fit()
res_am_tci.plot()
plt.show()

# %%
res_am_tci_resid = res_am_tci.resid
#res_am_tci_resid = am_tci_62

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(res_am_tci_resid,'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='TCI',
       title='AM TCI in 1960-2021')
#ax1.set_ylim(-0.05,0.6)
ax1.grid()

plt.show()

# %%
import fathon
from fathon import fathonUtils as fu

# %%
res_am_tci_resid_alpha = np.zeros(504)
for i in range(504):
    temp = 120+i
    pydfa = fathon.DFA(res_am_tci_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    res_am_tci_resid_alpha[i] = H

# %%
am_ndvi_DFA_alpha = np.load('am_ndvi_DFA_alpha.npy')

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,res_am_tci_resid_alpha,'#000000')
ax1.plot(date5,am_ndvi_DFA_alpha,'#DAA520')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='am_TCI_DFA in 1960-2021')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()

plt.show()

# %%
var_am_tci = np.zeros(len(res_am_tci_resid)-120)
for i in range(len(res_am_tci_resid)-120):
    temp = 120+i
    var_am_tci[i] = np.var(res_am_tci_resid[i:temp]*100)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,var_am_tci,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='VAR',
       title='VAR(TCI in am) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
import statsmodels.tsa.stattools as stattools
from statsmodels.graphics.tsaplots import plot_acf

# %%
cor_am_tci2 = np.zeros(len(res_am_tci_resid)-120)
for i in range(len(res_am_tci_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_am_tci2[i] = np.corrcoef(res_am_tci_resid[month_start:month_end-1],res_am_tci_resid[month_start+1:month_end])[0,1]

# %%
cor_am_tci = np.zeros(len(res_am_tci_resid)-120)
for i in range(len(res_am_tci_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_am_tci[i] = np.corrcoef(res_am_tci_resid[month_start:month_end-1],res_am_tci_resid[month_start+1:month_end])[0,1]

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,cor_am_tci,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='AR1',
       title='AR1(TCI in am) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
# 一共62年的数据
# 存储62年的数据
am_gamma_62 = np.zeros(624)
for i in range(52):
    # 导入am地区陆气耦合数据
    # 陆气耦合为三维数据nc格式文件
    #am_gamma_path = 'D:/GEE学习/陆气耦合计算/am_gamma/'
    am_gamma_path = 'F:/ERA5/陆气耦合重算/am_gamma/'
    year_num = i+1970
    year = str(year_num)
    # 打开npy文件
    data3 = np.load(am_gamma_path+ year + '.npy')
    # mask与数据矩阵对应相乘
    am_gamma = data3*am_mask_array2
    # 求均值（数据本身为月平均数据）
    am_gammamean_array = np.zeros(len(data3[:,1,1]))
    yearnum = len(data3[:,1,1])
    for j in range(yearnum):
        am_gammamean_array[j] = np.nanmean(am_gamma[j,:,:])
    am_gamma_62[i*12:(i+1)*12] = am_gammamean_array

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(am_gamma_62,'#100000',linewidth=1,label='am_gamma')
ax1.set(xlabel='time series (month)', ylabel='Gamma',
       title='AM Gamma in 1960-2021')
#ax1.set_ylim(-0.05,0.6)
ax1.legend()
ax1.grid()

plt.show()

# %%
res_am_gamma = STL(am_gamma_62, period=12).fit()
res_am_gamma.plot()
plt.show()

# %%
res_am_gamma_resid = res_am_gamma.resid

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(res_am_gamma_resid,'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='gamma',
       title='AM gamma res in 1960-202')
#ax1.set_ylim(-0.05,0.6)
ax1.grid()

plt.show()

# %%
np.min(res_am_gamma_resid)

# %%
res_am_gamma_resid[np.where(res_am_gamma_resid==np.min(res_am_gamma_resid))] = 0

# %%
import fathon
from fathon import fathonUtils as fu

# %%
res_am_gamma_resid_alpha = np.zeros(504)
for i in range(504):
    temp = 120+i
    pydfa = fathon.DFA(res_am_gamma_resid[i:temp])
    #pydfa = fathon.DFA(am_gamma_62[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    res_am_gamma_resid_alpha[i] = H

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,res_am_gamma_resid_alpha,'#000000')
ax1.plot(date5,am_ndvi_DFA_alpha,'#DAA520')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='am_gamma_DFA in 1960-2021')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()

plt.show()

# %%
var_am_gamma = np.zeros(len(res_am_gamma_resid)-120)
for i in range(len(res_am_gamma_resid)-120):
    temp = 120+i
    var_am_gamma[i] = np.var(res_am_gamma_resid[i:temp]*100)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,var_am_gamma,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='VAR',
       title='VAR(gamma in am) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
cor_am_gamma = np.zeros(len(res_am_gamma_resid)-120)
for i in range(len(res_am_gamma_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_am_gamma[i] = np.corrcoef(res_am_gamma_resid[month_start:month_end-1],res_am_gamma_resid[month_start+1:month_end])[0,1]

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,cor_am_gamma,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='AR1',
       title='AR1(gamma in am) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
# 一共62年的数据
# 存储62年的数据
am_twolegged_62 = np.zeros(624)
for i in range(52):
    # 导入am地区陆气耦合数据
    # 陆气耦合为三维数据nc格式文件
    #am_twolegged_path = 'D:/GEE学习/陆气耦合计算/am_twolegged/'
    am_twolegged_path = 'F:/ERA5/陆气耦合重算/am_twolegged/'
    year_num = i+1970
    year = str(year_num)
    # 打开npy文件
    data3 = np.load(am_twolegged_path+ year + '.npy')
    # mask与数据矩阵对应相乘
    am_twolegged = data3*am_mask_array2
    # 求均值（数据本身为月平均数据）
    am_twoleggedmean_array = np.zeros(len(data3[:,1,1]))
    yearnum = len(data3[:,1,1])
    for j in range(yearnum):
        am_twoleggedmean_array[j] = np.nanmean(am_twolegged[j,:,:])
    am_twolegged_62[i*12:(i+1)*12] = am_twoleggedmean_array

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(am_twolegged_62,'#100000',linewidth=1,label='am_twolegged')
ax1.set(xlabel='time series (month)', ylabel='twolegged',
       title='AM twolegged in 1960-2021')
#ax1.set_ylim(-0.05,0.6)
ax1.legend()
ax1.grid()

plt.show()

# %%
res_am_twolegged = STL(am_twolegged_62, period=12).fit()
res_am_twolegged.plot()
plt.show()

# %%
res_am_twolegged_resid = res_am_twolegged.resid

# %%
import fathon
from fathon import fathonUtils as fu

# %%
res_am_twolegged_resid_alpha = np.zeros(504)
for i in range(504):
    temp = 120+i
    pydfa = fathon.DFA(res_am_twolegged_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    res_am_twolegged_resid_alpha[i] = H

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,res_am_twolegged_resid_alpha,'#000000')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='am_twolegged_DFA in 1960-2021')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()

plt.show()

# %%
var_am_twolegged = np.zeros(len(res_am_twolegged_resid)-120)
for i in range(len(res_am_twolegged_resid)-120):
    temp = 120+i
    var_am_twolegged[i] = np.var(res_am_twolegged_resid[i:temp]*100)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,var_am_twolegged,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='VAR',
       title='VAR(twolegged in am) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
cor_am_twolegged = np.zeros(len(res_am_twolegged_resid)-120)
for i in range(len(res_am_twolegged_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_am_twolegged[i] = np.corrcoef(res_am_twolegged_resid[month_start:month_end-1],res_am_twolegged_resid[month_start+1:month_end])[0,1]

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,cor_am_twolegged,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='AR1',
       title='AR1(twolegged in am) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
# 一共62年的数据
# 存储62年的数据
am_tet_62 = np.zeros(624)
for i in range(52):
    # 导入am地区陆气耦合数据
    # 陆气耦合为三维数据nc格式文件
    #am_tet_path = 'F:/ERA5/landatmos_coupl/am_tet/'
    am_tet_path = 'F:/ERA5/陆气耦合重算/am_tet/'
    year_num = i+1970
    year = str(year_num)
    # 打开npy文件
    data3 = np.load(am_tet_path+ year + '.npy')
    # mask与数据矩阵对应相乘
    am_tet = data3*am_mask_array2
    # 求均值（数据本身为月平均数据）
    am_tetmean_array = np.zeros(len(data3[:,1,1]))
    yearnum = len(data3[:,1,1])
    for j in range(yearnum):
        am_tetmean_array[j] = np.nanmean(am_tet[j,:,:])
    am_tet_62[i*12:(i+1)*12] = am_tetmean_array

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(am_tet_62,'#100000',linewidth=1,label='am_twolegged')
ax1.set(xlabel='time series (month)', ylabel='tet',
       title='AM tet in 1960-2021')
#ax1.set_ylim(-0.05,0.6)
ax1.legend()
ax1.grid()

plt.show()

# %%
res_am_tet = STL(am_tet_62, period=12).fit()
res_am_tet.plot()
plt.show()

# %%
res_am_tet_resid = res_am_tet.resid

# %%
import fathon
from fathon import fathonUtils as fu

# %%
res_am_tet_resid_alpha = np.zeros(504)
for i in range(504):
    temp = 120+i
    pydfa = fathon.DFA(res_am_tet_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    res_am_tet_resid_alpha[i] = H

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,res_am_tet_resid_alpha,'#000000')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='am_tet_DFA in 1960-2021')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()

plt.show()

# %%
cor_am_tet = np.zeros(len(res_am_tet_resid)-120)
for i in range(len(res_am_tet_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_am_tet[i] = np.corrcoef(res_am_tet_resid[month_start:month_end-1],res_am_tet_resid[month_start+1:month_end])[0,1]

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,cor_am_tet,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='AR1',
       title='AR1(tet in am) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
var_am_tet = np.zeros(len(res_am_tet_resid)-120)
for i in range(len(res_am_tet_resid)-120):
    temp = 120+i
    var_am_tet[i] = np.var(res_am_tet_resid[i:temp]*100)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,var_am_tet,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='VAR',
       title='VAR(tet in am) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
# 一共62年的数据
# 存储62年的数据
am_pai_62 = np.zeros(744)
for i in range(62):
    # 导入am地区陆气耦合数据
    # 陆气耦合为三维数据nc格式文件
    am_pai_path = 'F:/ERA5/陆气耦合重算/am_pai/'
    year_num = i+1960
    year = str(year_num)
    # 打开npy文件
    data3 = np.load(am_pai_path+ year + '.npy')
    # mask与数据矩阵对应相乘
    am_pai = data3*am_mask_array
    am_pai[np.where(am_pai==-32767)] = np.nan
    # 求均值（数据本身为月平均数据）
    am_paimean_array = np.zeros(len(data3[:,1,1]))
    yearnum = len(data3[:,1,1])
    for j in range(yearnum):
        am_paimean_array[j] = np.nanmean(am_pai[j,:,:])
    # 当场转化为月平均数据
    monthly = np.zeros(12)
    for k in range(12):
        if (k+1)*30<len(am_paimean_array):
            monthly[k] = np.nanmean(am_paimean_array[k*30:(k+1)*30])
        else:
            monthly[k] = np.nanmean(am_paimean_array[k*30:len(am_paimean_array)-1])
    am_pai_62[i*12:(i+1)*12] = monthly

# %%
am_pai_52 = am_pai_62[120:780]

# %%
np.shape(am_pai_52)

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(am_pai_52,'#100000',linewidth=1,label='am_pai')
ax1.set(xlabel='time series (month)', ylabel='pai',
       title='AM pai in 1960-2021')
#ax1.set_ylim(-0.05,0.6)
ax1.legend()
ax1.grid()

plt.show()

# %%
res_am_pai = STL(am_pai_52, period=12).fit()
res_am_pai.plot()
plt.show()

# %%
res_am_pai_resid = res_am_pai.resid

# %%
res_am_pai_resid_alpha = np.zeros(504)
for i in range(504):
    temp = 120+i
    pydfa = fathon.DFA(res_am_pai_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    res_am_pai_resid_alpha[i] = H

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,res_am_pai_resid_alpha,'#000000')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='am_pai_DFA in 1960-2021')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()

plt.show()

# %%
cor_am_pai = np.zeros(len(res_am_pai_resid)-120)
for i in range(len(res_am_pai_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_am_pai[i] = np.corrcoef(res_am_pai_resid[month_start:month_end-1],res_am_pai_resid[month_start+1:month_end])[0,1]

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,cor_am_pai,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='AR1',
       title='AR1(pai in am) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
var_am_pai = np.zeros(len(res_am_pai_resid)-120)
for i in range(len(res_am_pai_resid)-120):
    temp = 120+i
    var_am_pai[i] = np.var(res_am_pai_resid[i:temp]*100)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,var_am_pai,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='VAR',
       title='VAR(pai in am) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
cor_am_ndvi = np.load('cor_am_ndvi.npy')
var_am_ndvi = np.load('var_am_ndvi.npy')
dfa_am_ndvi_new = np.load('dfa_am_ndvi_new.npy')

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel[0:60],res_am_pai_resid_alpha[0:60],'lightgray')
ax1.plot(xlabel[60:298],res_am_pai_resid_alpha[60:298],'#CD5C5C',label='pai')
ax1.plot(xlabel[298:504],res_am_pai_resid_alpha[298:504],'lightgray')

ax1.plot(xlabel[0:60],res_am_tet_resid_alpha[0:60],'lightgray')
ax1.plot(xlabel[60:298],res_am_tet_resid_alpha[60:298],'#000000',label='tet')
ax1.plot(xlabel[298:504],res_am_tet_resid_alpha[298:504],'lightgray')


#ax1.plot(xlabel,res_am_gamma_resid_alpha,'#DAA520',label='gamma')
#ax1.plot(xlabel,res_am_twolegged_resid_alpha,'#4169E1',label='twolegged')
ax1.plot(xlabel[0:60],res_am_tci_resid_alpha[0:60],'lightgray')
ax1.plot(xlabel[60:298],res_am_tci_resid_alpha[60:298],'teal',label='tci')
ax1.plot(xlabel[298:504],res_am_tci_resid_alpha[298:504],'lightgray')


ax1.plot(date3[0:30],dfa_am_ndvi_new[0:30],'lightgray')
ax1.plot(date3[30:270],dfa_am_ndvi_new[30:270],'lawngreen',label='ndvi-10year early')
ax1.plot(date3[270:504],dfa_am_ndvi_new[270:504],'lightgray')
#ax1.plot(date6,am_ndvi_DFA_alpha,'green',label='ndvi')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='am_pai_DFA in 1980-2021')
ax1.set_xlim(date1[164],date1[743])
ax1.grid()
ax1.legend()

plt.show()

# %%
la_mean = res_am_pai_resid_alpha+res_am_tet_resid_alpha+res_am_tci_resid_alpha
la_mean = la_mean/3

# %%
# 三合一陆气耦合指数
la_dfamat = np.array((res_am_pai_resid_alpha,
                      res_am_tet_resid_alpha,res_am_tci_resid_alpha))
# 求最大最小值
la_dfamat_max = np.max(la_dfamat,axis=0)
la_dfamat_min = np.min(la_dfamat,axis=0)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(10,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
#ax1.plot(xlabel[0:60],res_am_pai_resid_alpha[0:60],'lightgray')
#ax1.plot(xlabel[60:298],res_am_pai_resid_alpha[60:298],'#CD5C5C',label='pai')
#ax1.plot(xlabel[298:504],res_am_pai_resid_alpha[298:504],'lightgray')

#ax1.plot(xlabel[0:60],res_am_tet_resid_alpha[0:60],'lightgray')
#ax1.plot(xlabel[60:298],res_am_tet_resid_alpha[60:298],'#000000',label='tet')
#ax1.plot(xlabel[298:504],res_am_tet_resid_alpha[298:504],'lightgray')


#ax1.plot(xlabel,res_am_gamma_resid_alpha,'#DAA520',label='gamma')
ax2.plot(xlabel,la_mean,'royalblue',label='land-atmosphere coupling')
#ax1.plot(xlabel[0:60],res_am_tci_resid_alpha[0:60],'lightgray')
#ax1.plot(xlabel[60:298],res_am_tci_resid_alpha[60:298],'teal',label='tci')
#ax1.plot(xlabel[298:504],res_am_tci_resid_alpha[298:504],'lightgray')


ax1.plot(date3[0:30],am_ndvi_DFA_alpha[0:30],'green', alpha=0.9)
ax1.plot(date3[30:270],am_ndvi_DFA_alpha[30:270],'green',label='ndvi-12year early', alpha=0.9)
ax1.plot(date31[0:326],am_ndvi_DFA_alpha[0:326],ls='--',c='green',label='ndvi', alpha=0.2)
ax1.plot(date3[270:326],am_ndvi_DFA_alpha[270:326],'green', alpha=0.9)
#ax1.plot(date6,am_ndvi_DFA_alpha,'green',label='ndvi')
ax1.set(xlabel='time series', ylabel='DFA alpha')
ax1.set_xlim(date1[164],date1[743])
ax1.set_ylim(0.01,0.14)
#ax2.set_ylim(0.03,0.15)
ax1.grid()
ax1.legend(loc = 'upper left')
ax2.legend(['land-atmosphere coupling','TET'],loc='upper right',shadow=True,fancybox=True)

# 画出上下界
ax2.fill_between(xlabel, la_dfamat_min, la_dfamat_max, facecolor='royalblue', alpha=0.3)

plt.show()

# %%
am_ndvi_dfa_clip = am_ndvi_DFA_alpha[30:270]
n = len(am_ndvi_dfa_clip)
time_lag = range(len(res_am_tet_resid_alpha)-n)


# r_lag存储不同时滞下的相关系数
r_lag_dfa = np.zeros([len(time_lag),4])
# 下面开始循环计算
count = 504
for i in time_lag:
    temp1 = np.corrcoef(res_am_pai_resid_alpha[count-i-n:count-i],am_ndvi_dfa_clip)
    r_lag_dfa[i,0] = temp1[0,1]
    temp2 = np.corrcoef(res_am_tet_resid_alpha[count-i-n:count-i],am_ndvi_dfa_clip)
    r_lag_dfa[i,1] = temp2[0,1]
    temp3 = np.corrcoef(res_am_tci_resid_alpha[count-i-n:count-i],am_ndvi_dfa_clip)
    r_lag_dfa[i,2] = temp3[0,1]
    temp4 = np.corrcoef(la_ar1mean[count-i-n:count-i],am_ndvi_dfa_clip)
    r_lag_dfa[i,3] = temp4[0,1]

# %%
am_ndvi_dfa_clip = am_ndvi_DFA_alpha[156:328]
n = len(am_ndvi_dfa_clip)
time_lag = range(len(res_am_tet_resid_alpha)-n)


# r_lag存储不同时滞下的相关系数
r_lag_dfa = np.zeros([len(time_lag),4])
# 下面开始循环计算
count = 504
for i in time_lag:
    temp1 = np.corrcoef(res_am_pai_resid_alpha[count-i-n:count-i],am_ndvi_dfa_clip)
    r_lag_dfa[i,0] = temp1[0,1]
    temp2 = np.corrcoef(res_am_tet_resid_alpha[count-i-n:count-i],am_ndvi_dfa_clip)
    r_lag_dfa[i,1] = temp2[0,1]
    temp3 = np.corrcoef(res_am_tci_resid_alpha[count-i-n:count-i],am_ndvi_dfa_clip)
    r_lag_dfa[i,2] = temp3[0,1]
    temp4 = np.corrcoef(la_mean[count-i-n:count-i],am_ndvi_dfa_clip)
    r_lag_dfa[i,3] = temp4[0,1]

# %%
plot_color = ['salmon','darkorange','yellowgreen','royalblue']
legend = ['ndvi_pai','ndvi_tet','ndvi_tci','ndvi_la-mean']
fig = plt.figure(figsize=(10,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
for i in range(4):
    plt.plot(time_lag[0:200],r_lag_dfa[0:200,i],plot_color[i],label=legend[i])   
    #plt.axvline(lag_where[i],ls='-.',color=plot_color[i])
plt.grid()
plt.title('AM dfa time-lag')
ax1.legend(loc='upper left')
ax1.axvline(176,ls='-.',color=plot_color[0])
ax1.axvline(156,ls='-.',color=plot_color[1])
ax1.axvline(164,ls='-.',color=plot_color[2])
ax1.axvline(148,ls='-.',color=plot_color[3])
ax1.set(xlabel='ahead of time', ylabel='corrcoef')

# %%
la_ar1mean = cor_am_pai+cor_am_tet+cor_am_tci
la_ar1mean = la_ar1mean/3

# %%
# 三合一陆气耦合指数
la_ar1mat = np.array((cor_am_pai,cor_am_tet,cor_am_tci))
# 求最大最小值
la_ar1mat_max = np.max(la_ar1mat,axis=0)
la_ar1mat_min = np.min(la_ar1mat,axis=0)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(10,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
#ax1.plot(xlabel[0:184],cor_am_pai[0:184],'#CD5C5C',alpha = 0.3)
#ax1.plot(xlabel[184:394],cor_am_pai[184:394],'#CD5C5C',label='pai')
#ax1.plot(xlabel[394:504],cor_am_pai[394:504],'lightgray')


#ax1.plot(xlabel[0:180],cor_am_tet[0:180],'#000000',alpha = 0.3)
#ax1.plot(xlabel[180:394],cor_am_tet[180:394],'#000000',label='tet')
#ax1.plot(xlabel[394:504],cor_am_tet[394:504],'lightgray')


ax2.plot(xlabel,la_ar1mean,'royalblue',label='land-atmosphere coupling')
#ax1.plot(xlabel,cor_am_gamma,'#DAA520',label='gamma')
#ax1.plot(xlabel,np.absolute(cor_am_twolegged),'#4169E1',label='twolegged')
#ax1.plot(xlabel[0:180],cor_am_tci[0:180],'teal',alpha = 0.3)
#ax1.plot(xlabel[180:394],cor_am_tci[180:394],'teal',label='tci')
#ax1.plot(xlabel[394:504],cor_am_tci[394:504],'lightgray')



ax1.plot(date5[0:165],cor_am_ndvi[0:165],c='green',alpha = 1)
ax1.plot(date5[165:345],cor_am_ndvi[165:345],c='green',label='ndvi_10year_early')
ax1.plot(date51,cor_am_ndvi,ls='--',c='green',label='ndvi',alpha = 0.2)
#ax1.plot(date6,cor_am_ndvi,'green',label='ndvi')

# 画出上下界
ax2.fill_between(xlabel, la_ar1mat_min, la_ar1mat_max, facecolor='royalblue', alpha=0.3)

ax1.set(xlabel='time series', ylabel='AR1')
ax1.set_xlim(date1[164],date1[743])
ax1.set_ylim(-0.25,0.5)
ax2.set_ylim(-0.10,0.40)
ax1.grid()
ax1.legend(loc = 'upper left')
ax2.legend(['land-atmosphere coupling'],loc='upper right',shadow=True,fancybox=True)

plt.show()

# %%
cor_am_ndvi_clip = cor_am_ndvi[156:345]
n = len(cor_am_ndvi_clip)
time_lag = range(len(cor_am_pai)-n)


# r_lag存储不同时滞下的相关系数
r_lag_ar1= np.zeros([len(time_lag),4])
# 下面开始循环计算
count = 504
for i in time_lag:
    temp1 = np.corrcoef(cor_am_pai[count-i-n:count-i],cor_am_ndvi_clip)
    r_lag_ar1[i,0] = temp1[0,1]
    temp2 = np.corrcoef(cor_am_tet[count-i-n:count-i],cor_am_ndvi_clip)
    r_lag_ar1[i,1] = temp2[0,1]
    temp3 = np.corrcoef(cor_am_tci[count-i-n:count-i],cor_am_ndvi_clip)
    r_lag_ar1[i,2] = temp3[0,1]
    temp4 = np.corrcoef(la_ar1mean[count-i-n:count-i],cor_am_ndvi_clip)
    r_lag_ar1[i,3] = temp4[0,1]

# %%
plot_color = ['salmon','darkorange','yellowgreen','royalblue']
legend = ['ndvi_pai','ndvi_tet','ndvi_tci','ndvi_la-mean']
fig = plt.figure(figsize=(10,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
for i in range(4):
    plt.plot(time_lag[0:200],r_lag_ar1[0:200,i],plot_color[i],label=legend[i])   
    #plt.axvline(lag_where[i],ls='-.',color=plot_color[i])
plt.grid()
plt.title('AM AR1 time-lag')
ax1.legend(loc='upper center')
ax1.axvline(158,ls='-.',color=plot_color[0])
ax1.axvline(162,ls='-.',color=plot_color[1])
ax1.axvline(142,ls='-.',color=plot_color[2])
ax1.axvline(148,ls='-.',color=plot_color[3])
ax1.set(xlabel='ahead of time', ylabel='corrcoef')

# %%
plot_color = ['teal','navy','darkviolet']
legend = ['ndvi_pai','ndvi_tet','ndvi_tci']
plt.figure(figsize=(15,4)) # 设置画板大小
for i in range(3):
    plt.plot(time_lag_ar1,r_lag_ar1[:,i],plot_color[i],label=legend[i])   
    #plt.axvline(lag_where[i],ls='-.',color=plot_color[i])
plt.grid()
plt.title('ar1 time-lag')
plt.legend(loc='upper right')
plt.axvline(75,ls='-.',color=plot_color[0])
plt.axvline(68,ls='-.',color=plot_color[1])
plt.axvline(55,ls='-.',color=plot_color[2])

# %%


# %%


# %%


# %%
var_am_pai_normal = var_am_pai/np.linalg.norm(var_am_pai)
var_am_tet_normal = var_am_tet/np.linalg.norm(var_am_tet)
var_am_gamma_normal = var_am_gamma/np.linalg.norm(var_am_gamma)
var_am_twolegged_normal = var_am_twolegged/np.linalg.norm(var_am_twolegged)
var_am_tci_normal = var_am_tci/np.linalg.norm(var_am_tci)
var_am_ndvi_normal = var_am_ndvi/np.linalg.norm(var_am_ndvi)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel[0:298],var_am_pai_normal[0:298],'#CD5C5C',label='pai')
ax1.plot(xlabel[298:504],var_am_pai_normal[298:504],'lightgray')


ax1.plot(xlabel[0:298],var_am_tet_normal[0:298],'#000000',label='tet')
ax1.plot(xlabel[298:504],var_am_tet_normal[298:504],'lightgray')

#ax1.plot(xlabel,var_am_gamma_normal,'#DAA520',label='gamma')
#ax1.plot(xlabel,var_am_twolegged_normal,'#4169E1',label='twolegged')
ax1.plot(xlabel[0:298],var_am_tci_normal[0:298],'teal',label='tci')
ax1.plot(xlabel[298:504],var_am_tci_normal[298:504],'lightgray')

ax1.plot(date5[0:298],var_am_ndvi_normal[0:298],'lawngreen',label='ndvi-10year early')
ax1.plot(date5[298:504],var_am_ndvi_normal[298:504],'lightgray')

#ax1.plot(date6,var_am_ndvi_normal,'green',label='ndvi')
ax1.set(xlabel='time series (month)', ylabel='VAR',
       title='VAR in 1980-2021')
ax1.set_xlim(date1[164],date1[743])
ax1.grid()
ax1.legend()

plt.show()

# %%


# %%
