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
import fathon
from fathon import fathonUtils as fu

# %%
# 导入npy格式数据
tp_tet_path = 'F:/ERA5/陆气耦合重算/tp_tet/'
year_num = 1970
year = str(year_num)
# 打开npy文件
tp_tet = np.load(tp_tet_path+ year + '.npy')

# %%
np.shape(tp_tet)

# %%
tpfe_mask_path = 'D:/GEE学习/临界要素数据处理/研究区域掩膜tiff/tpfe3.tif'
tpgl_mask_path = 'D:/GEE学习/临界要素数据处理/研究区域掩膜tiff/tpgl3.tif'
# 读取tiff文件
tpfe_mask = tf.imread(tpfe_mask_path)
tpgl_mask = tf.imread(tpgl_mask_path)
print(tpgl_mask.shape)

# %%
tpfe_mask_array = np.array(tpfe_mask)
tpgl_mask_array = np.array(tpgl_mask)

# %%
plt.imshow(tp_tet[5,:,:])

# %%
plt.imshow(tpfe_mask_array)

# %%
plt.imshow(tpgl_mask_array)

# %%
tpfe_mask_array = np.where(tpfe_mask_array,tpfe_mask_array,np.nan)

# %%
tpgl_mask_array = np.where(tpgl_mask_array,tpgl_mask_array,np.nan)

# %%
date1 = pd.date_range('19600101','20220101',freq='M')
date2 = pd.date_range('19700101','20220101',freq='M')
date3 = pd.date_range('19820701','20090901',freq='M')
date4 = pd.date_range('19800101','20220101',freq='M')
date_tci = pd.date_range('19780101','20200101',freq='M')
#date5 = pd.date_range('19920101','20190301',freq='M')
date5 = pd.date_range('19720701','19990901',freq='M')
date6 = pd.date_range('19900101','20210101',freq='M')
ndvi_date = pd.date_range('19920101','20210101',freq='M')
ndvi_date2 = pd.date_range('19850101','20140101',freq='M')
ndvi_date22 = pd.date_range('19830101','20120101',freq='M')

# %%
# 一共62年的数据
# 存储62年的数据
tp_tci_62 = np.zeros(624)
for i in range(52):
    # 导入pe地区积雪深度数据
    # 积雪深度为三维数据nc格式文件
    #am_tci_path = 'F:/ERA5/landatmos_coupl/am_TCI/'
    tp_tci_path = 'F:/ERA5/陆气耦合重算/tp_TCI/'
    #am_tci_path = 'D:/GEE学习/陆气耦合计算/am_TCI/'
    year_num = i+1970
    year = str(year_num)
    # 打开npy文件
    data2 = np.load(tp_tci_path+ year + '.npy')
    # mask与数据矩阵对应相乘
    tp_tci = data2*tpgl_mask_array
    # 求均值（数据本身为月平均数据）
    tp_tcimean_array = np.zeros(len(data2[:,1,1]))
    yearnum = len(data2[:,1,1])
    for j in range(yearnum):
        tp_tcimean_array[j] = np.nanmean(tp_tci[j,:,:])
    tp_tci_62[i*12:(i+1)*12] = tp_tcimean_array

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(tp_tci_62,'#000000',linewidth=1,label='tp_tci')
ax1.set(xlabel='time series (month)', ylabel='TCI',
       title='TPFE TCI in 1960-2021')
#ax1.set_ylim(-0.05,0.6)
ax1.legend()
ax1.grid()

plt.show()

# %%
res_tp_tci = STL(tp_tci_62, period=12,robust=True).fit()
res_tp_tci.plot()
plt.show()

# %%
res_tp_tci_resid = res_tp_tci.resid

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(res_tp_tci_resid,'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='TCI',
       title='TPFE TCI res in 1960-2021')
#ax1.set_ylim(-0.05,0.6)
ax1.grid()

plt.show()

# %%
res_tp_tci_resid_alpha = np.zeros(504)
for i in range(504):
    temp = 120+i
    pydfa = fathon.DFA(res_tp_tci_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    res_tp_tci_resid_alpha[i] = H

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,res_tp_tci_resid_alpha,'#000000')
#ax1.plot(date5,pe_ndvi_DFA_alpha,'#DAA520')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='tp_TCI_DFA in 1960-2021')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()

plt.show()

# %%
cor_tp_tci = np.zeros(len(res_tp_tci_resid)-120)
for i in range(len(res_tp_tci_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_tp_tci[i] = np.corrcoef(res_tp_tci_resid[month_start:month_end-1],res_tp_tci_resid[month_start+1:month_end])[0,1]

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,cor_tp_tci,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='AR1',
       title='AR1(TCI in tp) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
var_tp_tci = np.zeros(len(res_tp_tci_resid)-120)
for i in range(len(res_tp_tci_resid)-120):
    temp = 120+i
    var_tp_tci[i] = np.var(res_tp_tci_resid[i:temp]*100)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,var_tp_tci,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='VAR',
       title='VAR(TCI in tp) in 1970-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
# 一共62年的数据
# 存储62年的数据
tp_tet_62 = np.zeros(624)
for i in range(52):
    # 导入am地区陆气耦合数据
    # 陆气耦合为三维数据nc格式文件
    #am_tet_path = 'F:/ERA5/landatmos_coupl/am_tet/'
    tp_tet_path = 'F:/ERA5/陆气耦合重算/tp_tet/'
    year_num = i+1970
    year = str(year_num)
    # 打开npy文件
    data3 = np.load(tp_tet_path+ year + '.npy')
    # mask与数据矩阵对应相乘
    tp_tet = data3*tpgl_mask_array
    # 求均值（数据本身为月平均数据）
    tp_tetmean_array = np.zeros(len(data3[:,1,1]))
    yearnum = len(data3[:,1,1])
    for j in range(yearnum):
        tp_tetmean_array[j] = np.nanmean(tp_tet[j,:,:])
    tp_tet_62[i*12:(i+1)*12] = tp_tetmean_array

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(tp_tet_62,'#100000',linewidth=1,label='tp_twolegged')
ax1.set(xlabel='time series (month)', ylabel='tet',
       title='TP tet in 1960-2021')
#ax1.set_ylim(-0.05,0.6)
ax1.legend()
ax1.grid()

plt.show()

# %%
res_tp_tet = STL(tp_tet_62, period=12).fit()
res_tp_tet.plot()
plt.show()

# %%
res_tp_tet_resid = res_tp_tet.resid

# %%
res_tp_tet_resid_alpha = np.zeros(504)
for i in range(504):
    temp = 120+i
    pydfa = fathon.DFA(res_tp_tet_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    res_tp_tet_resid_alpha[i] = H

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,res_tp_tet_resid_alpha,'#000000')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='tp_tet_DFA in 1960-2021')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()

plt.show()

# %%
cor_tp_tet = np.zeros(len(res_tp_tet_resid)-120)
for i in range(len(res_tp_tet_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_tp_tet[i] = np.corrcoef(res_tp_tet_resid[month_start:month_end-1],res_tp_tet_resid[month_start+1:month_end])[0,1]

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,cor_tp_tet,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='AR1',
       title='AR1(tet in tp) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
var_tp_tet = np.zeros(len(res_tp_tet_resid)-120)
for i in range(len(res_tp_tet_resid)-120):
    temp = 120+i
    var_tp_tet[i] = np.var(res_tp_tet_resid[i:temp]*100)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,var_tp_tet,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='VAR',
       title='VAR(tet in tp) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
# 一共62年的数据
# 存储62年的数据
tp_twolegged_62 = np.zeros(624)
for i in range(52):
    # 导入am地区陆气耦合数据
    # 陆气耦合为三维数据nc格式文件
    #am_twolegged_path = 'D:/GEE学习/陆气耦合计算/am_twolegged/'
    tp_twolegged_path = 'F:/ERA5/陆气耦合重算/tp_twolegged/'
    year_num = i+1970
    year = str(year_num)
    # 打开npy文件
    data3 = np.load(tp_twolegged_path+ year + '.npy')
    # mask与数据矩阵对应相乘
    tp_twolegged = data3*tpgl_mask_array
    # 求均值（数据本身为月平均数据）
    tp_twoleggedmean_array = np.zeros(len(data3[:,1,1]))
    yearnum = len(data3[:,1,1])
    for j in range(yearnum):
        tp_twoleggedmean_array[j] = np.nanmean(tp_twolegged[j,:,:])
    tp_twolegged_62[i*12:(i+1)*12] = tp_twoleggedmean_array

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(tp_twolegged_62,'#100000',linewidth=1,label='tp_twolegged')
ax1.set(xlabel='time series (month)', ylabel='twolegged',
       title='TP twolegged in 1960-2021')
#ax1.set_ylim(-0.05,0.6)
ax1.legend()
ax1.grid()

plt.show()

# %%
res_tp_twolegged = STL(tp_twolegged_62, period=12).fit()
res_tp_twolegged.plot()
plt.show()

# %%
res_tp_twolegged_resid = res_tp_twolegged.resid

# %%
res_tp_twolegged_resid_alpha = np.zeros(504)
for i in range(504):
    temp = 120+i
    pydfa = fathon.DFA(res_tp_twolegged_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    res_tp_twolegged_resid_alpha[i] = H

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,res_tp_twolegged_resid_alpha,'#000000')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='tp_twolegged_DFA in 1960-2021')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()

plt.show()

# %%
cor_tp_twolegged = np.zeros(len(res_tp_twolegged_resid)-120)
for i in range(len(res_tp_twolegged_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_tp_twolegged[i] = np.corrcoef(res_tp_twolegged_resid[month_start:month_end-1],res_tp_twolegged_resid[month_start+1:month_end])[0,1]

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,cor_tp_twolegged,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='AR1',
       title='AR1(twolegged in tp) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
var_tp_twolegged = np.zeros(len(res_tp_twolegged_resid)-120)
for i in range(len(res_tp_twolegged_resid)-120):
    temp = 120+i
    var_tp_twolegged[i] = np.var(res_tp_twolegged_resid[i:temp]*100)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,var_tp_twolegged,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='VAR',
       title='VAR(twolegged in tp) in 1970-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
# 一共62年的数据
# 存储62年的数据
tp_gamma_62 = np.zeros(624)
for i in range(52):
    # 导入am地区陆气耦合数据
    # 陆气耦合为三维数据nc格式文件
    #am_gamma_path = 'D:/GEE学习/陆气耦合计算/am_gamma/'
    tp_gamma_path = 'F:/ERA5/陆气耦合重算/tp_gamma/'
    year_num = i+1970
    year = str(year_num)
    # 打开npy文件
    data3 = np.load(tp_gamma_path+ year + '.npy')
    # mask与数据矩阵对应相乘
    tp_gamma = data3*tpgl_mask_array
    # 求均值（数据本身为月平均数据）
    tp_gammamean_array = np.zeros(len(data3[:,1,1]))
    yearnum = len(data3[:,1,1])
    for j in range(yearnum):
        tp_gammamean_array[j] = np.nanmean(tp_gamma[j,:,:])
    tp_gamma_62[i*12:(i+1)*12] = tp_gammamean_array

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(tp_gamma_62,'#100000',linewidth=1,label='tp_gamma')
ax1.set(xlabel='time series (month)', ylabel='Gamma',
       title='TP Gamma in 1960-2021')
#ax1.set_ylim(-0.05,0.6)
ax1.legend()
ax1.grid()

plt.show()

# %%
res_tp_gamma = STL(tp_gamma_62, period=12).fit()
res_tp_gamma.plot()
plt.show()

# %%
res_tp_gamma_resid = res_tp_gamma.resid

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(res_tp_gamma_resid,'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='gamma',
       title='TP gamma res in 1960-202')
#ax1.set_ylim(-0.05,0.6)
ax1.grid()

plt.show()

# %%
res_tp_gamma_resid_alpha = np.zeros(504)
for i in range(504):
    temp = 120+i
    pydfa = fathon.DFA(res_tp_gamma_resid[i:temp])
    #pydfa = fathon.DFA(am_gamma_62[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    res_tp_gamma_resid_alpha[i] = H

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,res_tp_gamma_resid_alpha,'#000000')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='tp_gamma_DFA in 1960-2021')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()

plt.show()

# %%
var_tp_gamma = np.zeros(len(res_tp_gamma_resid)-120)
for i in range(len(res_tp_gamma_resid)-120):
    temp = 120+i
    var_tp_gamma[i] = np.var(res_tp_gamma_resid[i:temp]*100)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,var_tp_gamma,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='VAR',
       title='VAR(gamma in tp) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
cor_tp_gamma = np.zeros(len(res_tp_gamma_resid)-120)
for i in range(len(res_tp_gamma_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_tp_gamma[i] = np.corrcoef(res_tp_gamma_resid[month_start:month_end-1],res_tp_gamma_resid[month_start+1:month_end])[0,1]

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,cor_tp_gamma,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='AR1',
       title='AR1(gamma in tp) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
# 一共62年的数据
# 存储62年的数据
tp_pai_62 = np.zeros(744)
for i in range(62):
    # 导入am地区陆气耦合数据
    # 陆气耦合为三维数据nc格式文件
    tp_pai_path = 'F:/ERA5/陆气耦合重算/tp_pai/'
    year_num = i+1960
    year = str(year_num)
    # 打开npy文件
    data3 = np.load(tp_pai_path+ year + '.npy')
    # mask与数据矩阵对应相乘
    tp_pai = data3*tpgl_mask_array
    tp_pai[np.where(tp_pai==-32767)] = np.nan
    # 求均值（数据本身为月平均数据）
    tp_paimean_array = np.zeros(len(data3[:,1,1]))
    yearnum = len(data3[:,1,1])
    for j in range(yearnum):
        tp_paimean_array[j] = np.nanmean(tp_pai[j,:,:])
    # 当场转化为月平均数据
    monthly = np.zeros(12)
    for k in range(12):
        if (k+1)*30<len(tp_paimean_array):
            monthly[k] = np.nanmean(tp_paimean_array[k*30:(k+1)*30])
        else:
            monthly[k] = np.nanmean(tp_paimean_array[k*30:len(pe_paimean_array)-1])
    tp_pai_62[i*12:(i+1)*12] = monthly

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(tp_pai_62,'#100000',linewidth=1,label='tp_pai')
ax1.set(xlabel='time series (month)', ylabel='pai',
       title='tp pai in 1960-2021')
#ax1.set_ylim(-0.05,0.6)
ax1.legend()
ax1.grid()

plt.show()

# %%
res_tp_pai = STL(tp_pai_62, period=12).fit()
res_tp_pai.plot()
plt.show()

# %%
res_tp_pai_resid = res_tp_pai.resid

# %%
res_tp_pai_resid_alpha = np.zeros(624)
for i in range(624):
    temp = 120+i
    pydfa = fathon.DFA(res_tp_pai_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    res_tp_pai_resid_alpha[i] = H

# %%
# Data for plotting
xlabel = date2
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,res_tp_pai_resid_alpha,'#000000')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='tp_pai_DFA in 1960-2021')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()

plt.show()

# %%
cor_tp_pai = np.zeros(len(res_tp_pai_resid)-120)
for i in range(len(res_tp_pai_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_tp_pai[i] = np.corrcoef(res_tp_pai_resid[month_start:month_end-1],res_tp_pai_resid[month_start+1:month_end])[0,1]

# %%
# Data for plotting
xlabel = date2
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,cor_tp_pai,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='AR1',
       title='AR1(pai in tp) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
var_tp_pai = np.zeros(len(res_tp_pai_resid)-120)
for i in range(len(res_tp_pai_resid)-120):
    temp = 120+i
    var_tp_pai[i] = np.var(res_tp_pai_resid[i:temp]*100)

# %%
# Data for plotting
xlabel = date2
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,var_tp_pai,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='VAR',
       title='VAR(pai in tp) in 1960-2019')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
cor_tpgl_ndvi = np.load('tpgl_ndvi_AR1.npy')
cor_tpgl_ndvi2 = np.load('cor_tpgl_ndvi.npy')
var_tpgl_ndvi = np.load('tpgl_ndvi_VAR.npy')
res_tpgl_ndvi_resid_alpha = np.load('tpgl_ndvi_DFA.npy')
res_tpgl_ndvi_resid_alpha2 = np.load('tpgl_ndvi_DFA_alpha.npy')

# %%
cor_tp_scf = np.load('cor_tp_scf.npy')
var_tp_scf = np.load('var_tp_scf.npy')
res_tp_scf_resid_alpha = np.load('tp_scf_DFA_alpha.npy')

# %%
la_mean = res_tp_pai_resid_alpha[120:748]+res_tp_tet_resid_alpha+res_tp_tci_resid_alpha
la_mean = la_mean/3

# %%
# 三合一陆气耦合指数
la_dfamat = np.array((res_tp_pai_resid_alpha[120:748],res_tp_tet_resid_alpha,res_tp_tci_resid_alpha))
# 求最大最小值
la_dfamat_max = np.max(la_dfamat,axis=0)
la_dfamat_min = np.min(la_dfamat,axis=0)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(10,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
#ax2.plot(date2[120:748],res_tp_pai_resid_alpha[120:748],'#CD5C5C',label='pai')
#ax1.plot(xlabel,res_tp_tet_resid_alpha,'#000000',label='tet')
#ax1.plot(xlabel,res_tp_gamma_resid_alpha*3,'#DAA520',label='gamma')
#ax1.plot(xlabel,res_tp_twolegged_resid_alpha*3,'#4169E1',label='twolegged')
#ax1.plot(date_tci,res_tp_tci_resid_alpha,'#DAA520',label='tci')
ax2.plot(xlabel,la_mean,'royalblue',label='land-atmosphere coupling')
ax1.plot(ndvi_date,res_tpgl_ndvi_resid_alpha2,ls='--',c='green',label='ndvi', alpha=0.2)
#ax1.plot(date6,res_tp_scf_resid_alpha,'#4169E1',label='scf')
ax1.plot(ndvi_date2,res_tpgl_ndvi_resid_alpha2,'green',label='ndvi-7 year early')


# 画出上下界
ax2.fill_between(xlabel, la_dfamat_min, la_dfamat_max, facecolor='royalblue', alpha=0.3)

ax1.set(xlabel='time series', ylabel='DFA alpha')
ax2.set(ylabel='Land-atmosphere DFA alpha')
ax1.set_xlim(date1[115],date1[743])
ax1.set_ylim(-0.05,0.15)
#ax2.set_ylim(0.04,0.10)
ax1.grid()
ax1.legend(loc='upper left')
ax2.legend()

plt.show()

# %%
DFA_tpgl_ndvi_clip = res_tpgl_ndvi_resid_alpha2[46:348]
n = len(DFA_tpgl_ndvi_clip)
time_lag_DFA = range(len(res_tp_tet_resid_alpha)-n)


# r_lag存储不同时滞下的相关系数
r_lag_DFA = np.zeros([len(time_lag_DFA),4])
# 下面开始循环计算
count = 504
for i in time_lag_DFA:
    temp1 = np.corrcoef(res_tp_tet_resid_alpha[count-i-n:count-i],DFA_tpgl_ndvi_clip)
    r_lag_DFA[i,0] = temp1[0,1]
    temp2 = np.corrcoef(res_tp_pai_resid_alpha[count-i-n:count-i],DFA_tpgl_ndvi_clip)
    r_lag_DFA[i,1] = temp2[0,1]
    temp3 = np.corrcoef(res_tp_tci_resid_alpha[count-i-n:count-i],DFA_tpgl_ndvi_clip)
    r_lag_DFA[i,2] = temp3[0,1]
    temp4 = np.corrcoef(la_mean[count-i-n:count-i],DFA_tpgl_ndvi_clip)
    r_lag_DFA[i,3] = temp4[0,1]

# %%
plot_color = ['salmon','darkorange','yellowgreen','royalblue']
legend = ['ndvi_pai','ndvi_tet','ndvi_tci','ndvi_la-mean']
fig = plt.figure(figsize=(10,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
for i in range(4):
    plt.plot(time_lag_DFA,r_lag_DFA[:,i],plot_color[i],label=legend[i])   
    #plt.axvline(lag_where[i],ls='-.',color=plot_color[i])
plt.grid()
plt.title('TP dfa time-lag')
ax1.legend(loc='upper left')
ax1.axvline(84,ls='-.',color=plot_color[0])
ax1.axvline(108,ls='-.',color=plot_color[1])
ax1.axvline(56,ls='-.',color=plot_color[2])
ax1.axvline(84,ls='-.',color=plot_color[3])
ax1.set(xlabel='ahead of time', ylabel='corrcoef')

# %%
la_ar1mean = cor_tp_pai[120:748]+cor_tp_tet+cor_tp_tci
la_ar1mean = la_ar1mean/3

# %%
# 三合一陆气耦合指数
la_ar1mat = np.array((cor_tp_pai[120:748],cor_tp_tet,cor_tp_tci))
# 求最大最小值
la_ar1mat_max = np.max(la_ar1mat,axis=0)
la_ar1mat_min = np.min(la_ar1mat,axis=0)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(10,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
#ax1.plot(date2[120:748],cor_tp_pai[120:748],'#CD5C5C',label='pai')
#ax2.plot(xlabel,cor_tp_tet,'#000000',label='tet')
#ax1.plot(xlabel,cor_pe_gamma,'#DAA520',label='gamma')
#ax1.plot(xlabel,cor_pe_twolegged,'#4169E1',label='twolegged')
#ax2.plot(xlabel,cor_tp_tci,'#DAA520',label='tci')
ax2.plot(xlabel,la_ar1mean,'royalblue',label='land-atmosphere coupling')
#ax1.plot(date6,cor_tp_scf,'#4169E1',label='ndvi')
ax1.plot(ndvi_date22,cor_tpgl_ndvi2,'green',label='ndvi-9 year early')
ax1.plot(ndvi_date,cor_tpgl_ndvi2,ls='--',c='green',label='ndvi', alpha=0.2)

# 画出上下界
ax2.fill_between(xlabel, la_ar1mat_min, la_ar1mat_max, facecolor='royalblue', alpha=0.3)


ax1.set(xlabel='time series', ylabel='AR1')
ax2.set(ylabel='Land-atmosphere AR1')
ax1.set_xlim(date1[115],date1[743])
ax1.grid()
ax1.set_ylim(-0.4,0.6)
ax2.set_ylim(-0.25,0.3)
ax1.legend(loc='upper left')
ax2.legend()

plt.show()

# %%
cor_tp_ndvi_clip = cor_tpgl_ndvi2[46:348]
n = len(cor_tp_ndvi_clip)
time_lag = range(len(cor_tp_tet)-n)


# r_lag存储不同时滞下的相关系数
r_lag_ar1= np.zeros([len(time_lag),4])
# 下面开始循环计算
count = 504
for i in time_lag:
    temp1 = np.corrcoef(cor_tp_pai[count-i-n+120:count-i+120],cor_tp_ndvi_clip)
    r_lag_ar1[i,0] = temp1[0,1]
    temp2 = np.corrcoef(cor_tp_tet[count-i-n:count-i],cor_tp_ndvi_clip)
    r_lag_ar1[i,1] = temp2[0,1]
    temp3 = np.corrcoef(cor_tp_tci[count-i-n:count-i],cor_tp_ndvi_clip)
    r_lag_ar1[i,2] = temp3[0,1]
    temp4 = np.corrcoef(la_ar1mean[count-i-n:count-i],cor_tp_ndvi_clip)
    r_lag_ar1[i,3] = temp4[0,1]

# %%
plot_color = ['salmon','darkorange','yellowgreen','royalblue']
legend = ['ndvi_pai','ndvi_tet','ndvi_tci','ndvi_la-mean']
fig = plt.figure(figsize=(10,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
for i in range(4):
    plt.plot(time_lag,r_lag_ar1[:,i],plot_color[i],label=legend[i])   
    #plt.axvline(lag_where[i],ls='-.',color=plot_color[i])
plt.grid()
plt.title('TP AR1 time-lag')
ax1.legend(loc='upper left')
ax1.axvline(100,ls='-.',color=plot_color[0])
ax1.axvline(120,ls='-.',color=plot_color[1])
ax1.axvline(75,ls='-.',color=plot_color[2])
ax1.axvline(112,ls='-.',color=plot_color[3])
ax1.set(xlabel='ahead of time', ylabel='corrcoef')

# %%
var_tp_pai_normal = var_tp_pai/np.linalg.norm(var_tp_pai)
var_tp_tet_normal = var_tp_tet/np.linalg.norm(var_tp_tet)
var_tp_gamma_normal = var_tp_gamma/np.linalg.norm(var_tp_gamma)
var_tp_twolegged_normal = var_tp_twolegged/np.linalg.norm(var_tp_twolegged)
var_tp_tci_normal = var_tp_tci/np.linalg.norm(var_tp_tci)
var_tp_scf_normal = var_tp_scf/np.linalg.norm(var_tp_scf)
var_tpgl_ndvi_normal = var_tpgl_ndvi/np.linalg.norm(var_tpgl_ndvi)

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(date2,var_tp_pai_normal,'#CD5C5C',label='pai')
ax1.plot(xlabel,var_tp_tet_normal,'#000000',label='tet')
#ax1.plot(xlabel,var_am_gamma_normal,'#DAA520',label='gamma')
#ax1.plot(xlabel,var_am_twolegged_normal,'#4169E1',label='twolegged')
ax1.plot(xlabel,var_tp_tci_normal,'#DAA520',label='tci')
#ax1.plot(date6,var_tp_scf_normal,'#4169E1',label='ndvi')
ax1.plot(ndvi_date,var_tpgl_ndvi_normal,'teal',label='ndvi')
ax1.set(xlabel='time series (month)', ylabel='VAR',
       title='VAR in 1970-2021')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
ax1.legend()

plt.show()

# %%
