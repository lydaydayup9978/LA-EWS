# %%
# 需要的环境
# stl库，matplotlib，numpy，

# %%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
import fathon
from fathon import fathonUtils as fu

# %%
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline

# %%
from statsmodels.tsa.seasonal import STL

# %%
from scipy.stats import kendalltau

# %%
import statsmodels.tsa.stattools as stattools
from statsmodels.graphics.tsaplots import plot_acf

# %%
# 读取亚马逊NDVI数据
am_data = np.loadtxt('D:/GEE学习/临界要素数据处理/ndvi_am_82_21.csv')
am_ndvi = am_data[:]

# %%
for i in range(len(am_ndvi)):
    if am_ndvi[i] == -9999:
        am_ndvi[i] = None
        #am_ndvi[i] = 0

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot((am_ndvi)/(10000),'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='NDVI',
       title='NDVI in 1982-2019')
ax1.grid()

plt.show()

# %%
# 使用mask可以给nan加回来
mask = np.isnan(am_ndvi)
am_ndvi_monthly_mask = am_ndvi[~mask]

# %%
res_am_ndvi = STL((am_ndvi_monthly_mask)/(10000), period=12).fit()
res_am_ndvi.plot()
plt.show()

# %%
res_am_ndvi_resid = res_am_ndvi.resid
res_am_ndvi_trend = res_am_ndvi.trend
res_am_ndvi_season = res_am_ndvi.seasonal
res_am_ndvi = np.multiply(res_am_ndvi_resid,res_am_ndvi_trend+1)
res_am_ndvi1 = res_am_ndvi_resid+res_am_ndvi_trend+res_am_ndvi_season

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(res_am_ndvi_resid,'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='NDVI res',
       title='NDVI res in 1982-2019')
#ax1.set_ylim(-0.05,0.6)
ax1.grid()

plt.show()

# %%
len(res_am_ndvi_resid)

# %%
am_ndvi_DFA_alpha = np.zeros(345)
for i in range(345):
    temp = 120+i
    pydfa = fathon.DFA(res_am_ndvi_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    am_ndvi_DFA_alpha[i] = H

# %%
#np.save('dfa_am_ndvi_new.npy',am_ndvi_DFA_alpha)

# %%
a = am_ndvi_DFA_alpha[0:163]
a = a[::-1]

# %%
kendalltau(a,am_ndvi_DFA_alpha[163:326])

# %%
date1 = pd.date_range('19600101','20220101',freq='M')
date2 = pd.date_range('19920401','20210101',freq='M')
date3 = pd.date_range('19900101','20210101',freq='M')

# %%
# Data for plotting
xlabel = date2
fig = plt.figure(figsize=(10,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,am_ndvi_DFA_alpha)
ax1.set(xlabel='time series (month)', ylabel='am_ndvi_DFA',
       title='am_ndvi_DFA in 1990-2020')
ax1.grid()

plt.show()

# %%
#np.save('am_ndvi_dfa_9204_2101.npy',am_ndvi_DFA_alpha)

# %%
am_evi_DFA_alpha = np.load('am_evi_DFA_alpha.npy') 
am_vod_DFA_alpha = np.load('am_vod_DFA_alpha.npy')

# %%
date_evi = pd.date_range('20110101','20240101',freq='M')
date_vod = pd.date_range('19970801','20180101',freq='M')

# %%
# Data for plotting
xlabel = date2
fig = plt.figure(figsize=(6,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
#ax1.plot(xlabel[0:36],am_ndvi_DFA_alpha[0:36],ls='--',c='lightgray')
ax1.plot(xlabel[0:156],am_ndvi_DFA_alpha[0:156],ls='--',c='royalblue')
ax1.plot(xlabel[156:345],am_ndvi_DFA_alpha[156:345],c='indianred')
#ax2.plot(date_evi,am_evi_DFA_alpha,c='dimgray')
ax2.plot(date_vod,am_vod_DFA_alpha,c='darkgray')
#ax1.plot(xlabel[290:345],am_ndvi_DFA_alpha[290:345],ls='--',c='#000000')
ax1.set(ylabel='NDVI  DFA-'+r'$\alpha$')
ax2.set(ylabel='VOD  DFA-'+r'$\alpha$')

ax1.set_xlim(date1[325],date1[743])
plt.yticks(np.arange(0,0.21,0.04))
#ax2.set_ylim(0,0.14)

#ax1.axvline(x='20050301',color='black')
#ax1.legend(['NDVI before tipping','NDVI after tipping'],loc='upper right',shadow=True,fancybox=True)
#ax2.legend(['EVI','VOD'],loc='upper right',shadow=True,fancybox=True)
#ax1.grid()

plt.show()

# %%
kendalltau(xlabel[0:156],am_ndvi_DFA_alpha[0:156])

# %%
kendalltau(xlabel[156:312],am_ndvi_DFA_alpha[156:312])

# %%
var_am_ndvi = np.zeros(len(res_am_ndvi_resid)-120)
for i in range(len(res_am_ndvi_resid)-120):
    temp = 120+i
    var_am_ndvi[i] = np.var(res_am_ndvi_resid[i:temp]*100)

# %%
kendalltau(var_am_ndvi[0:163],var_am_ndvi[163:326])

# %%
# Data for plotting
xlabel = date2
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,var_am_ndvi,'#000000',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='VAR',
       title='VAR(ndvi in am) in 1981-2019')
ax1.axvline(x='20070601',color='r')
ax1.set_xlim(date1[0],date1[743])
ax1.grid()
plt.show()

# %%
kendalltau(var_am_ndvi[0:163],var_am_ndvi[163:326])

# %%
cor_am_ndvi = np.zeros(len(res_am_ndvi_resid)-120)
for i in range(len(res_am_ndvi_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_am_ndvi[i] = np.corrcoef(res_am_ndvi_resid[month_start:month_end-1],res_am_ndvi_resid[month_start+1:month_end])[0,1]

# %%
kendalltau(xlabel[175:305],cor_am_ndvi[175:305])

# %%
#np.save('cor_am_ndvi.npy',cor_am_ndvi)

# %%
# Data for plotting
xlabel = date2
fig = plt.figure(figsize=(10,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图'r--'
ax1.plot(xlabel,cor_am_ndvi,linewidth=2.0)
ax1.set(xlabel='time series', ylabel='AR1',
       title='AR1(ndvi in am) in 1991-2019')
#ax1.set_xlim(date1[325],date1[743])
ax1.grid()
plt.show()

# %%
cor_am_evi = np.load('cor_am_evi.npy') 
cor_am_vod = np.load('cor_am_vod.npy')

# %%
# Data for plotting
xlabel = date2
fig = plt.figure(figsize=(6,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
#ax1.plot(xlabel[0:36],am_ndvi_DFA_alpha[0:36],ls='--',c='lightgray')
ax1.plot(xlabel[0:175],cor_am_ndvi[0:175],ls='--',c='royalblue')
ax1.plot(xlabel[175:345],cor_am_ndvi[175:345],c='indianred')
ax2.plot(date_vod,cor_am_vod,c='darkgray')
#ax2.plot(date_evi,cor_am_evi,c='dimgray')

#ax1.plot(xlabel[290:345],am_ndvi_DFA_alpha[290:345],ls='--',c='#000000')
#ax1.set(xlabel='time series', ylabel='AR1')
ax1.set_xlim(date1[325],date1[743])
#ax1.axvline(x='20060901',color='black')
ax1.set(ylabel='NDVI  AR1')
ax2.set(ylabel='VOD  AR1')
#ax1.legend(['NDVI before tipping','NDVI after tipping','VOD','EVI'],loc='lower left',shadow=True,fancybox=True)
#ax2.legend(['EVI','VOD'],loc='upper right',shadow=True,fancybox=True)
#ax1.grid()

plt.show()

# %%
date1[325]

# %%
# 读取青藏高原NDVI数据
tp_data = np.loadtxt('D:/GEE学习/临界要素数据处理/tpgl_ndvi_82_02.csv', delimiter=',')
tp_ndvi = tp_data[:,1]

# %%
for i in range(len(tp_ndvi)):
    if tp_ndvi[i] == 99999.0:
        tp_ndvi[i] = None
        #am_ndvi[i] = 0

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot((tp_ndvi)/(10000),'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='NDVI',
       title='NDVI in 1982-2022')
ax1.grid()

plt.show()

# %%
len(tp_ndvi)

# %%
# 把数据制作为月合成的显示一下
tp_ndvi_monthly = np.zeros(471)
for i in range(471):
    if (i+1)*30<len(tp_ndvi):
        tp_ndvi_monthly[i] = np.nanmean(tp_ndvi[i*30:(i+1)*30])
    else:
        tp_ndvi_monthly[i] = np.nanmean(tp_ndvi[i*30:len(tp_ndvi)-1])

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot((tp_ndvi_monthly)/(10000),'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='NDVI',
       title='NDVI in 1982-2019')
ax1.set_ylim(-0.05,0.3)
ax1.grid()

plt.show()

# %%
# 使用mask可以给nan加回来
mask = np.isnan(tp_ndvi_monthly)
tp_ndvi_monthly_mask = tp_ndvi_monthly[~mask]

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot((tp_ndvi_monthly_mask/(10000)),'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='NDVI',
       title='NDVI in 1982-2019')
#ax1.set_ylim(-0.05,0.3)
ax1.grid()

plt.show()

# %%
res_tp_ndvi = STL((tp_ndvi_monthly_mask)/(10000), period=12).fit()
res_tp_ndvi.plot()
plt.show()

# %%
res_tp_ndvi_resid = res_tp_ndvi.resid
res_tp_ndvi_trend = res_tp_ndvi.trend
res_tp_ndvi_season = res_tp_ndvi.seasonal
res_tp_ndvi = np.multiply(res_tp_ndvi_resid,res_tp_ndvi_trend+1)
res_tp_ndvi1 = res_tp_ndvi_resid+res_tp_ndvi_trend+res_tp_ndvi_season

# %%
len(res_tp_ndvi_resid)

# %%
tp_date1 = pd.date_range('19600101','20230101',freq='M')
tp_date2 = pd.date_range('19920101','20210201',freq='M')
tp_date3 = pd.date_range('19900101','20210101',freq='M')
tp_date41 = pd.date_range('20010101','20220101',freq='M')
tp_date42 = pd.date_range('20071027','20220101',freq='M')

# %%
tp_ndvi_DFA_alpha = np.zeros(349)
for i in range(349):
    temp = 120+i
    pydfa = fathon.DFA(res_tp_ndvi_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    tp_ndvi_DFA_alpha[i] = H

# %%
tp_evi_DFA_alpha = np.load('tp_evi_DFA_alpha.npy') 
tp_vod_DFA_alpha = np.load('tp_vod_DFA_alpha.npy')
cor_tpgl_ndvi = np.load('cor_tpgl_ndvi.npy')
tpgl_ndvi_DFA_alpha = np.load('tpgl_ndvi_DFA_alpha.npy')

# %%
# Data for plotting
xlabel = tp_date2
fig = plt.figure(figsize=(6,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
ax1.plot(xlabel[0:276],tpgl_ndvi_DFA_alpha[0:276],ls='--',c='royalblue')
ax1.plot(xlabel[276:348],tpgl_ndvi_DFA_alpha[276:348],c='indianred')
ax2.plot(date_evi,tp_evi_DFA_alpha,c='dimgray')
ax1.set(ylabel='NDVI  DFA-'+r'$\alpha$')
ax2.set(ylabel='EVI  DFA-'+r'$\alpha$')
#ax2.plot(date_vod[0:345],tp_vod_DFA_alpha[0:345],c='darkgray')
#ax1.set(xlabel='time series(monthly)', ylabel='NDVI DFA')
#ax2.set( ylabel='EVI/VOD DFA')
#ax1.axvline(x='20150101',color='black')
ax1.set_xlim(tp_date1[264],tp_date1[755])
ax2.set_ylim(0.02,0.13)
#ax1.legend(['NDVI before tipping','NDVI after tipping'],loc='upper left',shadow=True,fancybox=True)
#ax2.legend(['EVI','VOD'],loc='upper right',shadow=True,fancybox=True)
#ax1.grid()

plt.show()

# %%
kendalltau(xlabel[276:348],tpgl_ndvi_DFA_alpha[276:348])

# %%
kendalltau(xlabel[0:276],tpgl_ndvi_DFA_alpha[0:276])

# %%
var_tp_ndvi = np.zeros(len(res_tp_ndvi_resid)-120)
for i in range(len(res_tp_ndvi_resid)-120):
    temp = 120+i
    var_tp_ndvi[i] = np.var(res_tp_ndvi_resid[i:temp]*100)

# %%
# Data for plotting
xlabel = tp_date2
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,var_tp_ndvi,'#000000',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='VAR',
       title='VAR(ndvi in am) in 1981-2019')
ax1.set_xlim(tp_date1[0],tp_date1[755])
ax1.grid()
plt.show()

# %%
cor_tp_ndvi = np.zeros(len(res_tp_ndvi_resid)-120)
for i in range(len(res_tp_ndvi_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_tp_ndvi[i] = np.corrcoef(res_tp_ndvi_resid[month_start:month_end-1],res_tp_ndvi_resid[month_start+1:month_end])[0,1]

# %%
cor_tp_evi = np.load('cor_tp_evi.npy') 
cor_tp_vod = np.load('cor_tp_vod.npy')

# %%
# Data for plotting
xlabel = tp_date2
fig = plt.figure(figsize=(6,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
ax1.plot(xlabel[0:296],cor_tpgl_ndvi[0:296],ls='--',c='royalblue')
ax1.plot(xlabel[296:348],cor_tpgl_ndvi[296:348],c='indianred')
ax2.plot(date_evi,cor_tp_evi,c='dimgray')
#ax2.plot(date_vod[60:345],cor_tp_vod[60:345],c='darkgray')
#ax1.set(xlabel='time series', ylabel='NDVI AR1')
#ax2.set(ylabel='EVI/VOD AR1')
#ax1.axvline(x='20140101',color='black')
ax1.set(ylabel='NDVI  AR1')
ax2.set(ylabel='EVI  AR1')
ax1.set_xlim(tp_date1[264],tp_date1[755])
ax2.set_ylim(0.10,0.36)
#ax1.legend(['NDVI before tipping','NDVI after tipping'],loc='upper left',shadow=True,fancybox=True)
#ax2.legend(['EVI','VOD'],loc='upper right',shadow=True,fancybox=True)
#ax1.grid()

plt.show()

# %%
kendalltau(xlabel[296:348],cor_tpgl_ndvi[296:348])

# %%
kendalltau(xlabel[0:296],cor_tpgl_ndvi[0:296])

# %%
# 读取青藏高原NDVI数据
tpfe_data = np.loadtxt('D:/GEE学习/临界要素数据处理/tpfe_ndvi_82_22.csv', delimiter=',')
tpfe_ndvi = tpfe_data[:]

# %%
for i in range(len(tpfe_ndvi)):
    if tpfe_ndvi[i] == 99999.0:
        tpfe_ndvi[i] = None
        #am_ndvi[i] = 0

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot((tpfe_ndvi)/(10000),'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='NDVI',
       title='NDVI in 1982-2022')
ax1.grid()

plt.show()

# %%
# 把数据制作为月合成的显示一下
tpfe_ndvi_monthly = np.zeros(471)
for i in range(471):
    if (i+1)*30<len(tpfe_ndvi):
        tpfe_ndvi_monthly[i] = np.nanmean(tpfe_ndvi[i*30:(i+1)*30])
    else:
        tpfe_ndvi_monthly[i] = np.nanmean(tpfe_ndvi[i*30:len(tpfe_ndvi)-1])

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot((tpfe_ndvi_monthly)/(10000),'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='NDVI',
       title='NDVI in 1982-2019')
ax1.set_ylim(0,0.4)
ax1.grid()

plt.show()

# %%
# 使用mask可以给nan加回来
mask = np.isnan(tpfe_ndvi_monthly)
tpfe_ndvi_monthly_mask = tpfe_ndvi_monthly[~mask]

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot((tpfe_ndvi_monthly_mask)/(10000),'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='NDVI',
       title='NDVI in 1982-2019')
ax1.set_ylim(0,0.4)
ax1.grid()

plt.show()

# %%
res_tpfe_ndvi = STL((tpfe_ndvi_monthly_mask)/(10000), period=12).fit()
res_tpfe_ndvi.plot()
plt.show()

# %%
res_tpfe_ndvi_resid = res_tpfe_ndvi.resid

# %%
tpfe_ndvi_DFA_alpha = np.zeros(349)
for i in range(349):
    temp = 120+i
    pydfa = fathon.DFA(res_tpfe_ndvi_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    tpfe_ndvi_DFA_alpha[i] = H

# %%
# Data for plotting
xlabel = tp_date2
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,tpfe_ndvi_DFA_alpha,'#000000')
ax1.set(xlabel='time series (month)', ylabel='am_ndvi_DFA',
       title='tpfe_ndvi_DFA in 1980-2020')
ax1.set_xlim(tp_date1[0],tp_date1[755])
ax1.grid()

plt.show()

# %%
var_tpfe_ndvi = np.zeros(len(res_tpfe_ndvi_resid)-120)
for i in range(len(res_tpfe_ndvi_resid)-120):
    temp = 120+i
    var_tpfe_ndvi[i] = np.var(res_tpfe_ndvi_resid[i:temp]*100)

# %%
# Data for plotting
xlabel = tp_date2
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,var_tpfe_ndvi,'#000000',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='VAR',
       title='VAR(ndvi in am) in 1981-2019')
ax1.set_xlim(tp_date1[0],tp_date1[755])
ax1.grid()
plt.show()

# %%
cor_tpfe_ndvi = np.zeros(len(res_tpfe_ndvi_resid)-120)
for i in range(len(res_tpfe_ndvi_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_tpfe_ndvi[i] = np.corrcoef(res_tpfe_ndvi_resid[month_start:month_end-1],res_tpfe_ndvi_resid[month_start+1:month_end])[0,1]

# %%
np.save('tpfe_ndvi_AR1',cor_tpfe_ndvi)

# %%
# Data for plotting
xlabel = tp_date2
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,cor_tpfe_ndvi,'#000000',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='AR1',
       title='AR1(ndvi in am) in 1981-2019')
ax1.set_xlim(tp_date1[0],tp_date1[755])
ax1.grid()
plt.show()

# %%
# 读取青藏高原NDVI数据
tp_modis_data = np.loadtxt('D:/GEE学习/早期预警信号计算/modis_ndvi_tp.csv', delimiter=',')
tp_modis_ndvi = tp_modis_data

# %%
for i in range(len(tp_ndvi)-1):
    if tp_modis_ndvi[i] == 99999.0:
        tp_modis_ndvi[i] = None
        #am_ndvi[i] = 0

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot((tp_modis_ndvi)/(10000),'#000000',linewidth=1)
ax1.set(xlabel='time series (month)', ylabel='NDVI',
       title='NDVI in 1982-2022')
ax1.grid()

plt.show()

# %%
len(tp_modis_ndvi)

# %%
res_tp_modis_ndvi = STL((tp_modis_ndvi)/(10000), period=23).fit()
res_tp_modis_ndvi.plot()
plt.show()

# %%
res_tp_modis_ndvi_resid = res_tp_modis_ndvi.resid

# %%
tp_modis_ndvi_DFA_alpha = np.zeros(255)
for i in range(255):
    temp = 228+i
    pydfa = fathon.DFA(res_tp_modis_ndvi_resid[i:temp])
    winSizes = fu.linRangeByStep(10,228)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    tp_modis_ndvi_DFA_alpha[i] = H

# %%
# Data for plotting
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(tp_modis_ndvi_DFA_alpha,'#000000')
ax1.set(xlabel='time series (month)', ylabel='am_ndvi_DFA',
       title='tpfe_ndvi_DFA in 1980-2020')
ax1.grid()

plt.show()

# %%
cor_tp_modis_ndvi = np.zeros(len(res_tp_modis_ndvi_resid)-228)
for i in range(len(res_tp_modis_ndvi_resid)-228):
    temp = 228+i
    month_start = i
    month_end = temp
    cor_tp_modis_ndvi[i] = np.correlate(res_tp_modis_ndvi_resid[month_start:month_end-1]-res_tp_modis_ndvi_resid[month_start:month_end-1].mean(),res_tp_modis_ndvi_resid[month_start+1:month_end]-res_tp_modis_ndvi_resid[month_start+1:month_end].mean())[0]/(res_tp_modis_ndvi_resid[month_start:month_end-1].std()*res_tp_modis_ndvi_resid[month_start+1:month_end].std()*228)

# %%
# Data for plotting
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(cor_tp_modis_ndvi,'#000000')
ax1.set(xlabel='time series (month)', ylabel='am_ndvi_DFA',
       title='tpfe_ndvi_DFA in 1980-2020')
ax1.grid()

plt.show()

# %%


# %%
# 5到20年，60-240个月
windows = np.array([60,120,240])
windows_delay = np.array([0,30,90])

# %%
res_am_ndvi_resid_alpha = np.zeros([405,3])
c = 0
for k in windows:
    for i in range(465-k):
        temp = k+i
        pydfa = fathon.DFA(res_am_ndvi_resid[i:temp])
        winSizes = fu.linRangeByStep(10,k)
        revSeg = True
        polOrd = 3
        n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
        H, H_intercept = pydfa.fitFlucVec()
        res_am_ndvi_resid_alpha[i+windows_delay[c],c] = H
    c = c+1

# %%
cor_am_ndvi2 = np.zeros([405,3])
c = 0
for k in windows:
    for i in range(len(res_am_ndvi_resid)-k):
        temp = k+i
        month_start = i
        month_end = temp
        cor_am_ndvi2[i+windows_delay[c],c] = np.corrcoef(res_am_ndvi_resid[month_start:month_end-1],res_am_ndvi_resid[month_start+1:month_end])[0,1]
    c = c+1

# %%
cor_am_ndvi2 = np.where(cor_am_ndvi2,cor_am_ndvi2,np.nan)
res_am_ndvi_resid_alpha = np.where(res_am_ndvi_resid_alpha,res_am_ndvi_resid_alpha,np.nan)

# %%
# Data for plotting
xlabel = date1
fig = plt.figure(figsize=(6,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
#ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集


#ax2.plot(xlabel,la_ar1mean,'royalblue',label='land-atmosphere coupling')


ax1.plot(xlabel[324:729],res_am_ndvi_resid_alpha[:,0],label='60 months window')
ax1.plot(xlabel[324:729],res_am_ndvi_resid_alpha[:,1],label='120 months window')
ax1.plot(xlabel[324:729],res_am_ndvi_resid_alpha[:,2],label='240 months window')


ax1.set(ylabel='DFA-'+r'$\alpha$')
#ax1.set_xlim(date1[164],date1[743])
#ax1.set_ylim(-0.2,0.4)
#ax2.set_ylim(-0.10,0.40)
#ax1.grid()
ax1.legend(loc = 'upper left')

plt.show()

# %%
# Data for plotting
xlabel = date1
fig = plt.figure(figsize=(6,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
#ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集


#ax2.plot(xlabel,la_ar1mean,'royalblue',label='land-atmosphere coupling')


ax1.plot(xlabel[384:729],cor_am_ndvi2[60:405,0],label='60 months window')
ax1.plot(xlabel[384:729],cor_am_ndvi2[60:405,1],label='120 months window')
ax1.plot(xlabel[384:729],cor_am_ndvi2[60:405,2],label='240 months window')


ax1.set(ylabel='AR1')
#ax1.set_xlim(date1[164],date1[743])
#ax1.set_ylim(-0.2,0.4)
#ax2.set_ylim(-0.10,0.40)
#ax1.grid()
ax1.legend(loc = 'upper left')

plt.show()

# %%

