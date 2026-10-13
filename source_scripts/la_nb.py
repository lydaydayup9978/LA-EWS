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
nb_tci_path = r'D:/GEE学习/20250507计算陆气耦合指数/results/'
year_num = 1971
year = str(year_num)
# 打开npy文件
nb_tci = np.load(nb_tci_path+ year + '.npy')

# %%
np.shape(nb_tci)

# %%
nb_mask_path = 'D:/GEE学习/EWS空间数据处理/NB针叶林掩膜/nb_lamask/nb_lamask3_2.tif'
# 读取tiff文件
nb_mask = tf.imread(nb_mask_path)
print(nb_mask.shape)

# %%
nb_mask_array = np.array(nb_mask)
nb_mask_array = np.where(nb_mask_array,nb_mask_array,np.nan)

# %%
plt.imshow(nb_tci[5,:,:])

# %%
plt.imshow(nb_mask_array)

# %%
date1 = pd.date_range('19830101','20240101',freq='M')
date2 = pd.date_range('19710101','20231201',freq='M')
date3 = pd.date_range('19780101','20070101',freq='M')
date31 = pd.date_range('19920101','20210101',freq='M')
date32 = pd.date_range('19850101','20140101',freq='M')
date4 = pd.date_range('19810101','20230101',freq='M')
date5 = pd.date_range('19920101','20190301',freq='M')
date6 = pd.date_range('19600101','20220101',freq='M')

# %%
# 一共53年的数据
# 存储53年的数据
nb_tci_54 = np.zeros(624)
for i in range(52):
    nb_tci_path = r'D:/GEE学习/20250507计算陆气耦合指数/result_tci2/'
    #ca_tci_path = 'D:/GEE学习/陆气耦合计算/ca_TCI/'
    year_num = i+1971
    year = str(year_num)
    # 打开npy文件
    data2 = np.load(nb_tci_path+ year + '.npy')
    # mask与数据矩阵对应相乘
    nb_tci = data2*nb_mask_array
    # 求均值（数据本身为月平均数据）
    nb_tcimean_array = np.zeros(len(data2[:,1,1]))
    yearnum = len(data2[:,1,1])
    for j in range(yearnum):
        nb_tcimean_array[j] = np.nanmean(nb_tci[j,:,:])
    nb_tci_54[i*12:(i+1)*12] = nb_tcimean_array

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(nb_tci_54,'#000000',linewidth=1,label='nb_tci')
ax1.set(xlabel='time series (month)', ylabel='TCI',
       title='north boreal TCI in 1970-2023')
#ax1.set_ylim(-0.05,0.6)
ax1.legend()
ax1.grid()

plt.show()

# %%
res_nb_tci = STL(nb_tci_54, period=12, robust=True).fit()
res_nb_tci_resid = res_nb_tci.resid

# %%
res_nb_tci_resid_alpha = np.zeros(504)
for i in range(504):
    temp = 120+i
    pydfa = fathon.DFA(res_nb_tci_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    res_nb_tci_resid_alpha[i] = H

# %%
# 读取ndvi的dfa计算结果
nb_ndvi_DFA_alpha2 = np.load('nb_ndvi_DFA_alpha2.npy')

# Data for plotting
xlabel = date4
xlabel2 = pd.date_range('19830101','20120101',freq='M')
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
ax2.plot(xlabel2,nb_ndvi_DFA_alpha2,'green')
ax1.plot(xlabel,res_nb_tci_resid_alpha,'#DAA520')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='NB_TCI_DFA in 1981-2023')
ax1.set_xlim(date2[0],date2[634])
ax1.grid()

plt.show()

# %%
import statsmodels.tsa.stattools as stattools

# %%
cor_nb_tci = np.zeros(504)
for i in range(len(res_nb_tci_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_nb_tci[i] = np.corrcoef(res_nb_tci_resid[month_start:month_end-1],res_nb_tci_resid[month_start+1:month_end])[0,1]

# %%
# 读取ndvi的dfa计算结果
cor_nb_ndvi = np.load('cor_nb_ndvi.npy')

# Data for plotting
xlabel = date4
xlabel2 = pd.date_range('19780101','20070101',freq='M')
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
ax2.plot(xlabel2,cor_nb_ndvi,'green')
ax1.plot(xlabel,cor_nb_tci,'#DAA520')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='NB_TCI_DFA in 1981-2023')
ax1.set_xlim(date2[0],date2[634])
ax1.grid()

plt.show()

# %%
# 一共62年的数据
# 存储62年的数据
nb_tet_53 = np.zeros(624)
for i in range(52):
    # 陆气耦合为三维数据nc格式文件
    nb_tet_path = r'D:/GEE学习/20250507计算陆气耦合指数/result_tet/'
    year_num = i+1971
    year = str(year_num)
    # 打开npy文件
    data3 = np.load(nb_tet_path+ year + '.npy')
    # mask与数据矩阵对应相乘
    nb_tet = data3*nb_mask_array
    # 求均值（数据本身为月平均数据）
    nb_tetmean_array = np.zeros(len(data3[:,1,1]))
    yearnum = len(data3[:,1,1])
    for j in range(yearnum):
        nb_tetmean_array[j] = np.nanmean(nb_tet[j,:,:])
    nb_tet_53[i*12:(i+1)*12] = nb_tetmean_array

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(nb_tet_53,'#100000',linewidth=1,label='tet')
ax1.set(xlabel='time series (month)', ylabel='tet',
           title='NB tet in 1971-20231')
#ax1.set_ylim(-0.05,0.6)
ax1.legend()
ax1.grid()

plt.show()

# %%
res_nb_tet = STL(nb_tet_53, period=12, robust=True).fit()
res_nb_tet_resid = res_nb_tet.resid

# %%
res_nb_tet_resid_alpha = np.zeros(504)
for i in range(504):
    temp = 120+i
    pydfa = fathon.DFA(res_nb_tet_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    res_nb_tet_resid_alpha[i] = H

# %%
# 读取ndvi的dfa计算结果
nb_ndvi_DFA_alpha2 = np.load('nb_ndvi_DFA_alpha2.npy')

# Data for plotting
xlabel = date4
xlabel2 = pd.date_range('19820101','20110101',freq='M')
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
ax2.plot(xlabel2,nb_ndvi_DFA_alpha2,'green')
ax1.plot(xlabel,res_nb_tet_resid_alpha,'#DAA520')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='NB_TCI_DFA in 1981-2023')
ax1.set_xlim(date2[0],date2[634])
ax1.grid()

plt.show()

# %%
cor_nb_tet = np.zeros(504)
for i in range(len(res_nb_tet_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_nb_tet[i] = np.corrcoef(res_nb_tet_resid[month_start:month_end-1],res_nb_tet_resid[month_start+1:month_end])[0,1]

# %%
# 读取ndvi的dfa计算结果
cor_nb_ndvi = np.load('cor_nb_ndvi.npy')

# Data for plotting
xlabel = date4
xlabel2 = pd.date_range('19810101','20100101',freq='M')
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
ax2.plot(xlabel2,cor_nb_ndvi,'green')
ax1.plot(xlabel,cor_nb_tet,'#DAA520')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='NB_TET_AR1 in 1981-2023')
ax1.set_xlim(date2[0],date2[634])
ax1.grid()

plt.show()

# %%
# 一共62年的数据
# 存储62年的数据
ca_pai_62 = np.zeros(624)
for i in range(52):
    # 导入am地区陆气耦合数据
    # 陆气耦合为三维数据nc格式文件
    ca_pai_path = 'F:/ERA5/0512NB地区PAI指数/'
    year_num = i+1971
    year = str(year_num)
    # 打开npy文件
    data3 = np.load(ca_pai_path+ year + '.npy')
    # mask与数据矩阵对应相乘
    ca_pai = data3*nb_mask_array
    ca_pai[np.where(ca_pai==-32767)] = np.nan
    # 求均值（数据本身为月平均数据）
    ca_paimean_array = np.zeros(len(data3[:,1,1]))
    yearnum = len(data3[:,1,1])
    for j in range(yearnum):
        ca_paimean_array[j] = np.nanmean(ca_pai[j,:,:])
    # 当场转化为月平均数据
    monthly = np.zeros(12)
    for k in range(12):
        if (k+1)*30<len(ca_paimean_array):
            monthly[k] = np.nanmean(ca_paimean_array[k*30:(k+1)*30])
        else:
            monthly[k] = np.nanmean(ca_paimean_array[k*30:len(ca_paimean_array)-1])
    ca_pai_62[i*12:(i+1)*12] = monthly

# %%
# 输出为图像
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(ca_pai_62,'#100000',linewidth=1,label='ca_pai')
ax1.set(xlabel='time series (month)', ylabel='pai',
       title='CA pai in 1960-2021')
#ax1.set_ylim(-0.05,0.6)
ax1.legend()
ax1.grid()

plt.show()

# %%
res_ca_pai = STL(ca_pai_62, period=12, robust=True).fit()
res_ca_pai_resid = res_ca_pai.resid

# %%
res_ca_pai_resid_alpha = np.zeros(504)
for i in range(504):
    temp = 120+i
    pydfa = fathon.DFA(res_ca_pai_resid[i:temp])
    winSizes = fu.linRangeByStep(10,120)
    revSeg = True
    polOrd = 3
    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()
    res_ca_pai_resid_alpha[i] = H

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,res_ca_pai_resid_alpha,'#000000')
ax1.set(xlabel='time series (month)', ylabel='DFA alpha',
       title='ca_pai_DFA in 1960-2021')
ax1.set_xlim(date2[0],date2[634])
ax1.grid()

plt.show()

# %%
cor_nb_pai = np.zeros(504)
for i in range(len(res_ca_pai_resid)-120):
    temp = 120+i
    month_start = i
    month_end = temp
    cor_nb_pai[i] = np.corrcoef(res_ca_pai_resid[month_start:month_end-1],res_ca_pai_resid[month_start+1:month_end])[0,1]

# %%
# Data for plotting
xlabel = date4
fig = plt.figure(figsize=(15,4)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax1.plot(xlabel,cor_nb_pai,'#DAA520',linewidth=2.0)
ax1.set(xlabel='time series', ylabel='AR1',
       title='AR1(pai in ca) in 1960-2019')
ax1.set_xlim(date2[0],date2[634])
ax1.grid()
plt.show()

# %%
la_dfamean = res_nb_tci_resid_alpha+res_nb_tet_resid_alpha+res_ca_pai_resid_alpha
la_dfamean = la_dfamean/3

# %%
# 三合一陆气耦合指数
la_dfamat = np.array((res_ca_pai_resid_alpha,
                      res_nb_tet_resid_alpha,res_nb_tci_resid_alpha))
# 求最大最小值
la_dfamat_max = np.max(la_dfamat,axis=0)
la_dfamat_min = np.min(la_dfamat,axis=0)

# %%
# 读取ndvi的dfa计算结果
nb_ndvi_DFA_alpha2 = np.load('nb_ndvi_DFA_alpha2.npy')

# Data for plotting
xlabel = date4
xlabel2 = pd.date_range('19820101','20110101',freq='M')
xlabel3 = pd.date_range('19920101','20210101',freq='M')
fig = plt.figure(figsize=(10,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
ax2.plot(xlabel2,nb_ndvi_DFA_alpha2,'green',label='ndvi 10 years early')
ax2.plot(xlabel3,nb_ndvi_DFA_alpha2,ls='--',c='green',label='ndvi', alpha=0.2)
ax1.plot(xlabel,la_dfamean,'royalblue',label='land-atmosphere coupling')
ax1.fill_between(xlabel, la_dfamat_min, la_dfamat_max, facecolor='royalblue', alpha=0.3)
ax1.set(xlabel='time series (month)', ylabel='LA DFA-'+r'$\alpha$')
#ax1.set_xlim(date2[0],date2[634])
ax1.set_xlim(date6[164],date6[743])
ax1.set_ylim(0.02,0.10)
ax2.set_ylim(0,0.3)
ax2.set(ylabel='NDVI DFA-'+r'$\alpha$')
ax1.legend(loc = 'upper left',shadow=True,fancybox=True)
ax2.legend(loc='upper right',shadow=True,fancybox=True)

plt.show()

# %%
la_ar1mean = cor_nb_tci+cor_nb_tet+cor_nb_pai
la_ar1mean = la_ar1mean/3

# %%
# 三合一陆气耦合指数
la_ar1mat = np.array((cor_nb_tci,
                      cor_nb_tet,cor_nb_pai))
# 求最大最小值
la_ar1mat_max = np.max(la_ar1mat,axis=0)
la_ar1mat_min = np.min(la_ar1mat,axis=0)

# %%
# Data for plotting
xlabel = date4
xlabel2 = pd.date_range('19820101','20110101',freq='M')
xlabel3 = pd.date_range('19920101','20210101',freq='M')
fig = plt.figure(figsize=(10,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
ax2.plot(xlabel2,cor_nb_ndvi,'green',label='ndvi 10 years early')
ax2.plot(xlabel3,cor_nb_ndvi,ls='--',c='green',label='ndvi', alpha=0.2)
ax1.plot(xlabel,la_ar1mean,'royalblue',label='land-atmosphere coupling')
ax1.fill_between(xlabel, la_ar1mat_min, la_ar1mat_max, facecolor='royalblue', alpha=0.3)
ax1.set(xlabel='time series (month)', ylabel='LA AR1')
#ax1.set_xlim(date2[0],date2[634])
ax1.set_xlim(date6[164],date6[743])
ax1.set_ylim(-0.1,0.3)
ax2.set_ylim(-0.2,0.9)
ax2.set(ylabel='NDVI AR1')
ax1.legend(loc = 'upper left',shadow=True,fancybox=True)
ax2.legend(loc='upper right',shadow=True,fancybox=True)

plt.show()

# %%
nb_ndvi_DFA_alpha2_clip = nb_ndvi_DFA_alpha2
n = len(nb_ndvi_DFA_alpha2_clip)
time_lag = range(len(la_dfamean)-n)


# r_lag存储不同时滞下的相关系数
r_lag_dfa = np.zeros(len(time_lag))
# 下面开始循环计算
count = 504
for i in time_lag:
    temp = np.corrcoef(la_dfamean[count-i-n:count-i],nb_ndvi_DFA_alpha2_clip)
    r_lag_dfa[i] = temp[0,1]

# %%
# 领先时间柱形图
# 改为每个月显示,一个柱形图，按相关性排序
n_lag = np.floor(len(time_lag)/12)
dfa_lag_monthly = np.zeros(n_lag.astype(int))
for i in range(n_lag.astype(int)-1):
    dfa_lag_monthly[i] = r_lag_dfa[(i+1)*12]
# 为序列排序
dfa_lag_monthly_sort = np.argsort(dfa_lag_monthly)
dfa_lag_monthly_top10 = dfa_lag_monthly_sort[-10:]
#ar1_lag_monthly_sort = np.flipud(ar1_lag_monthly_sort)
# 画图
fig, ax = plt.subplots()
hbars = ax.barh(range(len(dfa_lag_monthly_top10)), dfa_lag_monthly[dfa_lag_monthly_top10], align='center', color='#D47500')#burlywood
ax.set_yticks(range(len(dfa_lag_monthly_top10)), (dfa_lag_monthly_top10+1))
ax.set_xlabel('DFA Corrcoef')
ax.set_ylabel('ahead of time (year)')
ax.set_xlim(right=1.0)
ax.bar_label(hbars, labels=[f'{e:.2f}' for e in dfa_lag_monthly[dfa_lag_monthly_top10]],
             padding=8, color='black', fontsize=10)
# 列出前十名

# %%
cor_nb_ndvi_clip = cor_nb_ndvi
n = len(cor_nb_ndvi_clip)
time_lag = range(len(la_ar1mean)-n)


# r_lag存储不同时滞下的相关系数
r_lag_ar1 = np.zeros(len(time_lag))
# 下面开始循环计算
count = 504
for i in time_lag:
    temp = np.corrcoef(la_ar1mean[count-i-n:count-i],cor_nb_ndvi_clip)
    r_lag_ar1[i] = temp[0,1]

# %%
# 领先时间柱形图
# 改为每个月显示,一个柱形图，按相关性排序
n_lag = np.floor(len(time_lag)/12)
ar1_lag_monthly = np.zeros(n_lag.astype(int))
for i in range(n_lag.astype(int)-1):
    ar1_lag_monthly[i] = r_lag_ar1[(i+1)*12]
# 为序列排序
ar1_lag_monthly_sort = np.argsort(ar1_lag_monthly)
ar1_lag_monthly_top10 = ar1_lag_monthly_sort[-10:]
#ar1_lag_monthly_sort = np.flipud(ar1_lag_monthly_sort)
# 画图
fig, ax = plt.subplots()
hbars = ax.barh(range(len(ar1_lag_monthly_top10)), ar1_lag_monthly[ar1_lag_monthly_top10], align='center', color='#D47500')#burlywood
ax.set_yticks(range(len(ar1_lag_monthly_top10)), (ar1_lag_monthly_top10+1))
ax.set_xlabel('AR1 Corrcoef')
ax.set_ylabel('ahead of time (year)')
ax.set_xlim(right=1.1)
ax.bar_label(hbars, labels=[f'{e:.2f}' for e in ar1_lag_monthly[ar1_lag_monthly_top10]],
             padding=8, color='black', fontsize=10)
# 列出前十名

# %%
from pyEDM import CCM

# %%
# NB
# 假设数据已加载为带时间索引的Series
nb_la_dfa = la_dfamean[24:360]
nb_ndvi_dfa = nb_ndvi_DFA_alpha2[0:336]

nb_la_cor = la_ar1mean[12:348]
nb_ndvi_cor = cor_nb_ndvi[0:336]

# 创建带时间戳的DataFrame
nb_df_dfa = pd.DataFrame({
    'LA': nb_la_dfa,
    'NDVI': nb_ndvi_dfa
})

nb_df_cor = pd.DataFrame({
    'LA': nb_la_cor,
    'NDVI': nb_ndvi_cor
})

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

nb_df_dfa[['LA_scaled', 'NDVI_scaled']] = scaler.fit_transform(nb_df_dfa[['LA', 'NDVI']])
nb_df_cor[['LA_scaled', 'NDVI_scaled']] = scaler.fit_transform(nb_df_cor[['LA', 'NDVI']])

# %%
# 1. 确定最优时间延迟tau
from pyEDM import PredictInterval
tau_output = PredictInterval(
    dataFrame=nb_df_cor,
    columns='NDVI_scaled',
    target='LA_scaled',
    lib="1 200",
    pred="201 336",
    showPlot=True
)

# %%
from pyEDM import EmbedDimension
# 假设数据总长度为324行（根据你的数据）
# 将前200行作为训练集，后124行作为预测集
# 2. 确定最优嵌入维度E
ed_output = EmbedDimension(
    dataFrame=nb_df_cor,
    columns='NDVI_scaled',
    target='LA_scaled',
    lib="1 200",
    pred="201 336",
    maxE=10,
    tau=5,          # 使用上一步得到的tau
    showPlot=True
)

# %%
# 关键参数说明：
# - E: 嵌入维度（通常通过FNN或互信息法确定）
# - tau: 时间滞后（常取1或自相关首次穿越0的时间）
# - lib_lengths: 用于测试的库长度范围

# 定义参数（示例值，需根据数据调整）
params = {
    "E": 8,                # 嵌入维度
    "tau": 5,              # 时间延迟
    "lib_sizes": "10 200 5",  # 库大小范围（字符串格式）
    "sample": 100,         # 每个库大小的抽样次数 > 0
    "random_state": 42     # 随机种子（可选）
}

# %%
# 检验 LA 是否影响 NDVI
ccm_la2ndvi = CCM(
    dataFrame=nb_df_cor,
    columns='LA_scaled',
    target='NDVI_scaled',
    E=params["E"],
    tau=params["tau"],
    libSizes=params["lib_sizes"],  # 必须为字符串
    sample=params["sample"],       # 必须 > 0
    random=params["random_state"], # 控制可重复性
    includeData=True,  # 关键参数：确保包含数据
    pathIn='./',
    dataFile=''
)

# %%
# 提取方向性rho值
lib_size = ccm_la2ndvi["LibMeans"]["LibSize"]  # 共享LibSize列
rho_la2ndvi = ccm_la2ndvi["LibMeans"]["LA_scaled:NDVI_scaled"]
rho_ndvi2la = ccm_la2ndvi["LibMeans"]["NDVI_scaled:LA_scaled"]

# 绘制双向CCM结果
plt.figure(figsize=(10,6))
plt.plot(lib_size, rho_la2ndvi, 'o-', label='LA→NDVI')
plt.plot(lib_size, rho_ndvi2la, 's--', label='NDVI→LA')
plt.xlabel('Library Size', fontsize=12)
plt.ylabel('Cross-Mapped Correlation (ρ)', fontsize=12)
plt.title('NB-AR1')
plt.legend()
plt.grid(True)
plt.show()

# %%
def find_turning_points(series, look_back=1, look_forward=1):
    peaks, valleys = [], []
    for i in range(look_back, len(series)-look_forward):
        # 扩展比较范围
        prev = series.iloc[i-look_back:i].max()
        curr = series.iloc[i]
        next_ = series.iloc[i+1:i+look_forward+1].max()
        
        prev2 = series.iloc[i-look_back:i].min()
        next2_ = series.iloc[i+1:i+look_forward+1].min()
        
        if prev < curr and next_ < curr:
            peaks.append(series.index[i])
        elif prev2 > curr and next2_ > curr:
            valleys.append(series.index[i])
    return peaks, valleys

# %%
# LA-EWS的最高点到NDVI的转折点的时间计算
# 生成示例数据
dates = pd.date_range(start="1980-01-01", periods=504, freq="M")
#ts = pd.Series(la_dfamean[0:504], index=dates)
ts = pd.Series(la_ar1mean[0:504], index=dates)


# 平滑处理（可选）
window_size = 5
smoothed = ts.rolling(window=window_size, center=True).mean().dropna()

# 使用前后3个点的范围检测 (共7点窗口)
peaks, valleys = find_turning_points(smoothed, look_back=36, look_forward=36)

# 可视化
plt.figure(figsize=(12, 6))
ts.plot(label="Original", alpha=0.5)
smoothed.plot(label="Smoothed", color='orange')
plt.scatter(peaks, smoothed.loc[peaks], color='red', label="Peaks")
plt.scatter(valleys, smoothed.loc[valleys], color='green', label="Valleys")
plt.legend()
plt.title("Turning Points Detection")
plt.show()

# %%
peaks

# %%
# LA-EWS的最高点到NDVI的转折点的时间计算
# 生成示例数据
dates2 = pd.date_range(start="1992-01-01", periods=len(nb_ndvi_DFA_alpha2), freq="M")
#ts2 = pd.Series(nb_ndvi_DFA_alpha2, index=dates2)
ts2 = pd.Series(cor_nb_ndvi, index=dates2)

# 平滑处理（可选）
window_size = 5
smoothed2 = ts2.rolling(window=window_size, center=True).mean().dropna()

# 使用前后3个点的范围检测 (共7点窗口)
peaks2, valleys2 = find_turning_points(smoothed2, look_back=36, look_forward=36)

# 可视化
plt.figure(figsize=(12, 6))
ts2.plot(label="Original", alpha=0.5)
smoothed2.plot(label="Smoothed", color='orange')
plt.scatter(peaks2, smoothed2.loc[peaks2], color='red', label="Peaks")
plt.scatter(valleys2, smoothed2.loc[valleys2], color='green', label="Valleys")
plt.legend()
plt.title("Turning Points Detection")
plt.show()

# %%
valleys2

# %%
peaks2

# %%
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# 绘制第一个子图 (LA-EWS)
line1, = ax1.plot(smoothed, color='blue', label='LA-EWS')
peak_scatter = ax1.scatter(peaks[0:2], smoothed.loc[peaks[0:2]], color='red', label='Peaks')
valley_scatter = ax1.scatter(valleys[2], smoothed.loc[valleys[2]], color='green', label='Valleys')

# 绘制第二个子图 (NDVI)
line2, = ax2.plot(smoothed2, color='orange', label='NDVI-EWS')
ax2.scatter(peaks2[1], smoothed2.loc[peaks2[1]], color='red')
ax2.scatter(valleys2[1], smoothed2.loc[valleys2[1]], color='green')

# 创建统一图例
handles = [line1, line2, peak_scatter, valley_scatter]
labels = [h.get_label() for h in handles]
fig.legend(handles=handles, 
          labels=labels,
          loc='upper center',
          ncol=4,  # 横向排列四个图例项
          bbox_to_anchor=(0.5, 0.98),  # 将图例放在图形上方
          frameon=False)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为上方图例留出空间

# 添加子图标题（可选）
ax2.set_xlabel("Time")

plt.show()

# %%
# 5到20年，60-240个月
windows = np.array([60,120,240])
windows_delay = np.array([0,30,90])

# %%
res_ca_tci_resid_alpha = np.zeros([564,3])
c = 0
for k in windows:
    for i in range(624-k):
        temp = k+i
        pydfa = fathon.DFA(res_ca_tci_resid[i:temp])
        winSizes = fu.linRangeByStep(10,k)
        revSeg = True
        polOrd = 3
        n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
        H, H_intercept = pydfa.fitFlucVec()
        res_ca_tci_resid_alpha[i+windows_delay[c],c] = H
    c = c+1

# %%
res_ca_tci_resid_alpha = np.where(res_ca_tci_resid_alpha,res_ca_tci_resid_alpha,np.nan)

# %%
cor_ca_tci2 = np.zeros([564,3])
c = 0
for k in windows:
    for i in range(len(res_ca_tci_resid)-k):
        temp = k+i
        month_start = i
        month_end = temp
        cor_ca_tci2[i+windows_delay[c],c] = np.corrcoef(res_ca_tci_resid[month_start:month_end-1],res_ca_tci_resid[month_start+1:month_end])[0,1]
    c = c+1

# %%
cor_ca_tci2 = np.where(cor_ca_tci2,cor_ca_tci2,np.nan)

# %%
res_ca_tet_resid_alpha = np.zeros([564,3])
c = 0
for k in windows:
    for i in range(624-k):
        temp = k+i
        pydfa = fathon.DFA(res_ca_tet_resid[i:temp])
        winSizes = fu.linRangeByStep(10,k)
        revSeg = True
        polOrd = 3
        n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
        H, H_intercept = pydfa.fitFlucVec()
        res_ca_tet_resid_alpha[i+windows_delay[c],c] = H
    c = c+1
    
res_ca_tet_resid_alpha = np.where(res_ca_tet_resid_alpha,res_ca_tet_resid_alpha,np.nan)

# %%
cor_ca_tet2 = np.zeros([564,3])
c = 0
for k in windows:
    for i in range(len(res_ca_tet_resid)-k):
        temp = k+i
        month_start = i
        month_end = temp
        cor_ca_tet2[i+windows_delay[c],c] = np.corrcoef(res_ca_tet_resid[month_start:month_end-1],res_ca_tet_resid[month_start+1:month_end])[0,1]
    c = c+1
    
    
cor_ca_tet2 = np.where(cor_ca_tet2,cor_ca_tet2,np.nan)

# %%
res_ca_pai_resid_alpha = np.zeros([564,3])
c = 0
for k in windows:
    for i in range(624-k):
        temp = k+i
        pydfa = fathon.DFA(res_ca_pai_resid[i:temp])
        winSizes = fu.linRangeByStep(10,k)
        revSeg = True
        polOrd = 3
        n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
        H, H_intercept = pydfa.fitFlucVec()
        res_ca_pai_resid_alpha[i+windows_delay[c],c] = H
    c = c+1
    
res_ca_pai_resid_alpha = np.where(res_ca_pai_resid_alpha,res_ca_pai_resid_alpha,np.nan)

# %%
cor_ca_pai2 = np.zeros([564,3])
c = 0
for k in windows:
    for i in range(len(res_ca_pai_resid)-k):
        temp = k+i
        month_start = i
        month_end = temp
        cor_ca_pai2[i+windows_delay[c],c] = np.corrcoef(res_ca_pai_resid[month_start:month_end-1],res_ca_pai_resid[month_start+1:month_end])[0,1]
    c = c+1
    
    
cor_ca_pai2 = np.where(cor_ca_pai2,cor_ca_pai2,np.nan)

# %%
ca_ar1 = (cor_ca_tet2+cor_ca_pai2)/2
ca_dfa = (res_ca_pai_resid_alpha+res_ca_tet_resid_alpha)/2

# %%
# Data for plotting
xlabel = date2
fig = plt.figure(figsize=(6,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
#ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集


#ax2.plot(xlabel,la_ar1mean,'royalblue',label='land-atmosphere coupling')


ax1.plot(xlabel[60:624],ca_dfa[:,0],label='60 months window')
ax1.plot(xlabel[60:624],ca_dfa[:,1],label='120 months window')
ax1.plot(xlabel[60:624],ca_dfa[:,2],label='240 months window')


ax1.set(ylabel='DFA-'+r'$\alpha$')
#ax1.set_xlim(date1[164],date1[743])
#ax1.set_ylim(-0.2,0.4)
#ax2.set_ylim(-0.10,0.40)
#ax1.grid()
ax1.legend(loc = 'upper left')

plt.show()

# %%
# Data for plotting
xlabel = date2
fig = plt.figure(figsize=(6,5)) # 设置画板大小
ax1 = fig.add_subplot(1,1,1) # 设置画板格式——一个图
#ax2 = ax1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集


#ax2.plot(xlabel,la_ar1mean,'royalblue',label='land-atmosphere coupling')


ax1.plot(xlabel[60:624],ca_ar1[:,0],label='60 months window')
ax1.plot(xlabel[60:624],ca_ar1[:,1],label='120 months window')
ax1.plot(xlabel[60:624],ca_ar1[:,2],label='240 months window')


ax1.set(ylabel='AR1')
#ax1.set_xlim(date1[164],date1[743])
#ax1.set_ylim(-0.2,0.4)
#ax2.set_ylim(-0.10,0.40)
#ax1.grid()
ax1.legend(loc = 'upper left')

plt.show()

# %%

