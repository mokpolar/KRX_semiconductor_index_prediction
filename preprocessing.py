#%%
# library use
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from datetime import datetime
import calendar
from math import sin, cos, sqrt, atan2, radians
from folium import FeatureGroup, LayerControl, Map, Marker
from folium.plugins import HeatMap
import matplotlib.dates as mdates
import matplotlib as mpl
from datetime import timedelta
import datetime as dt
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', -1)
plt.style.use('fivethirtyeight')
import folium
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle
%config InlineBackend.figure_format = 'retina'
from sklearn.preprocessing import Imputer

#%%
!ls CSV/* > list

#%%
!cat list


#%%
krx_index = pd.read_csv('CSV/KRXsemiconductor_wd.csv')
hynix = pd.read_csv('CSV/SKhynix_wd.csv')
tsmc = pd.read_csv('CSV/TSMC.csv')
phil = pd.read_csv('CSV/Philadelphia.csv')

#%%

fb = pd.read_csv('CSV/Facebook.csv')
google = pd.read_csv('CSV/Google.csv')
intel = pd.read_csv('CSV/Intel.csv')
micron = pd.read_csv('CSV/Micron.csv')
nv = pd.read_csv('CSV/Nvidia.csv')

# %%
krx_index.shape

#%%
krx_index.head()
#%%
# 데이터 형태 확인, -> 기존에 확인하면 object이고 숫자에 string도 들어있음
krx_index.info()

# %%
# 날짜 -> datetime, object에 dot 들어있는거 날리고 숫자로 변경

krx_index['date'] = pd.to_datetime(krx_index['date'])
krx_index['close'] = krx_index['close'].map(lambda x: re.sub("\D", "", x))
krx_index['volume'] = krx_index['volume'].map(lambda x: re.sub("\D", "", x))

#%%

krx_index['close'] = pd.to_numeric(krx_index['close'])
krx_index['volume'] = pd.to_numeric(krx_index['volume'])


#%%

hynix['hy_date'] = pd.to_datetime(hynix['date'])
hynix['hy_close'] = hynix['close'].map(lambda x: re.sub("\D", "", x))
hynix['hy_volume'] = hynix['volume'].map(lambda x: re.sub("\D", "", x))

#%%
#%%

hynix['hy_close'] = pd.to_numeric(hynix['hy_close'])
hynix['hy_volume'] = pd.to_numeric(hynix['hy_volume'])

#%%
tsmc.head()
# %%
tsmc['tsmc_date'] = pd.to_datetime(tsmc['DATE'])
#tsmc['tsmc_close'] = tsmc['CLOSE'].map(lambda x: re.sub("\D", "", x))
#tsmc['tsmc_volume'] = tsmc['VOLUME'].map(lambda x: re.sub("\D", "", x))
tsmc['tsmc_close'] = tsmc['CLOSE']
tsmc['tsmc_volume'] = tsmc['VOLUME']
#%%

tsmc['tsmc_close'] = pd.to_numeric(tsmc['tsmc_close'])
tsmc['tsmc_volume'] = pd.to_numeric(tsmc['tsmc_volume'])

#%%
#%%
phil.head()
#%%
phil['phil_date'] = pd.to_datetime(phil['DATE'])
#phil['phil_close'] = phil['CLOSE'].map(lambda x: re.sub("\D", "", x))
#phil['phil_volume'] = phil['VOLUME'].map(lambda x: re.sub("\D", "", x))
phil['phil_close'] = phil['CLOSE']
phil['phil_volume'] = phil['VOLUME']
#%%

phil['phil_close'] = pd.to_numeric(phil['phil_close'])
phil['phil_volume'] = pd.to_numeric(phil['phil_volume'])

# %%
df = krx_index[['date', 'close', 'volume']]

# %%
hynix.head()
# %%
tsmc['date'] = tsmc['tsmc_date']
hynix['date'] = hynix['hy_date']


# %%
tsmc.head()

# %%
hynix.head()

# %%
df = pd.merge(krx_index, hynix, on = 'date', how = 'left')

# %%
df = pd.merge(df, tsmc, on = 'date', how = 'left')


# %%
df.head()

# %%
df.columns

# %%
df = df[['date', 'close_x', 'volume_x', 'close_y', 'volume_y', 'CLOSE', 'VOLUME']]

# %%
df.head()

# %%
df.columns = ['date', 'krx_close', 'krx_volume', 'hy_close', 'hy_volume', 'ts_close', 'ts_volume']

# %%
df.head()

# %%

df.to_csv('df_test.csv')

# %%
fb.head()

# %%
google.head()

# %%
intel.head()

# %%
micron.head()
#%%

nv.head()

# %%
fb['date'] = pd.to_datetime(fb['DATE'])
google['date'] = pd.to_datetime(google['DATE'])
intel['date'] = pd.to_datetime(intel['DATE'])
micron['date'] = pd.to_datetime(micron['DATE'])
nv['date'] = pd.to_datetime(nv['DATE'])


# %%
df = pd.read_csv('df_test.csv')

# %%
df = pd.merge(df, nv, on = 'date', how = 'left')


# %%

# %%
df.head()

# %%
df.columns = ['0', 'date', 'krx_close', 'krx_volume', 'hy_close',
       'hy_volume', 'ts_close', 'ts_volume', 'fbDATE_x', 'fbOPEN_x', 'fbHIGH_x',
       'fbLOW_x', 'fb_close', 'fb_volume', 'fbCHANGE_x', 'gooDATE_y', 'gooOPEN_y',
       'gooHIGH_y', 'gooLOW_y', 'google_close', 'google_volume', 'gooCHANGE_y', 'inDATE_x',
       'inOPEN_x', 'inHIGH_x', 'inLOW_x', 'intel_close', 'intel_volume', 'inCHANGE_x',
       'miDATE_y', 'miOPEN_y', 'miHIGH_y', 'miLOW_y', 'micron_close', 'micron_volume',
       'miCHANGE_y', 'vnDATE', 'nvOPEN', 'nvHIGH', 'nvLOW', 'nv_close', 'nv_volume', 'nvCHANGE']

# %%
df2 = df[['date', 'krx_close', 'krx_volume', 'hy_close', 'hy_volume', 'ts_close', 'ts_volume', 'fb_close', 'fb_volume', 'google_close', 'google_volume', 'intel_close', 'intel_volume', 'micron_close', 'micron_volume', 'nv_close', 'nv_volume']]

# %%
df2.head()

# %%
# facebook은 생긴지 얼마 안되서 그런지 데이터 너무 없다. 무려 1500개가 null값임

df2.to_csv('df_test2.csv')

# %%
df2.info()

# %%
sns.distplot(df2['krx_close'], hist = False).set_title("KRX index distribution")


# %%
sns.distplot(df2['hy_close'], hist = False).set_title("Hynix distribution")

# %%
df2['hy_close'] = df2['hy_close'].map(lambda x: re.sub("\D", "", x))
df2['hy_volume'] = df2['hy_volume'].map(lambda x: re.sub("\D", "", x))
# %%
df2['hy_close'] = pd.to_numeric(df2['hy_close'])
df2['hy_volume'] = pd.to_numeric(df2['hy_volume'])

# %%
df2.info()

# %%
df2.to_csv('df_test2.csv')

# %%
