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
