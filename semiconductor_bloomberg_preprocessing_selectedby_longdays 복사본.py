#!/usr/bin/env python
# coding: utf-8

# In[1]:


ls


# In[2]:


pwd


# In[21]:


import pandas as pd
import re
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


eco = pd.read_csv('/Users/mokpolar/Downloads/bloomberg_raw/economic.csv')


# In[23]:


eco.set_index('col0', inplace = True)


# In[24]:


eco.head()


# In[25]:


eco = eco.loc[['FDTR Index',
'USGG3M Index',
'USGG10YR Index',
'USTW$ Index',
'INJCJC Index',
'INJCSP Index',
'USEMNCHG Index',
'NFP PCH Index',
'NFP TCH Index',
'USMMMNCH Index',
'OUTFGAF Index',
'KOIMTOTY Index',
'KOFETOT% Index',
'KOEXTOTY Index',
'KOCPIMOM Index',
'SKCIMOM Index',
'USHETOT% Index',
'KOCPIYOY Index',
'SKCIYOY Index',
'USHEYOY Index',
'USURTOT Index',
'KOFETOT Index',
'SAARTOTL Index',
'NAPMPRIC Index',
'KOCPI Index',
'CONCCONF Index',
'NAPMNEWO Index',
'SKCITTL Index',
'CHPMINDX Index',
'USWHTOT Index',
'NAPMPMI Index',
'SKBSIC Index',
'CONSSENT Index',
'USERTOT Index',
'LEI WKIJ Index',
'KOEXTOT Index',
'KOIMTOT Index',
'NFP T Index',
'USEMTOT Index',
'KOIVCONY Index',
'KOEXPTIY Index',
'KOIMPTIY Index',
'KOIPMY Index',
'KOIPIY Index',
'KOPSIY Index',
'LEI YOY Index',
'KOPIIY Index',
'IP  YOY Index',
'KOIPOPSM Index',
'KOEXPTIM Index',
'KOIPMSM Index',
'KOIPIMOM Index',
'TMNOCHNG Index',
'KOIPMCY Index',
'KOIMPTIM Index',
'LEI IRTE Index',
'PITLYOY Index',
'CFNAI Index',
'PITLCHNG Index',
'KOPPIYOY Index',
'IP  CHNG Index',
'CFNAIMA3 Index',
'LEI CHNG Index',
'LEI ACE Index',
'SKLILY Index',
'KOPPIMOM Index',
'CPI YOY Index',
'CPI CHNG Index',
'PCE DEFY Index',
'CPUPXCHG Index',
'CPI XYOY Index',
'PCE CYOY Index',
'PIDSDPS Index',
'KOIPMS Index',
'KOIPISA Index',
'SKLILI Index',
'SKLILC Index',
'SKLICI Index',
'KOIPMC Index',
'KOIMPTI Index',
'LEI AVGW Index',
'KOPPI Index',
'IP Index',
'LEI TOTL Index',
'KOEXPTI Index',
'CPTICHNG Index',
'LEI STKP Index',
'LEI BP Index',
'PIDSPINX Index',
'LEI MNO Index',
'LEI NWCN Index',
'KOBPCB Index',
'KOBPFIN Index',
'KOBPTB Index',
'KODIBAL Index',
'FRNTTNET Index',
'FRNTTOTL Index',
'KOBPCA Index',
'KOMSMBY Index',
'CICRTOT Index',
'KOMSM1FY Index',
'KOMSM1Y Index',
'MTIBCHNG Index',
'MGT2TB Index',
'MGT2RE Index',
'OEUSKLAP Index',
'OEUSKLAR Index',
'OEKRN022 Index',
'GPDITOC% Index',
'KOECGCPY Index',
'KOECIMPY Index',
'KOECGCSY Index',
'KOECSIMQ Index',
'KOECPRCY Index',
'KOECSPRQ Index',
'KOECFCSY Index',
'KOECFCOY Index',
'KOECSEMQ Index',
'GDP CQOQ Index',
'GPGSTOC% Index',
'KOECEXPY Index',
'GDP CYOY Index',
'GDPCTOT% Index',
'GDP CURY Index',
'KOECSGVQ Index',
'KOECGVTY Index',
'GDP PIQQ Index',
'GDPCPCEC Index',
'GDP CUR$ Index',
'USCABAL Index',
'JNVNIYOY Index',
'COSTNFR% Index',
'KOGNICNY Index',
'KODFTOTY Index',
'KOGNICUY Index',
'EHCAUS Index',
'KOGCGDPY Index',
'PRODNFR% Index',
'KOGCSTOQ Index'], :]


# In[33]:


eco.head()


# In[27]:


eco.reset_index(inplace = True)


# In[30]:


len(eco)


# In[29]:


eco.to_csv('bloomberg_smc_longdays.csv')


# In[ ]:





# In[34]:


eco.info()


# ### --- 여기까지는 Column 뽑아내기 ---

# In[236]:


eco = pd.read_csv('bloomberg_smc_longdays.csv.csv')


# In[35]:


len(eco)


# In[36]:


eco.head()


# In[37]:


eco['col4'] = pd.to_numeric(eco['col4'])


# In[38]:


eco.info()


# In[39]:


eco['col4'].sort_values()


# In[40]:


# col3 type datetime으로 만들기 
eco['col3'] = pd.to_datetime(eco['col3'])


# In[41]:


# 변수 종류는 289개
len(eco['col0'].unique())


# In[42]:


# 변수별로 데이터 갯수 확인 필요 뭐야 이거 시박 다 다르네
for i, j in enumerate(eco['col0'].unique()):
    print(i, eco['col0'].unique()[i], len(eco[eco['col0'] == j]))    


# In[46]:


k = pd.DataFrame()


# In[47]:


k_1 = []


# In[49]:


# 변수별로 데이터 갯수 확인해서 데이터 프레임으로 만들어 보자 
for i, j in enumerate(eco['col0'].unique()):
    k_1.append([i, eco['col0'].unique()[i], len(eco[eco['col0'] == j])])


# In[53]:


import numpy as np


# In[54]:


k_1.to_ndarray(column= ['index', 'ticker', 'len_data'])


# In[56]:


npa = np.asarray(k_1)


# In[59]:


k = pd.DataFrame(npa, columns = ['index', 'ticker', 'len_data'])


# In[63]:


k.drop('index', axis = 1, inplace = True)


# In[65]:


k.head()


# In[66]:


k['len_data'].sort_values()


# In[71]:


k['len_data'].unique()


# 데이터 갯수가 152, 151은 분기별 데이터

# In[67]:


k.loc[138, :]


# In[72]:


k.loc[30, :]


# In[68]:


k.to_csv('bloom_smc_longdays_len_data.csv')


# In[69]:


eco.head()


# In[70]:


eco[eco['col0'] == 'KOGCSTOQ Index']


# 일단 기간을 1981뇬 3월 31일까지 있는거부터 2019년 6월 30일 까지 있는 티커들만 뽑았음  
# 근데 보니까 데이터 151개만 있는건 분기별 데이터네
# 

# In[73]:


eco[eco['col0'] == 'NAPMPMI Index']


# 455, 456개 있는건 월별 데이터

# In[74]:


k.loc[3, :]


# In[75]:


eco[eco['col0'] == 'USTW$ Index']


# 9591개 있는 건 데일리 데이터네 그럼 분류는 일, 월, 분기 세 가지 분류로 잡고   
# 월을 선으로 이어서 만들어 보자 

# In[76]:


eco.info()


# 인덱스 별로 데이터 프레임을 발라내자 !

# In[78]:


list(eco['col0'].unique())


# In[82]:


for i in list(eco['col0'].unique()):
    i = eco[eco['col0'] == i]


# In[83]:


i


# In[ ]:


# input 은 dataframe

def index_slicer(dataframe, col_name):
    index_name = list(dataframe[col_name].unique())
    for i in index_name:
         = dataframe[dataframe[col_name] == i]
        
    return 생성된 수많은 데이터프레임      

