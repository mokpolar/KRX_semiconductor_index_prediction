#%%
import pandas as pd
import numpy as np
import sys

#%%
eco = pd.read_csv('bloomberg_smc_longdays.csv')
#%%

len(eco)
#%%


eco.head()
#%%
eco['col4'] = pd.to_numeric(eco['col4'])
#%%
eco.info()

#%%
eco['col4'].sort_values()
#%%

# col3 type datetime으로 만들기 
eco['col3'] = pd.to_datetime(eco['col3'])

#%%

# 변수 종류는 289개
len(eco['col0'].unique())
#%%

# 변수별로 데이터 갯수 확인 필요 뭐야 이거 시박 다 다르네
for i, j in enumerate(eco['col0'].unique()):
    print(i, eco['col0'].unique()[i], len(eco[eco['col0'] == j]))    

#%%

k = pd.DataFrame()
k_1 = []


# In[49]:


# 변수별로 데이터 갯수 확인해서 데이터 프레임으로 만들어 보자 
for i, j in enumerate(eco['col0'].unique()):
    k_1.append([i, eco['col0'].unique()[i], len(eco[eco['col0'] == j])])




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

         return 
        
    return 생성된 수많은 데이터프레임      



#%%
# 이거아냐
def index_slicer(dataframe, col_name):
    index_name = list(dataframe[col_name].unique())
    for i in index_name:
#%%
# 이것도 아냐
def make_df(index_name):
    new_df = dataframe[dataframe[col_name] == index_name]
    return new_df

#%%
name = list(eco['col0'].unique())




# %%
import sys
mod = sys.modules[__name__]

# %%
for i, j in enumerate(name):
    setattr(mod, name[i], eco[eco['col0'] == j])


# %%
name[0]

# %%
# 이제 일자기준으로 비는것은 null로 해서 merge하는 함수 쓰고 빈거 채우면 되겠다!
# 핳 ㅏㅎ ㅏ하
eco.head()
# %%
# 인덱스명에서 띄어쓰기 없애자 
eco['col0'] = eco['col0'].map(lambda x: x.replace(' ', ''))

# %%
eco.head()
#%%
len(name)

# %%
# 다시 
for i, j in enumerate(name):
    setattr(mod, name[i], eco[eco['col0'] == j])

# %%
FDTRIndex

# %%
name[1]

#%%
USGG3MIndex.head()
# %%
# 모든 데이터 프레임 객체에서 col0을 날리자
for i in name:
    globals()[i].drop('col0', axis = 1, inplace = True)

# %%
for i in name:
    globals()[i].columns = ['col1', 'col2', 'date', i]

#%%
for i in name:
    globals()[i].drop(['col1', 'col2'], axis = 1, inplace = True)
# %%
new_df = FDTRIndex

#%%
new_df.head()
# %%
for i in name:
    new_df = pd.merge(new_df, globals()[i], how = 'outer', on = 'date')


# %%
# %%

new_df.head()

# %%
pd.merge(new_df, globals()['USGG3MIndex'])

# %%
new_df.to_csv('bloomberg_total_index.csv')

# %%
new_df.info()

# %%
new_df.head()

# %%
len(new_df)

# %%
new_df['KOGCSTOQIndex'][0: 30]

# %%
KOGCSTOQIndex[0:30]

# %%
pd.fillna(new_df, 'ffill')

# %%
new_df_step = new_df

# %%
new_df_step.head()

# %%
new_df