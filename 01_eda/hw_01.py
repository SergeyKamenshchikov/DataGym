#!/usr/bin/env python
# coding: utf-8

# Import libraries:

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Используя параметры pandas красиво прочитать файл 
df = pd.read_csv('UCI_Credit_Card.csv', sep=',')
df.head().loc[:, 'ID':'AGE']


# In[3]:


for item in df.columns[:]:
    print(item)


# In[ ]:


# Выведите типы переменных
print('Data types:')
print(df.dtypes[:10])


# In[4]:


# Выведите число пропусков


# In[5]:


print('\nNumber of nulls:')
df.isnull().sum()[:10]


# In[7]:


# Для численных значений посчитайте пару статистик

print('Age median:', int(df['AGE'].median()))
print('Age minimum:', int(df['AGE'].min()))
print('Age maximum:', int(df['AGE'].max()))


# In[8]:


df['MARRIAGE'].describe()


# In[10]:


# Посчитать число женщин с университетским образованием
print('Well educated girls:', len(set(df[(df['SEX']==2) & (df['EDUCATION']==2)]['ID'])))


# In[11]:


filter_col = [i for i in list(df.columns) if (i.startswith('BILL_') | i.startswith('PAY_'))]
filter_col.append('default.payment.next.month')


# In[12]:


# Сгрупировать по 'default payment next month' и посчитать медиану для показателей начинающихся на BILL_ и PAY_
df[filter_col].groupby('default.payment.next.month').median()


# In[13]:


# Постройте pivot table по SEX, EDUCATION, MARRIAGE
df.pivot_table('MARRIAGE', 'SEX', 'EDUCATION',  aggfunc='count')


# In[ ]:


# (6) Создать новый строковый столбец в data frame-е, который:
# принимает значение A, если значение LIMIT_BAL <=10000
# принимает значение B, если значение LIMIT_BAL <=100000 и >10000
# принимает значение C, если значение LIMIT_BAL <=200000 и >100000
# принимает значение D, если значение LIMIT_BAL <=400000 и >200000
# принимает значение E, если значение LIMIT_BAL <=700000 и >400000
# принимает значение F, если значение LIMIT_BAL >700000

def limit_bal(x):
    
    if x<=10000:
        return 'A'
    elif x<=100000 and x>10000:
        return 'B'
    elif x<=200000 and x>100000:
        return 'C'
    elif x<=400000 and x>200000:
        return 'D'
    elif x<=700000 and x>400000:  
        return 'E'
    else:
        return 'F'
    
df['limit'] = df['LIMIT_BAL'].map(limit_bal)    


# In[30]:


# Построить распределение LIMIT_BAL
df[['LIMIT_BAL']].plot.hist(bins = 10, rwidth = 0.8, alpha = 0.8)

plt.xlabel('LIMIT_BAL')
plt.ylabel('NUMBER')

plt.grid()
plt.show()


# In[ ]:


# Построить зависимость кредитного лимита и образования для каждого из полов


# In[38]:


_, ax = plt.subplots(figsize=(8,5))

df_male_female = df[['EDUCATION', 'LIMIT_BAL']].groupby('EDUCATION').mean().sort_values('EDUCATION')['LIMIT_BAL']
df_male_female.plot(kind='bar', color='green')

plt.xlabel('Education')
plt.ylabel('Mean credit limit')
plt.xticks(rotation='horizontal')

plt.show()


# In[35]:


df.pivot_table('LIMIT_BAL', 'SEX', 'EDUCATION',  aggfunc='mean')


# In[36]:


# Построить зависимость кредитного лимита и образования только для одного из полов
_, ax = plt.subplots(figsize=(8,5))

df_male = df[df['SEX']==1][['EDUCATION', 'LIMIT_BAL']].groupby('EDUCATION').mean().sort_values('EDUCATION')['LIMIT_BAL']
df_male.plot(kind='bar', color='red', label='Male')

plt.xlabel('Education')
plt.ylabel('Mean credit limit')

plt.xticks(rotation='horizontal')
plt.legend()

plt.show()


# In[44]:


plt.figure(figsize=(8, 6))
sns.heatmap(df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].corr('kendall'), annot=True);

# PAY_0 
# PAY_2 


# In[ ]:


# (10) построить большой график (подсказка - используя seaborn) для построения завимисости всех возможных пар параметров
# разным цветом выделить разные значение "default payment next month"
# (но так как столбцов много - картинка может получиться "монструозной")
# (поэкспериментируйте над тем как построить подобное сравнение параметров)
# (подсказка - ответ может состоять из несколькольких графиков)
# (если не выйдет - программа минимум - построить один график со всеми параметрами)

sns.pairplot(df[['LIMIT_BAL', 'EDUCATION', 'default.payment.next.month']])


# Построим коррелляции платежей:

# In[46]:


plt.figure(figsize=(8, 6))
sns.heatmap(df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].corr('kendall'), annot=True)
plt.show()


# Высокая коррелляция между вторым платежом и остальными. Можем оставить PAY_6 в качестве признака, т.к. его наличие отражает всю цепочку оплаты. Еще лучше ситуация с признаками BILL_AMT - можем понизить размерность, оставив один из признаков:  

# In[47]:


plt.figure(figsize=(8, 6))
sns.heatmap(df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].corr('kendall'), annot=True)
plt.show()


# Похуже ситуация с PAY_AMT, но также есть слабая коррелляция:

# In[63]:


plt.figure(figsize=(8, 6))
sns.heatmap(df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].corr('kendall'), annot=True)
plt.show()


# In[68]:


features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_6', 'BILL_AMT6', 'PAY_AMT6']
sns.pairplot(df, vars=features, hue='default.payment.next.month')
plt.show()

