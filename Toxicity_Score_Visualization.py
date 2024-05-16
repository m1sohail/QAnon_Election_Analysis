#!/usr/bin/env python
# coding: utf-8

# ### Trump Tweet Toxicity Visualization

# ## Pre-Processing

# In[142]:


import pandas as pd
Trump_Toxic = pd.read_csv('/Users/sohail/Desktop/Trump_with_tox_scores.csv')


# In[387]:


Trump_Toxic


# In[413]:


df = pd.DataFrame(Trump_Toxic['date'].str.split('/',3).tolist(),
                                 columns = ['month','day','year'])


# In[414]:


df_2 = pd.DataFrame(df['year'].str.split(' ',1).tolist(),
                                 columns = ['year','time'])


# In[415]:


df = df.merge(df_2, left_index=True, right_index=True)


# In[416]:


df = df.drop(columns=['year_x'])


# In[417]:


df['year'] = df['year_y']


# In[418]:


df = df.drop(columns=['year_y'])


# In[419]:


Trump_Toxic = Trump_Toxic.merge(df, left_index=True, right_index=True)


# In[420]:


Data = Trump_Toxic[['day', 'month','year','Toxicity_score']]


# In[421]:


List = ['day', 'month','year']

for i in List:
    Data[i] = pd.to_numeric(Data[i])


# In[422]:


Trump_2020 = Data[Data["year"]==2020]
Trump_2019 = Data[Data["year"]==2019]
Trump_2018 = Data[Data["year"]==2018]
Trump_2017 = Data[Data["year"]==2017]


# Creating Datasets by year only, for average and median Toxicity scores.

# In[423]:


Trump_2017_mean = pd.DataFrame(Trump_2017.groupby(["month"]).mean())
Trump_2018_mean = pd.DataFrame(Trump_2018.groupby(["month"]).mean())
Trump_2019_mean = pd.DataFrame(Trump_2019.groupby(["month"]).mean())
Trump_2020_mean = pd.DataFrame(Trump_2020.groupby(["month"]).mean())
Trump_2017_median = pd.DataFrame(Trump_2017.groupby(["month"]).median())
Trump_2018_median = pd.DataFrame(Trump_2018.groupby(["month"]).median())
Trump_2019_median = pd.DataFrame(Trump_2019.groupby(["month"]).median())
Trump_2020_median = pd.DataFrame(Trump_2020.groupby(["month"]).median())
Trump_2017_mean = Trump_2017_mean.drop(columns=['day'])
Trump_2018_mean = Trump_2018_mean.drop(columns=['day'])
Trump_2019_mean = Trump_2019_mean.drop(columns=['day'])
Trump_2020_mean = Trump_2020_mean.drop(columns=['day'])
Trump_2017_median = Trump_2017_median.drop(columns=['day'])
Trump_2018_median = Trump_2018_median.drop(columns=['day'])
Trump_2019_median = Trump_2019_median.drop(columns=['day'])
Trump_2020_median = Trump_2020_median.drop(columns=['day'])


# Creating master dataset as well for both average and median Toxicity scores.

# In[424]:


Trump_Total_Data = Trump_2018_mean.merge(Trump_2019_mean, left_index=True, right_index=True)
Trump_Total_Data = Trump_Total_Data.merge(Trump_2020_mean, left_index=True, right_index=True)
Trump_Total_Data_mean = Trump_Total_Data.merge(Trump_2017_mean, how='outer', left_index=True, right_index=True)


# In[425]:


Trump_Total_Data_mean


# In[426]:


Trump_Total_Data_mean.columns = ['year_2018', '2018_TS','year_2019', '2019_TS','year_2020', '2020_TS','year_2017', '2017_TS']
Month_Names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
Trump_Total_Data_mean["Month_Names"] = Month_Names


# In[429]:


Trump_Total_Data = Trump_2018_median.merge(Trump_2019_median, left_index=True, right_index=True)
Trump_Total_Data = Trump_Total_Data.merge(Trump_2020_median, left_index=True, right_index=True)
Trump_Total_Data_median = Trump_Total_Data.merge(Trump_2017_median, how='outer', left_index=True, right_index=True)
Trump_Total_Data_median.columns = ['year_2018', '2018_TS','year_2019', '2019_TS','year_2020', '2020_TS','year_2017', '2017_TS']
Month_Names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
Trump_Total_Data_median["Month_Names"] = Month_Names
Trump_Total_Data_median


# # Visualization

# ### Visualization for multiple years  

# In[430]:


fig, ax = plt.subplots()

line4 = sns.lineplot( data=(Trump_Total_Data_mean), x="Month_Names", y='2017_TS', label = '2017')
line1 = sns.lineplot(data=(Trump_Total_Data_mean), x="Month_Names", y='2018_TS', label = '2018')
line2 = sns.lineplot(data=(Trump_Total_Data_mean), x="Month_Names", y='2019_TS', label = '2019')
line3 = sns.lineplot(data=(Trump_Total_Data_mean), x="Month_Names", y='2020_TS', label = '2020')


plt.title("Trump Tweet's Toxicity Score Average by Month")
plt.ylabel('Toxicity Score')
plt.xlabel('Month')
plt.legend()
plt.show()


# In[431]:


fig, ax = plt.subplots()

line4 = sns.lineplot( data=(Trump_Total_Data_median), x="Month_Names", y='2017_TS', label = '2017')
line1 = sns.lineplot(data=(Trump_Total_Data_median), x="Month_Names", y='2018_TS', label = '2018')
line2 = sns.lineplot(data=(Trump_Total_Data_median), x="Month_Names", y='2019_TS', label = '2019')
line3 = sns.lineplot(data=(Trump_Total_Data_median), x="Month_Names", y='2020_TS', label = '2020')


plt.title("Trump Tweet's Toxicity Score Median by Month")
plt.ylabel('Toxicity Score')
plt.xlabel('Month')
plt.legend()
plt.show()


# ## Individual Year Visualization

# In[436]:


sns.lineplot(data=Trump_2017_median, x="month", y="Toxicity_score",label = 2017).set_title("2017 Trump Tweet's Toxicity Score Median by Month")


# In[446]:


sns.lineplot(data=Trump_2018_median, x="month", y="Toxicity_score",label = 2018).set_title("2018 Trump Tweet's Toxicity Score Median by Month")


# In[447]:


sns.lineplot(data=Trump_2019_median, x="month", y="Toxicity_score",label = 2019).set_title(" 2019 Trump Tweet's Toxicity Score Median by Month")


# In[448]:


sns.lineplot(data=Trump_2020_median, x="month", y="Toxicity_score", label = 2020).set_title(" Trump Tweet's Toxicity Score Median by Month")


# In[449]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(40,10))
axes = axes.flatten()

ax = sns.lineplot(data=Trump_2017_median, x="month", y="Toxicity_score", label = 2017, ax=axes[0]).set_title(" 2017 Trump Tweet's Toxicity Score Median by Month")
ax = sns.lineplot(data=Trump_2018_median, x="month", y="Toxicity_score", label = 2018,  ax=axes[1]).set_title(" 2018 Trump Tweet's Toxicity Score Median by Month")
ax = sns.lineplot(data=Trump_2019_median, x="month", y="Toxicity_score", label = 2019,  ax=axes[2]).set_title(" 2019 Trump Tweet's Toxicity Score Median by Month")
ax = sns.lineplot(data=Trump_2020_median, x="month", y="Toxicity_score", label = 2020,  ax=axes[3]).set_title(" 2020 Trump Tweet's Toxicity Score Median by Month")


# In[441]:


sns.lineplot(data=Trump_2017_mean, x="month", y="Toxicity_score", label = 2017).set_title(" 2017 Trump Tweet's Toxicity Average by Month")


# In[442]:


sns.lineplot(data=Trump_2018_mean, x="month", y="Toxicity_score", label = 2018).set_title(" 2018 Trump Tweet's Toxicity Average by Month")


# In[443]:


sns.lineplot(data=Trump_2019_mean, x="month", y="Toxicity_score", label = 2019).set_title(" 2019 Trump Tweet's Toxicity Average by Month")


# In[444]:


sns.lineplot(data=Trump_2020_mean, x="month", y="Toxicity_score", label = 2020).set_title(" 2020 Trump Tweet's Toxicity Average by Month")


# In[445]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(40,10))
axes = axes.flatten()

ax = sns.lineplot(data=Trump_2017_mean, x="month", y="Toxicity_score", label = 2017, ax=axes[0]).set_title(" 2017 Trump Tweet's Toxicity Average by Month")
ax = sns.lineplot(data=Trump_2018_mean, x="month", y="Toxicity_score", label = 2018,  ax=axes[1]).set_title(" 2018 Trump Tweet's Toxicity Average by Month")
ax = sns.lineplot(data=Trump_2019_mean, x="month", y="Toxicity_score", label = 2019,  ax=axes[2]).set_title(" 2019 Trump Tweet's Toxicity Average by Month")
ax = sns.lineplot(data=Trump_2020_mean, x="month", y="Toxicity_score", label = 2020,  ax=axes[3]).set_title(" 2020 Trump Tweet's Toxicity Average by Month")


# ### QANON Data Pre-Processing

# In[496]:


import pandas as pd
Qanon_with_tox_scores = pd.read_csv('/Users/sohailbuttmac2018/Desktop/Qanon_with_tox_scores.csv')


# In[497]:


Qanon_with_tox_scores


# In[498]:


df = pd.DataFrame(Qanon_with_tox_scores['Date'].str.split(' ',5).tolist(),
                                 columns = ['month','day','year','time','EST'])


# In[499]:


Qanon_with_tox_scores = Qanon_with_tox_scores.merge(df, left_index=True, right_index=True)


# In[500]:


Data = Qanon_with_tox_scores[['day', 'month','year','Toxicity_score']]


# In[501]:


Data['month'] = Data['month'].replace(['Jan'],1)
Data['month'] = Data['month'].replace(['Feb'],2)
Data['month'] = Data['month'].replace(['Mar'],3)
Data['month'] = Data['month'].replace(['Apr'],4)
Data['month'] = Data['month'].replace(['May'],5)
Data['month'] = Data['month'].replace(['Jun'],6)
Data['month'] = Data['month'].replace(['Jul'],7)
Data['month'] = Data['month'].replace(['Aug'],8)
Data['month'] = Data['month'].replace(['Sep'],9)
Data['month'] = Data['month'].replace(['Oct'],10)
Data['month'] = Data['month'].replace(['Nov'],11)
Data['month'] = Data['month'].replace(['Dec'],12)


# In[502]:


Data['year'] = pd.to_numeric(Data['year'])
Data['year'] = pd.to_numeric(Data['year'])


# In[503]:


Qanon_2020 = Data[Data["year"]==2020]
Qanon_2019 = Data[Data["year"]==2019]
Qanon_2018 = Data[Data["year"]==2018]
Qanon_2017 = Data[Data["year"]==2017]


# In[506]:


Qanon_2017_mean = pd.DataFrame(Qanon_2017.groupby(["month"]).mean())
Qanon_2018_mean = pd.DataFrame(Qanon_2018.groupby(["month"]).mean())
Qanon_2019_mean = pd.DataFrame(Qanon_2019.groupby(["month"]).mean())
Qanon_2020_mean = pd.DataFrame(Qanon_2020.groupby(["month"]).mean())
Qanon_2017_median = pd.DataFrame(Qanon_2017.groupby(["month"]).median())
Qanon_2018_median = pd.DataFrame(Qanon_2018.groupby(["month"]).median())
Qanon_2019_median = pd.DataFrame(Qanon_2019.groupby(["month"]).median())
Qanon_2020_median = pd.DataFrame(Qanon_2020.groupby(["month"]).median())


# In[527]:


Qanon_Total_Data = Qanon_2018_median.merge(Qanon_2019_median, how='outer',left_index=True, right_index=True)
Qanon_Total_Data = Qanon_Total_Data.merge(Qanon_2020_median, how='outer',left_index=True, right_index=True)
Qanon_Total_Data_median = Qanon_Total_Data.merge(Qanon_2017_median, how='outer', left_index=True, right_index=True)
Qanon_Total_Data_median.columns = ['year_2018', '2018_TS','year_2019', '2019_TS','year_2020', '2020_TS','year_2017', '2017_TS']
Month_Names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
Qanon_Total_Data_median["Month_Names"] = Month_Names


# In[525]:


Qanon_Total_Data = Qanon_2018_mean.merge(Qanon_2019_mean, how='outer',left_index=True, right_index=True)
Qanon_Total_Data = Qanon_Total_Data.merge(Qanon_2020_mean, how='outer',left_index=True, right_index=True)
Qanon_Total_Data_mean = Qanon_Total_Data.merge(Qanon_2017_mean, how='outer', left_index=True, right_index=True)
Qanon_Total_Data_mean.columns = ['year_2018', '2018_TS','year_2019', '2019_TS','year_2020', '2020_TS','year_2017', '2017_TS']
Month_Names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
Qanon_Total_Data_mean["Month_Names"] = Month_Names


# ## Visualization for multiple years 

# In[526]:


fig, ax = plt.subplots()

line4 = sns.lineplot( data=(Qanon_Total_Data_mean), x="Month_Names", y='2017_TS', label = '2017')
line1 = sns.lineplot(data=(Qanon_Total_Data_mean), x="Month_Names", y='2018_TS', label = '2018')
line2 = sns.lineplot(data=(Qanon_Total_Data_mean), x="Month_Names", y='2019_TS', label = '2019')
line3 = sns.lineplot(data=(Qanon_Total_Data_mean), x="Month_Names", y='2020_TS', label = '2020')


plt.title("Qanon Tweet's Toxicity Score Average by Month")
plt.ylabel('Toxicity Score')
plt.xlabel('Month')
plt.legend()
plt.show()


# In[528]:


fig, ax = plt.subplots()

line4 = sns.lineplot( data=(Qanon_Total_Data_median), x="Month_Names", y='2017_TS', label = '2017')
line1 = sns.lineplot(data=(Qanon_Total_Data_median), x="Month_Names", y='2018_TS', label = '2018')
line2 = sns.lineplot(data=(Qanon_Total_Data_median), x="Month_Names", y='2019_TS', label = '2019')
line3 = sns.lineplot(data=(Qanon_Total_Data_median), x="Month_Names", y='2020_TS', label = '2020')


plt.title("Qanon Tweet's Toxicity Score median by Month")
plt.ylabel('Toxicity Score')
plt.xlabel('Month')
plt.legend()
plt.show()


# ### Individual Year Visualization

# In[534]:


sns.lineplot(data=Qanon_2017_median, x="month", y="Toxicity_score",label = 2017).set_title(" 2017 Qanon Tweet's Toxicity Median by Month")


# In[535]:


sns.lineplot(data=Qanon_2018_median, x="month", y="Toxicity_score",label = 2018).set_title(" 2018 Qanon Tweet's Toxicity Median by Month")


# In[536]:


sns.lineplot(data=Qanon_2019_median, x="month", y="Toxicity_score",label = 2019).set_title(" 2019 Qanon Tweet's Toxicity Median by Month")


# In[538]:


sns.lineplot(data=Qanon_2020_median, x="month", y="Toxicity_score",label = 2020).set_title(" 2020 Qanon Tweet's Toxicity Median by Month")


# In[533]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(40,10))
axes = axes.flatten()

ax = sns.lineplot(data=Qanon_2017_median, x="month", y="Toxicity_score", label = 2017, ax=axes[0]).set_title(" 2017 Qanon Tweet's Toxicity Median by Month")
ax = sns.lineplot(data=Qanon_2018_median, x="month", y="Toxicity_score", label = 2018,  ax=axes[1]).set_title(" 2018 Qanon Tweet's Toxicity Median by Month")
ax = sns.lineplot(data=Qanon_2019_median, x="month", y="Toxicity_score", label = 2019,  ax=axes[2]).set_title(" 2019 Qanon Tweet's Toxicity Median by Month")
ax = sns.lineplot(data=Qanon_2020_median, x="month", y="Toxicity_score", label = 2020,  ax=axes[3]).set_title(" 2020 Qanon Tweet's Toxicity Median by Month")


# In[539]:


sns.lineplot(data=Qanon_2020_mean, x="month", y="Toxicity_score", label = 2020).set_title(" 2020 Qanon Tweet's Toxicity Average by Month")


# In[540]:


sns.lineplot(data=Qanon_2019_mean, x="month", y="Toxicity_score", label = 2019).set_title(" 2019 Qanon Tweet's Toxicity Average by Month")


# In[541]:


sns.lineplot(data=Qanon_2018_mean, x="month", y="Toxicity_score", label = 2018).set_title(" 2018 Qanon Tweet's Toxicity Average by Month")


# In[542]:


sns.lineplot(data=Qanon_2017_mean, x="month", y="Toxicity_score", label = 2017).set_title(" 2017 Qanon Tweet's Toxicity Average by Month")


# In[543]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(40,10))
axes = axes.flatten()

ax = sns.lineplot(data=Qanon_2017_mean, x="month", y="Toxicity_score", label = 2017, ax=axes[0]).set_title(" 2017 Qanon Tweet's Average Toxicity Average by Month")
ax = sns.lineplot(data=Qanon_2018_mean, x="month", y="Toxicity_score", label = 2018,  ax=axes[1]).set_title(" 2018 Qanon Tweet's Average Toxicity Average by Month")
ax = sns.lineplot(data=Qanon_2019_mean, x="month", y="Toxicity_score", label = 2019,  ax=axes[2]).set_title(" 2019 Qanon Tweet's Average Toxicity Average by Month")
ax = sns.lineplot(data=Qanon_2020_mean, x="month", y="Toxicity_score", label = 2020,  ax=axes[3]).set_title(" 2020 Qanon Tweet's Average Toxicity Average by Month")


# ### Joint Visuals for QANON and Trump

# In[547]:


#2017 Median
fig, ax = plt.subplots()

line4 = sns.lineplot(data=Qanon_2017_median, x="month", y="Toxicity_score",label = 'Trump')
line1 = sns.lineplot(data=Trump_2017_median, x="month", y="Toxicity_score",label = 'QANON')

plt.title("2017 Comparitive Toxicity Score median by Month")
plt.ylabel('Toxicity Score')
plt.xlabel('Month')
plt.legend()
plt.show()


# In[548]:


#2017 Mean
fig, ax = plt.subplots()

line4 = sns.lineplot(data=Qanon_2017_mean, x="month", y="Toxicity_score",label = 'Trump')
line1 = sns.lineplot(data=Trump_2017_mean, x="month", y="Toxicity_score",label = 'QANON')

plt.title("2017 Comparitive Toxicity Score Average by Month")
plt.ylabel('Toxicity Score')
plt.xlabel('Month')
plt.legend()
plt.show()


# In[549]:


#2018 Median
fig, ax = plt.subplots()

line4 = sns.lineplot(data=Qanon_2018_median, x="month", y="Toxicity_score",label = 'Trump')
line1 = sns.lineplot(data=Trump_2018_median, x="month", y="Toxicity_score",label = 'QANON')

plt.title("2018 Comparitive Toxicity Score median by Month")
plt.ylabel('Toxicity Score')
plt.xlabel('Month')
plt.legend()
plt.show()


# In[550]:


#2018 Mean
fig, ax = plt.subplots()

line4 = sns.lineplot(data=Qanon_2018_mean, x="month", y="Toxicity_score",label = 'Trump')
line1 = sns.lineplot(data=Trump_2018_mean, x="month", y="Toxicity_score",label = 'QANON')

plt.title("2018 Comparitive Toxicity Score Average by Month")
plt.ylabel('Toxicity Score')
plt.xlabel('Month')
plt.legend()
plt.show()


# In[551]:


#2019 Median
fig, ax = plt.subplots()

line4 = sns.lineplot(data=Qanon_2019_median, x="month", y="Toxicity_score",label = 'Trump')
line1 = sns.lineplot(data=Trump_2019_median, x="month", y="Toxicity_score",label = 'QANON')

plt.title("2019 Comparitive Toxicity Score median by Month")
plt.ylabel('Toxicity Score')
plt.xlabel('Month')
plt.legend()
plt.show()


# In[552]:


#2019 Mean
fig, ax = plt.subplots()

line4 = sns.lineplot(data=Qanon_2019_mean, x="month", y="Toxicity_score",label = 'Trump')
line1 = sns.lineplot(data=Trump_2019_mean, x="month", y="Toxicity_score",label = 'QANON')

plt.title("2019 Comparitive Toxicity Score average by Month")
plt.ylabel('Toxicity Score')
plt.xlabel('Month')
plt.legend()
plt.show()


# In[553]:


#2019 Median
fig, ax = plt.subplots()

line4 = sns.lineplot(data=Qanon_2019_median, x="month", y="Toxicity_score",label = 'Trump')
line1 = sns.lineplot(data=Trump_2019_median, x="month", y="Toxicity_score",label = 'QANON')

plt.title("2019 Comparitive Toxicity Score Median by Month")
plt.ylabel('Toxicity Score')
plt.xlabel('Month')
plt.legend()
plt.show()


# In[554]:


#2020 Mean
fig, ax = plt.subplots()

line4 = sns.lineplot(data=Qanon_2020_mean, x="month", y="Toxicity_score",label = 'Trump')
line1 = sns.lineplot(data=Trump_2020_mean, x="month", y="Toxicity_score",label = 'QANON')

plt.title("2020 Comparitive Toxicity Score average by Month")
plt.ylabel('Toxicity Score')
plt.xlabel('Month')
plt.legend()
plt.show()


# In[555]:


#2020 Mean
fig, ax = plt.subplots()

line4 = sns.lineplot(data=Qanon_2020_median, x="month", y="Toxicity_score",label = 'Trump')
line1 = sns.lineplot(data=Trump_2020_median, x="month", y="Toxicity_score",label = 'QANON')

plt.title("2020 Comparitive Toxicity Score median by Month")
plt.ylabel('Toxicity Score')
plt.xlabel('Month')
plt.legend()
plt.show()


# In[ ]:




