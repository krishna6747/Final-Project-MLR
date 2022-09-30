# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:55:37 2022

@author: Krishna
"""


#libraries import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

#Extracting data 
df = pd.read_csv(r"E:\PGBI&DA\Multiple_linear_regression_Assignment\50_Startups.csv")
df.head()
df.shape
df.columns

#Some Columns have Special character names so we will change names
df.rename(columns={'R&D Spend':'RDS','Marketing Spend':'MS'}, inplace=True)
df
df.dtypes



#lets Check missing values
missing_value_count = df.isnull().sum()
print(missing_value_count)

#cheking duplicate values from data
df.duplicated()

#using bar  graph to check missing value
msno.bar(df)
df.describe()


#Lets check outliers in data
df.plot(kind='box')

#only one outliers is profit columns
#lets check correlation between all features

sns.pairplot(df)
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')
sns.pairplot(df, hue='State')

#From above pairplot we can RDS and Profit are highly correlated with each other

from scipy import stats
import pylab
stats.probplot(df.Profit, dist='norm', plot=pylab, rvalue=True)
plt.show()
stats.probplot(df.RDS, dist='norm', plot=pylab, rvalue=True)
plt.show()
stats.probplot(df.Administration, dist='norm', plot=pylab, rvalue=True)
plt.show()
stats.probplot(df.MS, dist='norm', plot=pylab, rvalue=True)
plt.show()

# We will check Co-linearity betweeen input variables
import statsmodels.formula.api as smf
model = smf.ols('Profit ~ RDS + MS + Administration', data=df).fit()
model.summary()

#The p value of MS and Administration is greater then 0.05, lets check the influential values

import statsmodels.api as sm
sm.graphics.influence_plot(model)

#Row 49 shows high influence on data lets remove this row and check P Value
df_new = df.drop(df.index[[49]])
df_new.shape

#Lets validate model again
model_new = smf.ols('Profit ~ RDS + MS + Administration', data=df_new).fit()
model_new.summary()

#We will check for colinearit to ecide to remove a variable using VIF score
#Assumption VIF > 10 = colinearity
#Lets check independent VIF values

rsq_profit = smf.ols('Profit ~ RDS + MS + Administration', data=df).fit().rsquared
vif_profit = 1/(1 - rsq_profit)
print(vif_profit)

rsq_MS = smf.ols('MS ~ Profit + RDS + Administration', data=df).fit().rsquared
vif_MS = 1/(1 - rsq_MS)
print(vif_MS)

rsq_RDS = smf.ols('RDS ~ Profit + MS + Administration', data=df).fit().rsquared
vif_RDS = 1/(1 - rsq_RDS)
print(vif_RDS)

rsq_admin = smf.ols('Administration ~ RDS + MS + Profit', data=df).fit().rsquared
vif_admin = 1/(1- rsq_admin)
print(vif_admin)

#Sorting vif Values in a dataframe
df_vif = {'Variables':['Profit','RDS','MS','Administration'], 'VIF':[vif_profit,vif_RDS,vif_MS,vif_admin]}
vif_frame = pd.DataFrame(df_vif)
vif_frame

#We Wil drop RDS from our prediction model
final_model = smf.ols('Profit ~ MS + Administration', data=df).fit()
final_model.summary()

#Prediction
pred = final_model.predict(df)

#Q-Q Plot
res = final_model.resid
sm.qqplot(res)
plt.show()

#Residual vs Fitted Plot

sns.residplot(x=pred, y=df.Profit, lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual plot')
plt.show()

#influence plot
sm.graphics.influence_plot(final_model)

#Building Models

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2)

#Preparing Train model
model_Train = smf.ols('Profit ~ MS + Administration', data=df_train).fit()
model_Train.summary()

test_pred = model_Train.predict(df_test)
test_resid = test_pred - df_test.Profit

test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

#df_train predictions
train_pred = model_Train.predict(df_train)

train_resid = train_pred - df_train.Profit
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

#train_rmse is 24295.933244229902 for This model
#test_rmse is 27788.12845532531 for this model



















