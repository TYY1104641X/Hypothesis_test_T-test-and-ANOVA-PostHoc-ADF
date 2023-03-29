# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:18:00 2023

@author: Yuanyuan_Tang

Function: learn different t-test models

"""



'''
Import package
'''

import pandas as pd
import numpy as np


from scipy.stats import ttest_ind  # Import the two-sample test



'''
-----------------------------------------------
Function: T-test
-----------------------------------------------
'''

print('-----------------------T-Test Start---------------------------')

'''
Import data from pandas
'''

data=pd.read_csv('flower.data.csv')   
#print(data.head()) 
#(data.tail()) 

group1=data[data['Species']=='setosa']
group2=data[data['Species']=='virginica']

results=ttest_ind(group1['Petal.Length'], group2['Petal.Length'])    # By assuming the same variance

print(results)


'''
Example 2: Welch’s t-Test in Pandas
 two populations that the samples came from have different variance.
'''
results1=ttest_ind(group1['Petal.Length'], group2['Petal.Length'], equal_var=False)    

print(results1)

'''
Example 3: Paired Samples t-Test in Pandas
determine if two population means are equal in which each observation in one sample
 can be paired with an observation in the other sample. 
 
By defualt: return 2-tailed p values.
 1-tailed p value should be used if we have a specific prediction of a “direction” of the difference. 
'''

from scipy.stats import ttest_rel    # Paired t-test function
results2=ttest_ind(group1['Petal.Length'], group2['Petal.Length'])    # By assuming the same variance

print(results2)




'''
-----------------------------------------------
Function: ANOVA test
-----------------------------------------------
'''

print('-----------------------ANOVA Test start---------------------------')

data_F=pd.read_csv("ANOVA.txt", sep="\t")
print(data_F.head())

# reshape the d dataframe suitable for statsmodels package 
df_melt = pd.melt(data_F.reset_index(), id_vars=['index'], value_vars=['A', 'B', 'C', 'D'])
# replace column names
df_melt.columns = ['index', 'treatments', 'value']

# generate a boxplot to see the data distribution by treatments. Using boxplot, we can 
# easily detect the differences between different treatments
import matplotlib.pyplot as plt
import seaborn as sns
ax = sns.boxplot(x='treatments', y='value', data=df_melt, color='#99c2a2')
ax = sns.swarmplot(x="treatments", y="value", data=df_melt, color='#7d0013')
plt.show()


'''
One-way ANOVA
'''

from scipy.stats import f_oneway
fval, pval=f_oneway(data_F['A'],data_F['B'],data_F['C'],data_F['D'] )
print('f=',fval, ', P-val=',pval)


'''
Post-hoc test
'''
print('-----------------------Post-hoc Start---------------------------')

import statsmodels.api as sa
import statsmodels.formula.api as sfa
import scikit_posthocs as sp

#re=sp.posthoc_ttest(df_melt, val_col='value', group_col='treatments', p_adjust='holm')  # With two input arrays
re=sp.posthoc_tukey(df_melt, val_col='value', group_col='treatments') 
print('tukey_hsd=',re)


from bioinfokit.analys import stat
# perform multiple pairwise comparison (Tukey's HSD)
# unequal sample size data, tukey_hsd uses Tukey-Kramer test
res = stat()
res.tukey_hsd(df=df_melt, res_var='value', xfac_var='treatments', anova_model='value ~ C(treatments)')
res.tukey_summary
