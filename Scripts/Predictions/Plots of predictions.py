# -*- coding: utf-8 -*-
"""
Created on Sun May 15 23:57:24 2022

@author: bened
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\bened\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\Machine learning data\ML_ready\XG_Corr_panel_predictions.csv')
df = df.set_index(['region', 'År']).rename(columns={'0' : 'Predicted'})

df_em = pd.read_csv(r"C:\Users\bened\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\Machine learning data\ML_ready\emissions_master.csv")
df_em = df_em.set_index(['region', 'År'])

df = pd.concat([df['Predicted'], df_em['tCO2e']], axis=1).dropna().reset_index()


sarp = df.loc[df['region']=='Sarpsborg']
Oslo = df.loc[df['region']=='Oslo']
aren = df.loc[df['region']=='Arendal']
fauske = df.loc[df['region']=='Fauske']
alver = df.loc[df['region']=='Alver']

#%%
sns.lineplot(data=sarp, x='År',  y='Predicted', label='Predicted emissions')
sns.lineplot(data=sarp, x='År',  y='tCO2e', label='Actual emissions')
plt.suptitle('Predicted vs actual emissions, Sarpsborg')
plt.xlabel('Year')
plt.show()

#%%

sns.lineplot(data=Oslo, x='År',  y='Predicted', label='Predicted emissions')
sns.lineplot(data=Oslo, x='År',  y='tCO2e', label='Actual emissions')
plt.suptitle('Predicted vs actual emissions, Oslo')
plt.ylabel('tCO2e')
plt.xlabel('Year')
plt.show()

#%%

sns.lineplot(data=aren, x='År',  y='Predicted', label='Predicted emissions')
sns.lineplot(data=aren, x='År',  y='tCO2e', label='Actual emissions')
plt.suptitle('Predicted vs actual emissions, Arendal')
plt.ylabel('tCO2e')
plt.xlabel('Year')
plt.show()

#%%

sns.lineplot(data=fauske, x='År',  y='Predicted', label='Predicted emissions')
sns.lineplot(data=fauske, x='År',  y='tCO2e', label='Actual emissions')
plt.suptitle('Predicted vs actual emissions, Fauske')
plt.ylabel('tCO2e')
plt.xlabel('Year')
plt.show()

#%%

sns.lineplot(data=alver, x='År',  y='Predicted', label='Predicted emissions')
sns.lineplot(data=alver, x='År',  y='tCO2e', label='Actual emissions')
plt.suptitle('Predicted vs actual emissions, Alver')
plt.ylabel('tCO2e')
plt.xlabel('Year')
plt.show()