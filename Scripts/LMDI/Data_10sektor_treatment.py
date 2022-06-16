# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:12:58 2022

@author: bengo
"""

import pandas as pd
import numpy as np

import os

# Set dir for emCode helper functions
# os.chdir('C:/Users/bened/OneDrive - Norwegian University of Life Sciences/Fag/Masteroppgave/Python stuff/Functions')

# Set dir for emCode helper functions
os.chdir('C:/Users/bengo/OneDrive - Norwegian University of Life Sciences/Fag/Masteroppgave/Python stuff/Functions')


# # Import helper functions
# from emcode_funcs import emCode_parent_to_sector


import seaborn as sns
import matplotlib.pyplot as plt

# Desktop path - working dir
# os.chdir(r'C:\Users\bened\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\Ferdige_data')

# laptop path - working dir
os.chdir(r'C:\Users\bengo\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\Ferdige_data')


#%%

def emCode_parent_to_sector(df):
    
    """
    Input: df
    
    Operation:
        - makes emCode_parent type int
        - reads emCode_parent column
        - applies aggregate sector name to emCode in new column 'næring'
        
    Returns: 
        df with aggregate sector names
        
    Key: emCode lvl1
    Value: SN2012 aggregate sector
    """
    
    df['emCode_parent'] = df['emCode_parent'].astype('int')

    # Associate parent codes and sector columns
    koder = pd.read_csv(r'C:/Users/bengo/OneDrive - Norwegian University of Life Sciences/Fag/Masteroppgave/Data/SSB/Klassifisering/emcodeparent_to_næring_final.csv', sep=',', encoding='ANSI')
    d = dict(zip(koder.emCode_parent, koder.næring_11))
    
    df['agg_næring'] = df['emCode_parent'].map(d)
    
    return df

def emCode_associator_rev_gdp(df):
    
    """
    Associates relevant emCode and em-sector name
    Key: emCode lvl2
    
    """
    df['emCode'] = df['emCode'].astype('str')
    
    # Import associated names and NACE codes from SSB metadatafile related to emissions
    koder = pd.read_csv(r'C:\Users\bengo\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\SSB\Klassifisering\MiljøregnskapSN_naceBNP_final.csv', sep=';', encoding='ANSI')
    koder['sourceCode'] = koder['sourceCode'].astype('str')
    d = dict(zip(koder.sourceCode, koder.sourceName))
    
    # Map NACE codes to uslipp
    df['næring'] = df['emCode'].map(d)
    
    return df

#%%

# read in data - sub-sectors
nrg31 = pd.read_csv('energi_sektorer_31.csv', )
emi31 = pd.read_csv('utslipp_sektorer_31.csv')
gdp31 = pd.read_csv('bnp_sektorer_31.csv')
foss31 = pd.read_csv('energi_foss_31.csv')

# read in data - aggreggate sectors
nrg10 = pd.read_csv('energi_sektorer_10.csv')
emi10 = pd.read_csv('utslipp_sektorer_10.csv')
gdp10 = pd.read_csv('bnp_sektorer_10.csv')
foss10 = pd.read_csv('energi_foss_10.csv')

# Read in data - total
gdp = pd.read_csv('bnp_totalt.csv')

#%%

# Check for sector gwh == total gwh for sectors
nrg_t = pd.read_csv(r'C:\Users\bengo\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\SSB\Rådata\energibalanse_gwh.csv',
                    sep=';', header=1, encoding='ANSI')

nrg_t['Mengde (GWh)'] = nrg_t['Mengde (GWh)'].str.replace(':', '0')
nrg_t['Mengde (GWh)'] = nrg_t['Mengde (GWh)'].str.replace('.', '0')
nrg_t['Mengde (GWh)'] = nrg_t['Mengde (GWh)'].str.replace('-', '0')
nrg_t['Mengde (GWh)'] = nrg_t['Mengde (GWh)'].astype('float')

# Get total energy usage from sectors without households
nrg_t = nrg_t.loc[nrg_t['næring'] == 'Alle næringer']
nrg_t = nrg_t.loc[nrg_t['energiproduktregnskap'] == 'Alle energiprodukter'].reset_index()['Mengde (GWh)']

# Get total usage from 10 sectors
test = nrg10.groupby('år').sum().reset_index()['value']

# concatenate together
test = pd.concat([nrg_t, test], axis =1)

# check difference
test['diff'] = test['Mengde (GWh)'] - test['value']

# Get total gdp
t_gdp = gdp10.groupby('år').sum().reset_index()['value']

#concat in gdp
test = pd.concat([test, t_gdp], axis = 1)

# Calculate energy intensity
test['gwh/gdp'] = test['Mengde (GWh)'] / test.iloc[:,3]

#%%

# Plot energy intensity
sns.set(rc = {'figure.figsize':(24,16)}, style="whitegrid")
plot = sns.lineplot(data=test['gwh/gdp'], palette='viridis', linewidth=3, size_norm=())
plt.suptitle('Norwegian energy intensity, all sectors, 1990 - 2019', size=18)
plot.set_ylabel('GWh/GDP')
plot.set_xlabel('Year')
# plt.legend(title='GDP, Fixed 2015-prices', loc='lower right')
plt.show()
# plt.savefig('utslipp_nasjonalt_1990-2020.png')



#%%

# Rename columns

class renamer:
    
    def __init__(self, df, colname):
        
        self.df = df
        self.colname = colname
    
    def renamefunc(self):
        df = self.df.rename(columns={'value' : self.colname})
        return df

# Rename columns to correct names
nrg10, nrg31 = renamer(nrg10, 'sec_GWh').renamefunc(), renamer(nrg31, 'sub_GWh').renamefunc()
emi10, emi31 = renamer(emi10, 'sec_mtCO2e').renamefunc(), renamer(emi31, 'sub_mtCO2e').renamefunc()
gdp10, gdp31 = renamer(gdp10, 'sec_gdp').renamefunc(), renamer(gdp31, 'sub_gdp').renamefunc()
foss10, foss31 = renamer(foss10, 'foss_GWh').renamefunc(), renamer(foss31, 'foss_GWh').renamefunc()


df1, df2 = nrg31, nrg10

#Apply sub sector names to fossil dataframe
foss31 = foss31.pipe(emCode_associator_rev_gdp)

# Apply aggregate sector names to parent codes in dfs
df_list = [nrg31, foss31, gdp31, emi31]
df_list = [df.pipe(emCode_parent_to_sector) for df in df_list]

def tidyfunc_aggsector(df):
    df = df.sort_values(by=['emCode', 'år'])
    df = df[df.år != 2020] # drop 2020 from dataframes, GDP lacks data for 2020
    df = df.set_index(['agg_næring', 'år'])
    

    return df

# Tidy and apply correct index in df
df_list = [df.pipe(tidyfunc_aggsector) for df in df_list]
df_list = [df.reset_index() for df in df_list]

#%%

# Create master dataframe
df = pd.concat(df_list, axis = 1)

# Remove duplicate columns
df = df.loc[:,~df.columns.duplicated()]

# Define columns to keep
columns = ['sub_gdp', 'sub_GWh', 'GWh_Fossil', 'sub_mtCO2e', 'emCode_parent', 'næring', 'år']

# Apply mapping to dataframe
df = df[columns]


"""
Note to self, create kaya dataframe for sub sectors
"""


#%%

"""
Make 10 sector kaya dataframe

"""

# Make copy dataframe
df10 = df.copy()

# Aggregate from sub-sector level to sector level
df10 = df10.groupby(['emCode_parent', 'år'])['sub_gdp', 'sub_GWh', 'GWh_Fossil', 'sub_mtCO2e'].sum().reset_index()

# Apply sector names
df10 = emCode_parent_to_sector(df10)

# Map total gdp to year
d = dict(zip(gdp['år'], gdp['totBNP']))
df10['totGDP'] = df10['år'].map(d)



#%%

"""
Mapping sector gdp to dataframe containing sub-sector data
"""

# Create nested dict sorted by emCode_parent: {år : sec_gdp}
d = {n: grp.loc[n].to_dict('index')
  for n, grp in gdp10[['sec_gdp', 'emCode_parent', 'år']].set_index(['emCode_parent', 'år']).groupby(level='emCode_parent')}

# Define mapping function, extracts value from dict based on content in two dataframes
def get_secgdp(key, value):    
    v = d[key][value]
    return v

# Map nested dict to dataframe, yields sector gdp based on emCode_parent and år
df['sec_gdp'] = df.apply(lambda x: get_secgdp(x['emCode_parent'], x['år']), axis=1).apply(pd.Series)



#%%

def kaya_prepper_func_sub(df):
    
    df = df.set_index(['næring', 'år'])
    
    # df['sec_gdp'] = df['sec_gdp']

    # Create gdp per capita
    df['subgdp/secgdp'] = df['sub_gdp'] / df['sec_gdp'] 
    
    # Create energy efficiency
    df['GWh_gdp']= df['sub_GWh'] / df['sub_gdp']
    
    # Create efficiency of fossil energy
    df['fossGWh_totGWh'] = df['GWh_Fossil'] / df['sub_GWh']
    
    # Create carbon fuel effiency
    df['mtCO2e_fossGWh'] = df['sub_mtCO2e'] / df['GWh_Fossil']
    
    # Add emissions
    df['mtCO2e'] = df['sub_mtCO2e']
    
    df = df[['sec_gdp', 'subgdp/secgdp', 'GWh_gdp', 'fossGWh_totGWh', 'mtCO2e_fossGWh', 'mtCO2e']]
    
    test = df.copy().round(2)

    test['prod']= test[['sec_gdp', 'subgdp/secgdp', 'GWh_gdp', 'fossGWh_totGWh',
           'mtCO2e_fossGWh']].prod(axis=1)
    
    if test['prod'].equals(test['mtCO2e']):
    
        return df

# Make finalised sector df with augmented kaya indicators
df_kaya = df.pipe(kaya_prepper_func_sub)



# # Commit to dataframe
# df_kaya.to_csv('./LMDI_ready/subsector_kaya.csv')

#%%

def kaya_prepper_func_sectors(df):
    
    df = df.set_index(['agg_næring', 'år'])
    
    # df['sec_gdp'] = df['sec_gdp']

    # Create gdp per capita
    df['secGDP_totGDP'] = df['sub_gdp'] / df['totGDP'] 
    
    # Create energy efficiency
    df['GWh_gdp']= df['sub_GWh'] / df['sub_gdp']
    
    # Create efficiency of fossil energy
    df['fossGWh_totGWh'] = df['GWh_Fossil'] / df['sub_GWh']
    
    # Create carbon fuel effiency
    df['mtCO2e_fossGWh'] = df['sub_mtCO2e'] / df['GWh_Fossil']
    
    # Add emissions
    df['mtCO2e'] = df['sub_mtCO2e'].round(2)
    
    df = df[['totGDP', 'secGDP_totGDP', 'GWh_gdp', 'fossGWh_totGWh', 'mtCO2e_fossGWh', 'mtCO2e']]
    
    df['res'] = df['mtCO2e'] - df[['totGDP', 'secGDP_totGDP', 'GWh_gdp', 'fossGWh_totGWh',
           'mtCO2e_fossGWh']].prod(axis=1).round(2)
    
    if df['res'].sum() == 0:
    
        return df
    
    else: print('Product of factors is not equal to emissions, data does not form identity')
    
# Apply kaya transformation func to 
df10 = df10.pipe(kaya_prepper_func_sectors)

# # Export to csv
# df10.to_csv('sektor_kaya.csv')



