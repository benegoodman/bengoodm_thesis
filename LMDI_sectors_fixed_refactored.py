# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:05:16 2022

@author: bened
"""

import os

# Set working directory to where PyLMDI module is located

# Laptop directory
# os.chdir(r"C:\Users\bengo\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Python stuff\PyLMDI")

# Desktop directory
os.chdir(r"C:\Users\bened\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Python stuff\PyLMDI")



import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


"""
Insert credit and source for PyLMDI library

"""

from PyLMDI import PyLMDI

# Desktop directory
os.chdir(r"C:\Users\bened\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Python stuff\Functions")

# Laptop directory
# os.chdir(r"C:\Users\bengo\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Python stuff\Functions")



#%%

"""
Create nested dict of dataframes in order to do operations, including LMDI to dataframes quickly

"""

def df_to_nested_dict(df):
    
    #create unique list of sectors
    sectors = df.næring.unique()
    
    #create a data frame dictionary to store your data frames
    dfd = {elem : pd.DataFrame for elem in sectors}
    
    # Create nested dict where each dataframe containing sector number is key
    for key in dfd.keys():
        dfd[key] = df[:][df.næring == key]
        
    return dfd # dict containing dataframes

#%%

class LMDI_machine_agg():
    
    
    def __init__(self, df, year, mode):
       
        """
        Takes drivers, and finds out how much each driver affects mtCO2e emissions
        Yields results in either absolute numbers, or as % effect on emissions
        
        Inputs:
            df = dataframe
            year = int indicating start year of analysis
            mode = string indicating mode of decomposition. Must be set to 'add' or 'mul'.
                add = additive decomposition
                mul = multiplicative decomposition
                
            Note: needs column mtCO2e to be present to work
        
        Input order:
            "totGDP"           : ans[1],           
            "sec_gdp/totGDP"   : ans[2],           
            "totGWh/sec_gdp"    : ans[3],           
            "fossGWh/totGWh"    : ans[4],           
            "fossGWh/mtCO2e"    : ans[5]    
            
        
        
        Output:
            df with absolute or %-effect on emissions in time period

        """
       
        
        self.df0 = df.set_index(['år'])
        self.df1 = df.shift(-1).dropna().set_index(['år'])
        self.year0 = year
        self.year1 = year + 1
        self.mode = mode
        self.drivers = list(self.df0.loc[:, self.df0.columns != 'mtCO2e']) # Every column except mtCO2e
    
    def LMDI_decomposer_agg(self):
        
        # Emissions for t and t+1
        C0 = list(self.df0['mtCO2e'].loc[self.df0.index == self.year0]) # emissions for t0
        Ct = list(self.df1['mtCO2e'].loc[self.df1.index == self.year1]) # emissions for t+1, as variables have been shifted in dataframe

        # define drivers of emissions for t and t+1
        factors0 = self.df0[self.drivers].loc[self.df0.index == self.year0]            # drivers in t0
        factors1 = self.df1[self.drivers].loc[self.df1.index == self.year1]           # drivers for t+1
    
        # Reshape drivers of emissions for t and t+1 to vertical vectors
        X0 = np.array(factors0).reshape([-1,1])
        Xt = np.array(factors1).reshape([-1,1])
        
        # LMDI additive decomposer function
        LMDI = PyLMDI(Ct,C0,Xt,X0)
        ans = []
        
        try:
            if self.mode == 'add':
                ans = LMDI.Add() # Additive decomposition
            
            elif self.mode == 'mul':
                ans = LMDI.Mul() # Multiplicative decomposition
                
        except:
            print('Mode must be set to either additive (keyword: add) or multiplicative (keyword: mul) decomposition')
        
        results = {}
        results = {
            "totGDP"           : ans[1],           # contributions from overal activity in the economy
            "sec_gdp/totGDP"   : ans[2],           # contributions from ratio of subsector gdp/sector gdp (sector structure)
            "totGWh/sec_gdp"    : ans[3],           # contributions from energy efficiency per unit of GDP
            "fossGWh/totGWh"    : ans[4],           # contributions from efficiency of fossil energy
            "mtCO2e/fossGWh"    : ans[5]            # contributions from efficiency of carbon emissions per unit of fossil energy
                       
            }
        
        return results

#%%

class LMDI_analysis():

    def __init__(self, df, start, stop, mode):
        
        self.df = df
        self.start = start
        self.stop = stop
        self.mode = mode
        
    def LMDI_analysis_func_v3(self):
        
        """
        What it does:
            Runs the LMDI_machine and subsequent decomposer n times and stores results in a dataframe
        
        Inputs:
            df = dataframe
            start = int indicating start year of chained decomposition
            stop = int indicating stop year
            mode = str 'add' or 'mul' indicating additive or multiplicative decomposition
            
        Process:
            Creates cloned dataframe and shifts data back one step
            Concatenates original and cloned dataframe
            Creates empty list to store results in
            Runs the LMDI chained decomposer function from start to stop
            Appends results for each year to dataframe
            
        Output:
            Nested dictionary with dataframes containing results
        
        """
        
        try:
            
            # Define empty list to store results in
            res = []
        
            # Run loop with additive LMDI for the years 1990 - 2020. Yields chained results 
            for years in list(range(self.start, self.stop)):
                res.append(LMDI_machine_agg(self.df, years, self.mode).LMDI_decomposer_agg()) # appends results from each year to list
                
                years += 1
                
            #Add results to dataframe
            result = pd.DataFrame(res[0:],columns=res[0])
            result.index = list(range(self.start, self.stop))
            
            return result
        
        except:
            print('Dataframe must be of type dataframe, start and stop must be of type int. Start and stop cannot be longer than the dataframe, mode must be of type str. Df must be called df')


#%%

def rename_shift_func(df):
    
    """
    Shifts year by -1 (90 becomes 91) in results column
    
    Reason for shifting year by -1:
        When doing the LMDI analysis the result is the difference between t and t+1.
        First year in data is 1990, hence first result year is 1991 and last year is 2020
    """
    
    df = df.reset_index()
    df = df.rename(columns={'index': 'year'})
    year = df['year'].shift(-1).fillna(2019).astype('int')
    df['year'] = year
    df = df.set_index('year')
    
    return df

#%%

# Dekstop
df = pd.read_csv(r"C:\Users\bened\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\Ferdige_data\LMDI_ready\sektor_kaya.csv")

def identity_assurance_func(df):
    
    """
    Input: Dataframe with factors ready for composition
    Operation: 
        Calculates difference between product of factors and emissions
    Returns: 
        If residual column is 0, returns dataframe ready for decomposition
    """
    
    factors = ['totGDP', 'secGDP_totGDP', 'GWh_gdp', 'fossGWh_totGWh',
           'mtCO2e_fossGWh']
    
    df['res'] = df['mtCO2e'] - df[factors].prod(axis=1).round(2) # round to eliminate rounding errors
    
    # Assert that residuals are 0
    if df['res'].sum() == 0:
        return df
    
    else:
        return print('Factors are not equal to emissions')

# Apply identity assurance function
df = df.pipe(identity_assurance_func)

# Rename columns
df = df.rename(columns={'agg_næring' : 'næring', 'secGDP_totGDP': 'secGDP/totGDP'})

# Make nested dict of dataframes with næring as key
dfd_agg = df_to_nested_dict(df)


print('Sectors present in nested dictionary:')
print(dfd_agg.keys())



#%%

# Create dict to store multiplicative results
res_agg_mul = {}

# Create dict to store additive results
res_agg_add = {}

# Define column list for dicts
col_list = ['år', 'totGDP', 'secGDP/totGDP', 'GWh_gdp', 'fossGWh_totGWh',
       'mtCO2e_fossGWh', 'mtCO2e']

# Get multiplicative results - store to nested dict
for k in dfd_agg:
    res_agg_mul[k] = dfd_agg[k][col_list]
    res_agg_mul[k] = LMDI_analysis(res_agg_mul[k], 1990, 2019, 'mul').LMDI_analysis_func_v3()
    res_agg_mul[k] = res_agg_mul[k] - 1  # remove 1 from multiplicative results to get correct change

# Get additive results, store to nested dict
for k in dfd_agg:
    res_agg_add[k] = dfd_agg[k][col_list]
    res_agg_add[k] = LMDI_analysis(
        res_agg_add[k]
        , 1990
        , 2019
        , 'add'
        ).LMDI_analysis_func_v3()
    
"""
Results: Chained additive and multiplicative LMDI-IDA results for 1991 to 2019
Years in results dataframes are incorrectly shown due to how LMDI analysis objects work
This is fixed in the next section

"""

#%%

# Shift indexes in nested dataframes by 1 year by applying function
for k in res_agg_mul:
    res_agg_add[k] = res_agg_add[k].pipe(rename_shift_func)

#%%

# Function for summing results per sector - sum of all years in order to get total change
def result_sum_func(n_dict):
    
    """
    Input: Nested dict with chained results from LMDI-IDA
    Output: Nested dict with sum of contribution from drivers
    
    """
    
    # Pull out sector names in dictionary
    sectors = list(n_dict.keys())
    
    # Create place to store results
    d_sum = {}
    
    # Define loop integer
    i = 0
    
    # Summing loop
    for k in n_dict:
    
        d_sum[k] = pd.DataFrame(n_dict[k].sum())                                # sums contribution for each driver within results
        d_sum[k]['sector'] = sectors[i]                                         # adds corresponding sector as column in dataframe
        d_sum[k]['factor'] = d_sum[k].index                                     # makes factors into column
        d_sum[k] = d_sum[k].pivot(columns='factor', values=0, index = 'sector') # pivots result dataframe into pretty format
        
        i += 1
    
    return d_sum

# Get additive results in nested dict
sum_add =  result_sum_func(res_agg_add)

# Get multiplicative results results in nested dict
sum_mul = result_sum_func(res_agg_mul)

#%%

def nested_dict_to_df(n_dict):
    
    """
    Input: Nested dict with results
    Output: Dataframe with drivers as columns, sectors as index
    
    """
    
    # Make dataframe from nested dict
    df = pd.concat({k: pd.DataFrame(v).T for k, v in n_dict.items()}, axis=1)
    
    # Drop multilevel columns
    df.columns = df.columns.droplevel()
    
    # reset index, tranpose dataframe
    df = df.reset_index().transpose()
    
    # set factors as column headers
    df.columns = df.iloc[0]
    
    # remove row containing column names, make dataframe into float
    df = df.iloc[1: , :].astype('float')
    
    # Redefine column order
    df = df[['totGDP', 'sec_gdp/totGDP', 'totGWh/sec_gdp', 'fossGWh/totGWh', 'mtCO2e/fossGWh']]
    
    return df

# Get multiplicative results - ready for export
result_mul = nested_dict_to_df(sum_mul)

# Get additive results - ready for export
result_add = nested_dict_to_df(sum_add)


#%%

"""
Create residual column for test dict, then turn to dataframe

I basically need to test if additive decomposition factors add up to emissions

THIS is a correct version of the data
"""


def res_test_func(res_agg_add):
    
    # Get additive results
    test = res_agg_add

    # Add differenced emissons to results (change in emissions)
    # Takes trend out of emissions
    for k in dfd_agg:
        test[k]['mtCO2e'] = dfd_agg[k].set_index('år')['mtCO2e']
        test[k]['mtCO2e_d'] = test[k]['mtCO2e'].diff(1)
    
    # Make dataframe from nested dict
    em = pd.concat({k: pd.DataFrame(v).T for k, v in test.items()}, axis=1)
    
    # reset index, tranpose dataframe
    em = em.reset_index().transpose()
    
    # set factors as column headers
    em.columns = em.iloc[0]
    
    # Drop first row
    em = em.iloc[1: , :]
    
    # # Drop sector column, turn into float
    em = em.astype('float')
    
    # Add up factors, see if they are the same as differenced emissions
    em['res'] = em['mtCO2e_d'] - em.iloc[:, 0:5].sum(axis=1).round(2)
    
    try: 
        em['res'].sum() == 0
        
        return em
        
    except:
        print('residuals not 0, LMDI decomposition incorrect')

em = res_test_func(res_agg_add)

"""

Works :)
"""

#%%

# Create additive drivers aggregate columns -> yields total effect across sectors
add_agg = pd.DataFrame(em.iloc[:,0:5].sum()).transpose()


#%%

"""
Define lists of drivers and english sectors for plots
"""

# Labels dravers
factors = ['GDP growth', 'Ec. structure'
                 , 'En. efficiency'
                 , 'Share of fossil energy'
                 , 'CO2e eff. of foss. fuels'
                 # , 'Total Emissions Change'
                 ]


# Labels for sectors
sectors = ['Agri- & aquaculture and forestry'
                 ,'Mining and petroleum'
                 , 'Industry'
                 , 'Energy, water, sewage and waste management'
                 , 'Construction'
                 , 'Retail & hospitality'
                 , 'Other serivces'
                 , 'Transport'
                 , 'Education, healthcare and social services'
                 , 'Public administration and defence'] # labels for y-axis

## make dict of norwegian and english names
d = dict(zip(result_add.index.unique(), sectors))

result_add['sectors'] = sectors
result_add = result_add.set_index('sectors')

#%%

"""
Plot of additive decomposition total effect by driver
"""

def barplot_additive_aggregate(df, factors):
    
    df.columns = factors
    
    sns.set(style='whitegrid', rc = {'figure.figsize':(12,6)}) # need to set size
    plot = sns.barplot(data = df, palette='mako')
    # plot.bar_label(plot.containers[0])
    plot.set(ylabel ='mktCO2e', xlabel='Driver')
    # plt.legend(list(df.columns))
    plt.suptitle('Absolute effect of drivers on total emissions, 1990 - 2019', size = 18)

    plt.show()



# Apply plot function to results
add_agg.pipe(barplot_additive_aggregate, factors)

plt.close()


#%%

"""
Plotting of drivers across sectors

"""

def barplot_additive_drivers(df, sectors, title):
    
    
    df = pd.DataFrame(df)
    sns.set(style='whitegrid', rc = {'figure.figsize':(12,6)}) # need to set size
    plot = sns.barplot(y = df.index, x = df.iloc[:,0], palette='mako')
    # plot.bar_label(plot.containers[0])
    plot.set(ylabel = sectors, xlabel='mktCO2e')
    # plt.legend(list(df.columns))
    plt.suptitle(title, size = 18)

    plt.show()


plt.close()

# Plot results of economic activity
barplot_additive_drivers(
    result_add['totGDP'], 'Sector', 
    'Effect of total GDP-growth on sectoral emissions, 1990-2019')

# Plot results of structural change
barplot_additive_drivers(
    result_add['sec_gdp/totGDP'], 'Sector', 
    'Effect of structural change on sectoral emissions, 1990-2019')

# Plot results of energy efficiency
barplot_additive_drivers(
    result_add['totGWh/sec_gdp'], 'Sector', 
    'Effect of energy intensity on sectoral emissions, 1990-2019')

# Plot results of fossil energy efficiency
barplot_additive_drivers(
    result_add['fossGWh/totGWh'], 'Sector', 
    'Effect of fossil energy efficiency on sectoral emissions, 1990-2019')

# Plot results of carbon intensity of fossils
barplot_additive_drivers(
    result_add['mtCO2e/fossGWh'], 'Sector', 
    'Effect of carbon intensity of fossil energy on sectoral emissions, 1990-2019')




#%%

# Heatmapper function for summarising results
def heatmapper_func(df, factors, sectors):
    
    df['total emissions change'] = df.sum(axis=1)

    mask = np.zeros((10, 6))
    mask[:,5] = True
    
    sns.set(rc = {'figure.figsize':(16, 8)}, style="whitegrid")
    
    
    
    # # Make heatmap of data
    # my_map = sns.diverging_palette(-800, -700, s=60, l=50, center='light',  as_cmap=True)
    
    sns.heatmap(df, mask=mask, cbar=True, cmap='Spectral_r',)
    plt.suptitle('Contribution of drivers per sector, 1990-2019', y=0.95, x=0.4, size =24) # Consider different colour scheme ...blue?
    sns.heatmap(df, alpha=0, cbar=False, annot=True
                , annot_kws={"size": 16, "color":"g"}, fmt='g',
                xticklabels=factors, yticklabels=sectors)
    plt.xlabel('Factor')

    plt.show()
    

# Run heatmapper on additive results
heatmapper_func(result_add, factors, sectors)

# Plot results of carbon intensity of fossils
barplot_additive_drivers(
    result_add['total emissions change'], 'Sector', 
    'Total emissions change per sector, 1990-2019')



#%%


def barplot_additive_single(df, factors, title):
    
    sns.set(style='whitegrid', rc = {'figure.figsize':(12,6)}) # need to set size
    plot = sns.barplot(data = df, palette='mako', ci=None)
    # plot.bar_label(plot.containers[0])
    plot.set(ylabel ='mktCO2e', xlabel='Driver')
    # plt.legend(list(df.columns))
    plt.suptitle(title, size = 18)

    plt.show()


tran = pd.read_csv(r'C:\Users\bened\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\Ferdige_data\LMDI_ready\subsector_kaya.csv')

# Subset transportation sub sectors
sjo = tran.loc[tran['næring'].str.contains('Utenriks sjøfart')].set_index(['næring'])
tran = tran.loc[tran['næring'].str.contains('Transport ellers')].set_index(['næring'])

# Run LMDI analysis for transport subsectors
sjo = LMDI_analysis(sjo, 1990, 2019, 'add').LMDI_analysis_func_v3()
tran = LMDI_analysis(tran, 1990, 2019, 'add').LMDI_analysis_func_v3()

sjo.columns = factors
tran.columns = factors

# Plot results
barplot_additive_single(sjo, factors, 'Effect of drivers - international sea freight')
barplot_additive_single(tran, factors, 'Effect of drivers - Other transport')


#%%

tran2 = res_agg_add['Transport']

sns.lineplot(data=tran2['sec_gdp/totGDP'])
sns.lineplot(data=tran2['totGWh/sec_gdp'])

#%%




"""
Note to self: Works! Now I need to do multiplicative ones....
"""
em_sum = em.reset_index().dropna().groupby('level_0').sum()
                       
em_sum['res'] = em_sum['mtCO2e_d'] - em_sum.iloc[:, 0:5].sum(axis=1).round(2)

em_sum = em_sum.reset_index().rename(columns={'level_0' : 'Sector'}).set_index('Sector')

em_sum.iloc[:, 0:5].pipe(heatmapper_func, factors, sectors)

#%%

def residual_check_func(result_add):
    
    tot_em = pd.read_csv(r"C:\Users\bened\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\SSB\Rådata\total_sectoral_emissions.csv", 
                         encoding='ANSI', sep=';', header=1)
    
    # Calculate difference in emissions from secotral emissions between 1990 and 2019
    diff = tot_em.iloc[29,3] - tot_em.iloc[0,3] # Shows difference in emissions between 1990 and 2019
    
    # Get total change in sectoral emissions according to decomposition
    decomp_res = result_add['total emissions change'].sum()
    
    # print output
    print()
    print('Difference between actual change in sectoral emissions and decomposed emissions:')
    print(diff - decomp_res)

# apply function
residual_check_func(result_add)

#%%

# Stacked barplot of changes over time
subset = ['fossGWh/totGWh', 'mtCO2e/fossGWh', 'res',
       'sec_gdp/totGDP', 'totGDP', 'totGWh/sec_gdp']

# Pivot results
em_wide = em[subset].reset_index().melt(id_vars=['level_0', 'year']
                                , var_name = 'Driver'
                                )


# Rename sectors column and map english names to sectors
em_wide = em_wide.rename(columns={'level_0' : 'Sector'})
em_wide['Sector'] = em_wide['Sector'].map(d)
d = dict(zip(em_wide['Driver'].unique(), factors))
em_wide['Driver'] = em_wide['Driver'].map(d)

em_wide_d = em_wide.groupby(['Driver', 'year']).sum('value').reset_index()

em_wide = em_wide_d.pivot_table(index='year', columns='Driver', values='value')



# Make plot

#%%

# set plot style: grey grid in the background:
sns.set(style="dark", rc = {'figure.figsize':(12,5)})
my_cmap = sns.color_palette('crest', as_cmap=True)

fig, ax = plt.subplots() 
ax = em_wide.plot(kind='bar', stacked=True, cmap=my_cmap)
ax.set_ylabel('ktCO2e')
plt.suptitle('Net effect of drivers per year 1990 - 2019', size=16)
plt.legend(loc="lower left", ncol=6, bbox_to_anchor=(0, -0.3))

ems = em.groupby('year').sum()['mtCO2e']



# adding line
ax2 = ax.twinx()
ax2.plot(ax.get_xticks(), ems.pct_change(1)*100, color='k', label='ktCO2e_d')
ax2.set_ylim(20, -20)
ax2.set_ylabel('% change in emissions (ktCO2e)')

# plt.tight_layout()
plt.show()


# set plot style: grey grid in the background:
sns.set(style="darkgrid", rc = {'figure.figsize':(12,5)})
ems.plot()
plt.suptitle('Aggregate sectoral emissions 1990 - 2019')
plt.ylabel('ktCO2e')
plt.xlabel('Year')

print()
print()
print('Growth in sectoral emissions 1990 - 2019')
print(65836/59618)


print(65836 - 59618)

#%%

# tran = df.loc[df['næring'] == 'Transport'].set_index('år')['secGDP/totGDP'].plot()
# df.loc[df['næring'] == 'Bergverksdrift og utvinning av råolje og naturgass, inkl. tjenester'].set_index('år')['secGDP/totGDP'].plot()

# Plot of sectoral balances

## make dict of norwegian and english names
d = dict(zip(df.næring.unique(), sectors))

df_plot = df

df_plot['næring'] = df['næring'].map(d)

df_plot['sec_bal'] = df_plot['secGDP/totGDP']

sns.lineplot(x='år'
             , y='sec_bal'
             , hue='næring'
             , style='næring'
             , data=df_plot
             , linewidth=3)
plt.legend(loc="upper right", bbox_to_anchor=(1.42, 0.75))