# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:15:33 2022

@author: bened
"""
import glob
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np


pd.options.mode.chained_assignment = None


path = r'C:/Users/bened/OneDrive - Norwegian University of Life Sciences/Fag/Masteroppgave/Data/Machine learning data/data_raw'

os.chdir(path)


#%%


# merge function for old and new kommuner
def kommune2019_to_kommune2020(df):
    
    """
    Input: df:dataframe
    
    
    Operation:
        - makes dic
        - reads emCode_parent column
        - applies aggregate sector name to emCode in new column 'næring'
        
    Returns: 
        df with aggregate sector names
    """

    
    df['region'] = df['region'].str.lower()

    # Associate parent codes and sector columns via dict-mapping
    koder = pd.read_excel(r'C:\Users\bened\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\SSB\Klassifisering\gamle_nye_kommuner\fylker-kommuner-2019-2020-alle.xlsx')
    koder = koder[['Kommunenavn 2019', 'Kommunenavn 2020']]
    for columns in koder.columns:
        koder[columns] = koder[columns].str.lower() 
    d = dict(zip(koder['Kommunenavn 2019'], koder['Kommunenavn 2020']))
    
    # Map new kommunennames to new column
    df['Kommune'] = df['region'].map(d)
    
    df['Kommune'] = df['Kommune'].str.title()
    
    return df
    

#%%

def query_to_df(url, urlmeta, file):
    
    """
    Function: Imports df from json query via SSB's API

    Inputs:
       url = api url for relevant table
       urlmeta = api url for relavant metadata for table
       file = path where json query file lies
      
    Outputs:
        Dataframe containing data

    """
    
    file = open(file)
    query = json.load(file)
    result = requests.post(url, json = query)
    dataset = pyjstat.Dataset.read(result.text)
    df = dataset.write('dataframe')
    df['år'] = df['år'].astype('int')
    
    
    return df

#%%


# in the folder
def folder_csv_importer():
    
    """
    Looks through working directory, imports all csv files and gives names
    based on filename. 
    
    Operations:
        Removes special characters in columns indicating missing data
        Replaces missing data with null values
    
    Returns:
        Nested dict containing dataframes

    """
    
    path = os.getcwd()
    
    # use glob to get all the csv files in folder
    csvs = glob.glob(os.path.join(path, "*.csv"))
      
    # reads in filenames, removes .csv suffix
    fns = [os.path.splitext(os.path.basename(x))[0] for x in csvs]
    
    # Import all csv files as nested dict with dfs inside
    d = {}
    d2 = {}
    d3 = {}
    # samisk = {}
    for i in range(len(fns)):
        d[fns[i]] = pd.read_csv(csvs[i], encoding='ANSI', sep=';', header=1)
    
    # Remove special characters indicating missing data
    for key in d:
        d[key] = d[key].replace(['.',':', '..', '-', ';'], np.nan)
        
        # Convert all columns with numbers to numbers, keep strings
        d2[key] = d[key].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',',''), errors='coerce'))
        d3[key] = d2[key].fillna(d[key])
        
    # Remove kommunenumbers from dataframes
    for key in d3.keys():
        if 'region' in d3[key]:
            
            # Characters to remove
            pattern = '|'.join([
                                '0', '1', '2', '3'
                                , '4' , '5', '6', '7', '8'
                                ,'9'
                                ])
            
            # Removes characters in dataframe itineratively per dataframe
            d3[key]['region'] = d3[key]['region'].str.replace(pattern, '')
            d3[key]['region']= d3[key]['region'].str.replace('K. ', '')
            
            # Remove kommuner sans location 
            d3[key] = d3[key][d3[key]['region'].str.contains('Uoppgitt')==False]
            d3[key] = d3[key][d3[key]['region'].str.contains('Unknown')==False]
            
            # Remove unnecessary substrings
            d3[key]['region'] = d3[key]['region'].str.rstrip('[(-)]')
            d3[key]['region'] = d3[key]['region'].str.rstrip(' ')
            d3[key]['region'] = d3[key]['region'].str.lstrip(' ')
            
            # Remove parathese all together
            d3[key]['region'] = d3[key]['region'].str.replace('[()]', '')
        
        else:
            pass
        
    return d3

# apply function
data = folder_csv_importer() 

print('Datasets present in nested dict:')
print(data.keys())

#%%

"""
Fix of employee data, merge into one separate dataframe
Missing data in some sectors at random - imputation necessary

Deletion of data was considered, but that would imply deleting data from 302 kommuner ... which would
lead me to have a paltry 54 municipalities to work with. Which would be worse.

So... imputations necessary. Out of 12816 observations, 1100 were missing (most in 2020)

Imputing is the least time consuming at this point. Had there been more time ML could be used here for time series
although 2020 is an outlier year so....

In any case we just roll with it. This is only 1 of many variables

"""


def emp_fix(df, val_name):
    
    # Set index
    df = df.set_index(['region', 'næring (SN2007)'])
    
    # Save null values to dataframe
    na = df[df.isna().any(axis=1)]
    
    # Interpolate missing values
    df = df.apply(pd.to_numeric).interpolate(method='linear')
    
    df = df.groupby('region').sum()
    df.columns = df.columns.str.strip('Sysselsatte personer etter arbeidssted ')
    df = df.reset_index()
    
    df['region']= df['region'].str.replace('K. ', '')
    
    df = df.melt(id_vars='region', var_name='År', value_name=val_name).set_index(['region','År'])
    
    return df, na

d1 = emp_fix(data['ansatte_primær'], 'employees in primary sector')
d2 = emp_fix(data['ansatte_sekundær'], 'employess in secondary sector')
d3 = emp_fix(data['ansatte_tertier'], 'employees in tertiary sector ')

emp = pd.concat([d1[0], d2[0], d3[0]], axis = 1).dropna().reset_index()



na = pd.concat([d1[1], d2[1], d3[1]], axis = 0).reset_index()

na.isnull().count()

del d1, d2, d3

#%%

def samisk_removal(df):

    """
    Removes Sami name suffixes via dict-mapping

    """    
    
    df = df
    
    #Subsets all rows where - is present
    samisk = df.loc[df['region'].str.contains('-')]
    
    # List of sami names
    sam = ['Aurskog-Høland', 'Deatnu - Tana', 'Evenes - Evenássi',
           'Fauske - Fuossko', 'Guovdageaidnu - Kautokeino',
           'Gáivuotna - Kåfjord - Kaivuono', 'Kárásjohka - Karasjok',
           'Loabák - Lavangen', 'Midt-Telemark', 'Nord-Aurdal', 'Nord-Fron',
           'Nord-Odal', 'Porsanger - Porsángu - Porsanki\xa0',
           'Raarvihke - Røyrvik', 'Snåase - Snåsa', 'Sortland - Suortá',
           'Stor-Elvdal', 'Storfjord - Omasvuotna - Omasvuono', 'Sør-Aurdal',
           'Sør-Fron', 'Sør-Odal', 'Sør-Varanger', 'Unjárga - Nesseby', 'Fauske - Fuosko',
           'Hamarøy - Hábmer', 'Divtasvuodna - Tysfjord',
           'Sør-Varanger', 'Harstad - Hárstták', 'Nordreisa - Ráisa - Raisi',
           'Porsanger - Porsángu - Porsanki']
    
    # List of norwegian names
    nor = ['Aurskog-Høland', 'Tana', 'Evenes',
           'Fauske', 'Kautokeino',
           'Kåfjord', 'Karasjok',
           'Lavangen', 'Midt-Telemark', 'Nord-Aurdal', 'Nord-Fron',
           'Nord-Odal', 'Porsanger',
           'Røyrvik', 'Snåsa', 'Sortland',
           'Stor-Elvdal', 'Storfjord', 'Sør-Aurdal',
           'Sør-Fron', 'Sør-Odal', 'Sør-Varanger', 'Nesseby', 'Fauske',
           'Hamarøy', 'Tysfjord',
           'Sør-Varanger', 'Harstad', 'Nordreisa',
           'Porsanger']
    
    # Associate lists above
    d = dict(zip(sam, nor))
       
    # Map norwegian names  to 
    samisk['region_fix'] = samisk['region'].map(d)
    
    samisk = samisk.drop('region',1)
    
    samisk = samisk.rename(columns={'region_fix' : 'region'})
    
    df.update(samisk)
    
    df['region'] = df['region'].str.title()
    
    return df

#%%

def idx_func(df):
    df[['region', 'År']] = df[['region', 'År']].astype('str')
    df = df.sort_values(['region', 'År']).set_index(['region', 'År'])
    return df

#%%

# Check amount of unique regions
print('Amount of regions in emplpyment data')
print(emp.region.nunique())

# Remove samisk from employment data
emp = samisk_removal(emp).pipe(idx_func)

#%%

"""
Fix pop data, no data dropped

Operation: Made percent change

Note: Might be something wrong here - more rows than other dataframes


"""


def pop_fix(df):
    df = df.copy()
    df['region'] = df['region'].str.replace('K-', '')
    df.columns = df.columns.str.strip('Personer ')
    df = df.rename(columns={'gi' : 'region'})
    # df = df.set_index('region')
    df = df[['region', '2008', '2009', '2010', '2011', '2012',
           '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']]
    
    df = df.melt(id_vars='region', var_name='År', value_name='Population')
    
    #Create population growth
    df1 = df.pivot(columns='region', index = 'År', ).pct_change()*100
    
    df = df.loc[df['År'] > '2008']
    
    return df, df1

# Apply fixer function, export touple
pop = pop_fix(data['befolkningstall'])

# flip population growth df into same shape as population
pop_g = pop[1].reset_index().melt(id_vars = 'År')

# set index to ready for concatenation
pop = pop[0].set_index(['År', 'region'])

# Ready pop growth for concatenation
pop_g = pop_g[['År', 'region', 'value']].rename(columns={'value' : 'Population growth rate'}).set_index(['År', 'region'])

# Concat, drop 2008 from observations
pop = pd.concat([pop, pop_g], axis=1).reset_index().sort_values(['region', 'År'])

# Remove sami names
pop = pop.pipe(samisk_removal)

# Check for null values
na = pop.loc[pop['Population'].isna()]

print('Amount of regions present in population data:')
print(pop.region.nunique())

# drop null values (should only be 2008)
pop = pop.dropna().pipe(idx_func)

del pop_g

#%%

# concatenate into master df
df_master = pd.concat([emp, pop], axis =1)

# check null values - only contains data where data is 0
na = df_master[df_master.isna().any(axis=1)]

# Drop null values
df_master = df_master.dropna()

print(df_master.columns[df_master.columns.duplicated(keep=False)])

test = df_master.reset_index()

dup = test[test.duplicated(subset=['region','År'], keep=False)]


#%%)

"""
Fix boligdata & bygningsdata
Bolig = Boligbygg
Andre bygg enn boligbygg = Næringsbygg & sekundærboliger
Note: these are EXISTING buildings
Note: Sammenslått tidsserie

"""


def house_data_fix(df):
    
    
    bygg = df
    
    # Subset housing
    bolig = bygg.loc[bygg['bygningstype'] == 'Boligbygg']
    
    # Subset nonhousing
    bygg = bygg.loc[bygg['bygningstype'] == 'Andre bygg enn boligbygg']
    
    # Remove samisk 
    bolig = bolig.pipe(samisk_removal)
    bygg = bygg.pipe(samisk_removal)
    
    # Make long version dataset
    bolig = bolig.drop(['statistikkvariabel', 'bygningstype'], 1).melt(
        id_vars='region', value_name='n_housing', var_name='År'
        ).sort_values(['region', 'År'])
    
    # Make long version dataset
    bygg = bygg.drop(['statistikkvariabel', 'bygningstype'], 1).melt(
        id_vars='region', value_name='buldings', var_name='År'
        ).sort_values(['region', 'År'])
    
    
    return bygg, bolig

# Extract df with boligdata
bygg_m = data['eksisterende_byggmasse'].copy()

#apply house fix func
bygg_m = house_data_fix(bygg_m)

# Concat into dataframe
bygg_m = pd.concat([bygg_m[0],bygg_m[1]], axis=1)

# Drop duplicate columns
bygg_m = bygg_m.loc[:,~bygg_m.columns.duplicated()]

# Keep only observations up to 2020
bygg_m = bygg_m.loc[bygg_m['År'] <= '2020'].pipe(idx_func)

#%%

# Concat byggmasse into master df
df_master = pd.concat([df_master, bygg_m], axis = 1)

# check null values - only contains data where data is 0 again
na = df_master[df_master.isna().any(axis=1)]

# Drop null values
df_master = df_master.dropna()

print(df_master.columns[df_master.columns.duplicated(keep=False)])

test = df_master.reset_index()

dup = test[test.duplicated(subset=['region','År'], keep=False)]


#%%

"""
Electricity usage per consumer group

removed samisk
turned data to float
interpolated 2020 in 2 kommuner for 2 consumer groups:
    Mining and Industry
    Services
    
NOTE: Only goes from 2010 onwards, imputed for 2009

"""

# Create dataframe
el = data['elektrisitetsforbruk_pr_forbrukergruppe'].copy().pipe(samisk_removal).set_index(['region', 'forbrukergruppe']).astype('float')

# Fix column names
el.columns = el.columns.str.strip('Forbruk i alt ')

# Inspect missing data
na = el[el.isna().any(axis=1)]

# print missing values
print(na) # only 4 missing, can be imputed safely

# Add interpolation
na = na.interpolate(method='linear', axis = 1)

# Add interpolated results to electricity dataframe
el.update(na)

el['2009'] = np.nan

# Sort columns in asscending order
el = el.reindex(sorted(el.columns), axis=1)

el = el.interpolate(method='linear', axis = 1, limit_direction='backward')

# Pivot and melt to get proper long form data
el = el.reset_index().melt(id_vars=['region','forbrukergruppe'], var_name='År', value_name='el GWh')
el = el.pivot(columns=['forbrukergruppe'], index=['region', 'År'], values='el GWh')

el = el.rename(columns={'ALLE FORBRUKERGRUPPER' : 'GWh usage total'
                        , 'BERGVERKSDRIFT OG INDUSTRI MV.' : 'GWh Mining and industry'
                        , 'HUSHOLDNINGER OG JORDBRUK': 'GWh Households and agriculture'
                        , 'TJENESTEYTING MV.' : 'GWh Service-sector'})

el = el.reset_index()
el['el_imp'] = np.where(el['År'] == '2009', 1, 0)
el = el.pipe(idx_func)

#%%

def concat_func(df_master, df):
    
    
    # Concat byggmasse into master df
    df_master = pd.concat([df_master, df], axis = 1)
    
    # check null values - only contains data where data is 0 again
    na = df_master[df_master.isna().any(axis=1)]
    
    # Drop null values
    df_master = df_master.dropna()

    print(df_master.columns[df_master.columns.duplicated(keep=False)])

    test = df_master.reset_index()

    dup = test[test.duplicated(subset=['region','År'], keep=False)]
    
    return df_master, na, dup

# Apply function
df_master = concat_func(df_master, el)


#%%

"""
Havnegods
Operations:
    Aggregated from quartely to yearly data
    Added respective kommune
    
Location of ports were found by googling


Note:
    Narvik has near 0 data
    verdal lacks data from 2013 onwards

"""

# Extract data of havnegods, make yearly column
gods = data['Godsmengde_pr_havn'].copy()

def gods_tidy_func(df):
    
    gods = df
    
    gods['kvartal'] = gods['kvartal'].str.replace('K', 'Q')
    gods['År'] = gods['kvartal'].str[:4].astype('str')
    
    # Aggregate to yearly data
    gods = gods.groupby(['År' ,'havn', 'innenriks/utenriks'])['Godsmengde i TEU-containere (tonn)'].sum().reset_index()
    
    # Map kommnune to 
    komm = pd.read_csv(r'C:\Users\bened\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\Machine learning data\Associative dicts\havn_kommune_dict.csv', encoding='ANSI')
    d = dict(zip(komm.Havn, komm.Kommune))
    gods['region'] = gods['havn'].map(d)
    
    gods = gods.pivot_table(index=['region', 'År'], columns='innenriks/utenriks', values='Godsmengde i TEU-containere (tonn)')
    
    gods['tot freight (tons)'] = gods['Innenriks'] + gods['Innenriks']
    
    gods = gods.rename(columns={'Innenriks' : 'sea freight dom (tons)', 'Utenriks' : 'sea freight int (tons)' })

    return gods


# Apply tidy func
gods = gods_tidy_func(gods)

#%%

df_master = pd.concat([df_master[0], gods], axis = 1, join='outer')

# check null values - only contains data where data is 0 again
na = df_master[df_master.isna().any(axis=1)]

# Drop null values
df_master = df_master.fillna(0)

print(df_master.columns[df_master.columns.duplicated(keep=False)])

test = df_master.reset_index()

dup = test[test.duplicated(subset=['region','År'], keep=False)]


#%%

def old_kommuner_to_2020(df):
    
    # Remove kommuner sans location 
    df = df[df['region'].str.contains('Uoppgitt')==False]
    df = df[df['region'].str.contains('Unknown')==False]
    
    # Remove unnecessary substrings
    df['region'] = df['region'].str.rstrip('[(-)]')
    df['region'] = df['region'].str.rstrip(' ')
    df['region'] = df['region'].str.lstrip(' ')
    
    # Remove parathese all together
    df['region'] = df['region'].str.replace('[()]', '')
    
    # remove sami names from substrings
    df = df.pipe(samisk_removal)
    
    # make region lower case
    df['region'] = df['region'].str.lower()
    
    # Import old/new municipality file
    koder = pd.read_csv(r'C:\Users\bened\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\SSB\Klassifisering\gamle_nye_kommuner\fylker-kommuner-2019-2020-alle.csv', encoding='ANSI')
    koder = koder[['Kommunenavn 2019', 'Kommunenavn 2020']]
    
    # Make columns in municipality file lower case and remove intances of brackets ()
    for columns in koder.columns:
        koder[columns] = koder[columns].str.lower()
        koder[columns] = koder[columns].str.replace('[()]', '')
        
    # Create mapping dict
    d = dict(zip(koder['Kommunenavn 2019'], koder['Kommunenavn 2020']))
    
    # apply mapping dict
    df['Kommune'] = df['region'].map(d)
    
    df['Kommune'] = df['Kommune'].str.title()
    
    df = df.rename(columns={
        'region' : 'region_old',
        'Kommune' : 'region',
            }
        )
    
    return df



#%%

"""
Treatment of surface area of newly built buildings other than houses

    Unit: m2
    Variables: Started buidlings and finished buildings
    Started buildings is pretty much a lagged version of finished buildings
    
    Is a total variant of nybygget industribygg
    
    

"""

# Extract surface area under construction and newly finished
bm = data['nybygget_byggmasse']

# Reshape dataframe into table format, remove sami names
bm = bm.melt(
    id_vars=['statistikkvariabel', 'region']
    , var_name='År'
    , value_name='m2_built'
    ).pivot(
        index=['region', 'År']
        , columns='statistikkvariabel'
        , values='m2_built').reset_index().pipe(samisk_removal).set_index(['region', 'År'])


bm.isnull().any()

bm = bm.rename(columns={
    'Fullført bruksareal til annet enn bolig' : 'compl_area_nonhousing'
    , 'Igangsatt bruksareal til annet enn bolig' :'constr_area_nonhousing'
    })
        

#%%

df_master = pd.concat([df_master, bm], axis = 1, join='outer')

# check null values - only contains data where data is 0 again
na = df_master[df_master.isna().any(axis=1)]

# # Drop null values
# df_master = df_master.dropna()

print(df_master.columns[df_master.columns.duplicated(keep=False)])

test = df_master.reset_index()

dup = test[test.duplicated(subset=['region','År'], keep=False)]

#%%

"""
Surface area allocated used for aggriculture

Operations:
    Apply new municipality names
    Impute values linearly 

"""

# Extract surface area for aggriculture
m2_ag = data['jordbruksareal'].copy().set_index(['vekst', 'region'])
m2_ag = m2_ag.astype('float').reset_index()

na = m2_ag[m2_ag.isna().any(axis=1)]

m2_ag['region'].nunique()

# Apply new/old municipality associtor function
m2_ag = old_kommuner_to_2020(m2_ag).set_index(['vekst', 'region'])

# Extract null values
na = m2_ag.loc[m2_ag.isna().any(axis=1)].reset_index()

# If no null values for kommune were returned, dothe below
if na['region'].isna().any() == False:
    
    #Drop old regions
    m2_ag = m2_ag.drop('region_old', 1)
    
    """
    Note: Imputed quite a few values here due to the fact that the nature of the missing data was noise-like
    i.e. indication of data missing at random ...as far as I know there are no tests for this so missing at random
    was assumed
    """

    
    # Impute where possible
    m2_ag= m2_ag.interpolate(method='linear', axis=1)
    
    # m2_ag[['2009']] = np.nan

    # # Sort columns in asscending order
    # el = el.reindex(sorted(el.columns), axis=1)

    # Extract remaining null values
    # mean = m2_ag.loc[m2_ag.isna().any(axis=1)]

    # Aggregate from old to new kommuner 
    m2_ag = m2_ag.groupby(['vekst', 'region']).sum().reset_index()
    
    # Remove suffixes from column names
    m2_ag.columns = m2_ag.columns.str.replace('Jordbruksareal ', '')
    
    # Reshape dataframe into table format,
    m2_ag = m2_ag.melt(
        id_vars=['vekst', 'region']
        , var_name='År'
        , value_name='acre_agr_area'
        ).pivot(
            index=['region', 'År']
            , columns='vekst'
            , values='acre_agr_area').reset_index()
    
        
# If null values are preset: return error
else:
    print('old/new municipality matching was unsuccessful')

print(m2_ag['region'].nunique())

# rename variable
m2_ag = m2_ag.rename(columns={'Jordbruksareal i drift' : 'acre_agr_area'}).set_index(['region', 'År'])

# Extract null values
na2 = m2_ag.loc[m2_ag.isna().any(axis=1)].reset_index()

#%%

df_master = pd.concat([df_master, m2_ag], axis = 1, join='outer')

# check null values - only contains data where data is null
na = df_master[df_master.isna().any(axis=1)]

print(df_master.columns[df_master.columns.duplicated(keep=False)])

test = df_master.reset_index()

dup = test[test.duplicated(subset=['region','År'], keep=False)]

#%%

#Import road data
df_r = data["N driftsutgifter til komm veier og gater pr km -pre 2016"].copy().pipe(
    old_kommuner_to_2020
    ).set_index('region').drop('region_old', 1).astype('float')

df_r2 = data['N driftsutgifter til komm veier og gater pr km - post 2016'].copy().pipe(
    old_kommuner_to_2020
    ).set_index('region').drop('region_old', 1).astype('float')

# Remove ugly column names
df_r.columns = df_r.columns.str.replace(r'[(Nto. dr.utg. i kr pr. km kommunal vei og gate )]', '')
df_r2.columns = df_r2.columns.str.replace(r'[(Netto driftsutgifter til kommunale veier og gater per km (kr) )]', '')

na = df_r[df_r.isna().any(axis=1)]
na2 = df_r2[df_r2.isna().any(axis=1)]


# Aggregate old and new kommuner in df_r2, creates consistent time series
df_r = df_r.reset_index().groupby('region').sum()
df_r2 = df_r2.reset_index().groupby('region').sum()

df_r.index.nunique() == df_r2.index.nunique()

# Looks like pattern of missing years is due to databeing present in one column and not there other... I.e. report error
s = pd.DataFrame()
s['diff1'] = df_r['2015'] - df_r2['2015']
s['diff2'] = df_r['2016'] - df_r2['2016']
s.loc[s['diff1']>0].nunique()
s.loc[s['diff2']>0].nunique()

# Concat into one dataframe, remove 2021 ...don't need those
df_r = pd.concat([df_r, df_r2], axis=1, join='outer').reset_index().iloc[:,:15].interpolate(method='linear', axis=0)

# Melt into correct shape
df_r = df_r.melt(
    id_vars=['region']
    , var_name='År'
    , value_name='roadinv_NOK/km')

# Make År int
df_r['År'] = df_r['År'].astype('int')
df_r = df_r.pipe(idx_func)

# Remove duplicate rows
df_r = df_r[~df_r.index.duplicated(keep='first')]


del df_r2

#%%

# Merge into master df
df_master = pd.concat([df_master, df_r], axis = 1, join='outer')

# check null values - only contains data where data is 0 again
na = df_master[df_master.isna().any(axis=1)]

print(df_master.columns[df_master.columns.duplicated(keep=False)])

test = df_master.reset_index()

dup = test[test.duplicated(subset=['region','År'], keep=False)]


#%%

"""
Number of ships & total tonnage of ships
arriving in a given port per year

Operations:
    Aggregated from quarters to years
    Calculcated total amounts columns


"""

# Extract number of ships 
df_ships = data['Havnetilløp_en'].copy()

def ships_tidy_func(df):
    
    df_st = df

    # Tidy data
    df_st['quarter'] = df_st['quarter'].str.replace('K', 'Q')
    df_st['År'] = df_st['quarter'].str[:4].astype('str')
    df_st['tonnage'] = df_st['Gross tonnage'].astype('float') #### Note: something wrong here... 
    
    # Aggregate to yearly data
    df_st = df_st.groupby(['År' ,'port', 'type of vessel'])[['Arrivals of vessels','Gross tonnage']].sum().reset_index()
    
    # Map kommnune to port
    komm = pd.read_csv(r'C:\Users\bened\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\Machine learning data\Associative dicts\havn_kommune_dict.csv', encoding='ANSI')
    d = dict(zip(komm.Havn, komm.Kommune))
    df_st['region'] = df_st['port'].map(d)
    
    # Inspect missing values
    na = df_st[df_st.isna().any(axis=1)]
    
    # Drop nulls, and port
    df_st = df_st.dropna().drop('port', 1).sort_values(['region', 'type of vessel', 'År']).set_index(['region', 'type of vessel', 'År'])
    
    
    # Extract arrivals of vessels
    df_sa = pd.DataFrame(df_st['Arrivals of vessels'].copy()).reset_index()
    df_sa = df_sa.pivot_table(
        values='Arrivals of vessels'
        , index=['region', 'År']
        , columns='type of vessel'
        )
    
    # Pivot ship tonnage and calculate total
    df_st = df_st.drop('Arrivals of vessels', 1).reset_index().pivot_table(
        values='Gross tonnage'
        , index=['region', 'År']
        , columns='type of vessel')
    
    # add prefixes to columns indicating type of variable
    df_st = df_st.add_prefix('tonnage ')
    df_sa = df_sa.add_prefix('n ')
    
    # Make totals columns
    df_st['ships tonnage total'] = df_st[df_st.columns].sum(axis=1)
    df_sa['ships arrivals total'] = df_sa[df_sa.columns].sum(axis=1)
    
    # Concat 
    df_ships = pd.concat([df_sa, df_st], axis = 1)
    
    # Add categorical value for ports
    df_ships['port'] = 1
    
    del df_sa, df_st, komm, d
    
    return df_ships, na

# Apply tidy func
df_ships = ships_tidy_func(df_ships)

# df_ships = df_ships[0][['ships arrivals total', 'ships tonnage total']]

df_ships = df_ships[0]

df_ships.columns

#%%

# merge with master df
df_master = pd.concat([df_master, df_ships], axis = 1, join='outer')

# check null values - only contains data where data is 0 again
na = df_master[df_master.isna().any(axis=1)]

# # Drop null values
df_master[df_ships.columns] = df_master[df_ships.columns].fillna(0)

print(df_master.columns[df_master.columns.duplicated(keep=False)])

test = df_master.reset_index()

dup = test[test.duplicated(subset=['region','År'], keep=False)]



#%%

"""
Pop density
Numbers for old kommuner wasnt avaialble backwards in time
Thus I made population densities for the new municipalities

process:
    new surface area = old surface areas added
    new population = old populations of previous municipalities added together

Note: There's two Aures in the dictionary due to SSB sometimes calling Aure
for Aure kommune in certain datasets. The one producing nan values here is thus dropped
"""
# Extract population and surface area data
df_popd = data['pop_density'].copy().pipe(old_kommuner_to_2020)

df_popd.region.nunique()


# Check why aure kommune produces nan values when divided
aure = df_popd.loc[df_popd['region'] == 'Aure']
aure.groupby(['contents', 'region']).sum()

def komm_agg_func(df):
    
    
    # Subset area and population...
    df_area = df[df['contents'].str.contains('Land area')].dropna().drop('region_old', 1)
    df_pop = df[df['contents'].str.contains('Population')].dropna().drop('region_old', 1)
        
    # Aggregate by new kommuner, yields area and population of new kommuner
    df_area = df_area.groupby(['region', 'contents']).sum().reset_index().drop('contents',1).set_index('region')
    df_pop = df_pop.groupby(['region', 'contents']).sum().reset_index().drop('contents',1).set_index('region')
    
    # Calculate pop density
    df = df_pop / df_area
    
    # df = df.add_prefix('pop/km2 ')
    
    return df

# Apply function
df_popd = df_popd.pipe(komm_agg_func).reset_index().melt(
    id_vars='region'
    , var_name='År'
    , value_name = 'pop/km2')

# Turn år column into int
df_popd['År'] = df_popd['År'].astype('str')

# Set indexes ready for concat
df_popd = df_popd.sort_values(['region', 'År'])

# Analyse nan values
na = df_popd[df_popd.isna().any(axis=1)]

print('Amount of regions present in population density data:')
print(df_popd.region.nunique())

# Drop null values
df_popd = df_popd.dropna()

df_popd = df_popd.loc[df_popd['År'] > '2008'].set_index(['region', 'År'])

df_popd.info()

# df_master['pop/km2'] = df_popd['pop/km2'].copy()

df_master.info()

df_master = pd.concat([df_master, df_popd], axis = 1, join='outer')

# check null values - only contains data where data is 0 again
na = df_master[df_master.isna().any(axis=1)]

print(df_master.columns[df_master.columns.duplicated(keep=False)])

test = df_master.reset_index()

dup = test[test.duplicated(subset=['region','År'], keep=False)]

#%%

"""
Amount of registered cars and attached fuel types

"""

# Import data, 
df_veh = data['biler_registrert'].copy()

# Check nan values
na = df_veh[df_veh.isna().any(axis=1)]

def veh_fix_func(df):
    
    df_cars = df
    
    # Get rid of old kommuner and samisk, then aggregate into new ones.
    df_cars = df_cars.pipe(old_kommuner_to_2020).drop('region_old', 1).groupby(
        ['contents'
         , 'region'
         , 'type of fuel'
         ]).sum().reset_index()

    # Melt into longform dataframe
    df_cars = df_cars.melt(id_vars=['region', 'contents', 'type of fuel']
                , var_name='År'
                , value_name='vehicles_regsistered')
    
    #Cange data type of numerical column
    df_cars['vehicles_regsistered'] = df_cars['vehicles_regsistered'].astype('int')
    
    # subset fuel types
    df_fuel = df_cars.groupby(['region', 'År', 'type of fuel'])['vehicles_regsistered'].sum().reset_index().pivot_table(
        values='vehicles_regsistered',
        index=['region', 'År'],
        columns=['type of fuel']).add_prefix('n vehicle fueltype: ')
    
    # Create electrification variable for fuels
    df_fuel['fuel: electricity %'] = (df_fuel['n vehicle fueltype: Electricity'] / df_fuel[df_fuel.columns].sum(axis=1) ) * 100
    
    # subset vehichle type
    df_veh = df_cars.groupby(['region', 'År', 'contents'])['vehicles_regsistered'].sum().reset_index().pivot_table(
        values='vehicles_regsistered',
        index=['region', 'År'],
        columns=['contents']).add_prefix('n vehicles: ')
    
    df_veh['n vehicles: total'] = df_veh.sum(axis=1)
    
    # Concat into one dataframe
    df_veh = pd.concat([df_fuel, df_veh], axis=1)
    
    return df_veh

# Apply tidy func
df_veh = df_veh.pipe(veh_fix_func)

df_veh = df_veh.reset_index()
df_veh['År'] = df_veh['År'].astype('str')

print(df_veh.info())
df_veh = df_veh.loc[df_veh['År']>'2008']

df_veh.region.nunique()

"""
Could make percentage growths... hwoever, see how your data works with prediction
could always circle back...

"""

# Remove duplicate rows
df_veh = df_veh[~df_veh.index.duplicated(keep='first')].set_index(['region', 'År'])

#%%

df_master = pd.concat([df_master, df_veh], axis = 1, join='outer')

# check null values - only contains data where data is 0 again
na = df_master[df_master.isna().any(axis=1)]

# # Drop null values
# df_master = df_master.dropna()

print(df_master.columns[df_master.columns.duplicated(keep=False)])

test = df_master.reset_index()

dup = test[test.duplicated(subset=['region','År'], keep=False)]


#%%

os.chdir(r'C:/Users/bened/OneDrive - Norwegian University of Life Sciences/Fag/Masteroppgave/Data/Machine learning data/treated data')

#Commit to csv
df_master.to_csv('master_data.csv')
