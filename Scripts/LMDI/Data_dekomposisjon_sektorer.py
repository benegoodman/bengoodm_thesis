# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:06:07 2022

@author: bened
"""

import pandas as pd
import requests
from pyjstat import pyjstat
import numpy as np

import os
import json

import seaborn as sns
import matplotlib.pyplot as plt

#Turn off copy with setting warning
pd.options.mode.chained_assignment = None

# Laptop path
# os.chdir("C:/Users/bengo/OneDrive - Norwegian University of Life Sciences/Fag/Masteroppgave/Data/SSB")

#Desktop path
os.chdir("C:/Users/bened/OneDrive - Norwegian University of Life Sciences/Fag/Masteroppgave/Data/SSB")


#%%

"""
Some sector names in the data from SSB are different from the names in the metadata
which associates sectors and respective NACE kodes. The names in the metadata
were changed to reflect those in the data such that the correct NACE code was applied.

Codes codes changed from/to:
    Utvinning av råolje og naturgass, inkl. tjenester og rørtransport = Bergverksdrift og utvinning av råolje og naturgass, inkl. tjenester
    Kringkasting og film-, video- og musikkproduksjon = 'Film-, video- og musikkproduksjon, kringkasting 
    Informasjons- og teknologitjenester = Tjenester tilknyttet informasjonsteknologi og informasjonstjenester


"""

def emCode_associator_emissions_v2(df):
    
    """
    Associates relevant SN - Miljøregnskap 2012 codes (aka emCode) with sector names
    """
    
    # Import associated names and NACE codes from SSB metadatafile related to emissions
    koder = pd.read_csv(r'./Klassifisering/MiljøregnskapSN_naceBNP_final.csv', sep=';', encoding='ANSI')
    d = dict(zip(koder.sourceName, koder.sourceCode))
    
    # Map NACE codes to uslipp
    df['emCode'] = df['næring'].map(d)
    
    df['emCode'] = df['emCode'].astype('str')
    
    # Extract lvl 2 codes from NACE column
    df['emCode_parent'] = df['emCode'].str[:2]
    
    # Strip dots from emCode_parent
    df['emCode_parent'] = df['emCode_parent'].str.rstrip('.')
    
    return df

def emCode_associator_GDPtoemCode_v2(df):
    
    """
    Associates relevant SN - Miljøregnskap 2012 codes (aka emCode) with sector names
    """
    
    # Import associated names and NACE codes from SSB metadatafile related to emissions
    koder = pd.read_csv(r'./Klassifisering/MiljøregnskapSN_naceBNP_final.csv', sep=';', encoding='ANSI')
    d = dict(zip(koder.targetName, koder.sourceCode))
    
    # Map NACE codes to uslipp
    df['emCode'] = df['næring'].map(d)
    
    df['emCode'] = df['emCode'].astype('str')
    
    # Extract lvl 2 codes from NACE column
    df['emCode_parent'] = df['emCode'].str[:2]
    
    # Strip dots from emCode_parent
    df['emCode_parent'] = df['emCode_parent'].str.rstrip('.')
    
    return df


def emCode_associator_rev_emissions(df):
    
    """
    Associates relevant SN - Miljøregnskap 2012 codes (aka emCode) with sector names of same classification
    Assigns parent code to emCode_parent column for aggregation purposes
    
    Makes it possible to assign correct codes to sectors that are part of other sectors in SN 2012 Miljøregnskap
    (I.e. rørtransport is part of oljeutvinning... )
    """
    
    # Import associated names and NACE codes from SSB metadatafile related to emissions
    koder = pd.read_csv(r'./Klassifisering/MiljøregnskapSN_naceBNP_final.csv', sep=';', encoding='ANSI')
    koder['sorceCode'] = koder['sourceCode'].astype('str')
    d = dict(zip(koder.sourceCode, koder.sourceName))
    
    
    # Map NACE codes to uslipp
    df['næring'] = df['emCode'].map(d)
    
    # Extract lvl 2 codes from NACE column
    df['emCode_parent'] = df['emCode'].str[:2]
    
    # Strip dots from emCode_parent
    df['emCode_parent'] = df['emCode_parent'].str.rstrip('.')
    
    return df


def emCode_name_associator_lvl2_to_lvl1(df):
    
    """
    Creates a new column in dataframe and associates lvl 1 names with lvl 2 names
    
    Requires emCode_associator_emissions function to have already been run
    Requires emCode column to work
    """
    
    
    # Import associated names and emCode codes from SSB metadatafile related to emissions
    koder = pd.read_csv(r'./Klassifisering/NACE-koder_txt.txt', sep=';', encoding='ANSI')
    d = dict(zip(koder.code, koder.name))

    # Map emCode codes to uslipp
    df['emCode_lvl1'] = df['emCode'].map(d)
    
    return df


def emCode_associator_rev_gdp(df):
    
    """
    Associates relevant emCode and em-sector name
    """
    
    # Import associated names and NACE codes from SSB metadatafile related to emissions
    koder = pd.read_csv(r'./Klassifisering/MiljøregnskapSN_naceBNP_final.csv', sep=';', encoding='ANSI')
    koder['sourceCode'] = koder['sourceCode'].astype('str')
    d = dict(zip(koder.sourceCode, koder.sourceName))
    
    # Map NACE codes to uslipp
    df['næring'] = df['emCode'].map(d)
    
    # Extract lvl 2 codes from NACE column
    df['emCode_parent'] = df['emCode'].str[:2]
    
    # Strip dots from emCode_parent
    df['emCode_parent'] = df['emCode_parent'].str.rstrip('.')
    
    return df

def emCode_child_to_parent(df):
    
    """
    Input:
        dataframe
    
    Process:
        Turns emcode colunmn into string
        Saves string to sublists
        Extracts first item of each sublist and turns it into column
        Adds in 10 where child code is 10
        
    Output:
        dataframe with parent code column
    
    """
    
    df['emCode'] = df['emCode'].astype('str')
    lst = df['emCode'].str.split(pat='.')
    df['emCode_parent'] = [item[0] for item in lst]
    df['emCode_parent'].loc[df['emCode'] == 10] = 10
    
    return df


def sector_removal_func(df):
    
    """Removes sectors not encountered in GDP measures, i.e. Alle næringer, Alle næringer og husholdninger"""
    
    df = df[df['næring'].str.contains('Alle næringer|Husholdninger') == False]

    
    return df

def emCode_parent_to_sector(df):
    
    """
    Input: df
    
    Operation:
        - makes emCode_parent type int
        - reads emCode_parent column
        - applies aggregate sector name to emCode in new column 'næring'
        
    Returns: 
        df with aggregate sector names
    """
    
    df['emCode_parent'] = df['emCode_parent'].astype('int')

    # Associate parent codes and sector columns
    koder = pd.read_csv(r'./Klassifisering/emcodeparent_to_næring_final.csv', sep=',', encoding='ANSI')
    d = dict(zip(koder.emCode_parent, koder.næring_11))
    
    df['næring'] = df['emCode_parent'].map(d)
    
    return df

#%%

# Create query to df function
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

# """
# Collection of national emission data from SSB - detailed sectors

# Metadata information

# Amount of sectors: 64

# Range: 1990 - 2020

# Unit: Utslipp til luft (1 000 tonn CO2-ekvivalenter)

# Source: SSB table 09288

# URL: https://www.ssb.no/statbank/table/09288/

# Footnotes from SSB:
    
#     På grunn av avrunding vil totaler kunne avvike fra summen av undergrupper. 
#     Utslipp fra utenriks sjøfart og luftfart er inkludert 
#     Utvinning av råolje og naturgass, inkludert tjenester og rørtransport samt 
#     Utenriks sjøfart og aggregerte størrelser er rettet for perioden 1990 til 2020, 30. juni 2021. 
#     . = Ikke mulig å oppgi tall. Tall finnes ikke på dette tidspunktet fordi kategorien ikke var i bruk da tallene ble samlet inn.

#     Rørtransport
#     'N.08.01.00 Rørtransport' er inkludert i 'N.02.02.00 Utvinning av råolje og naturgass' 

# """

# # Set URLs for data and metadata
# url = 'https://data.ssb.no/api/v0/no/table/09288/'
# urlmeta = 'https://data.ssb.no/api/v0/no/console/meta/table/09288/'
# file = './JSON queries/utslippsdata_table_09288.json'

# # Collect emission data
# utslipp_detailed = query_to_df(url, urlmeta, file)

# # Apply NACE-associator function to utslipp
# utslipp_detailed = utslipp_detailed.pipe(emCode_associator_emissions)

# # Extract aggregate emissions
# df_tot_utslipp = utslipp_detailed.loc[utslipp_detailed['næring'] =='Alle næringer og husholdninger']


#%%

"""
Collection of national emission data from SSB - aggregated sectors

Metadata information

Amount of sectors: 34

Range: 1990 - 2020

Unit: Utslipp til luft (1 000 tonn CO2-ekvivalenter)

Source: SSB table 09288

URL: https://www.ssb.no/statbank/table/09288/

Footnotes from SSB:
    
    På grunn av avrunding vil totaler kunne avvike fra summen av undergrupper. 
    Utslipp fra utenriks sjøfart og luftfart er inkludert 
    Utvinning av råolje og naturgass, inkludert tjenester og rørtransport samt 
    Utenriks sjøfart og aggregerte størrelser er rettet for perioden 1990 til 2020, 30. juni 2021. 
    . = Ikke mulig å oppgi tall. Tall finnes ikke på dette tidspunktet fordi kategorien ikke var i bruk da tallene ble samlet inn.

    Rørtransport
    'N.08.01.00 Rørtransport' er inkludert i 'N.02.02.00 Utvinning av råolje og naturgass'

Note: Will be using 34 sector breakdown of emissions as energy data isnt avaiable
on a finer level until after 2010

Note: Rørtransport, tjenester tilknyttet utvinning av olje og gass and utvinning av olje og gass
are all fitted together into one category within the SSB data. The translation key on SSB website puts these in different sectors.
For the purposes of this data, they have been put together and given the emCode 2.04

"""

# Set URLs for data and metadata
url = 'https://data.ssb.no/api/v0/en/table/09288/'
urlmeta = 'https://data.ssb.no/api/v0/no/console/meta/table/09288/'
file = './JSON queries/ssbapi_table_09288_utslipp_34.json'

# Collect emission data
df_utslipp_31 = query_to_df(url, urlmeta, file).set_index('år')

# næring_uts1 = pd.DataFrame(df_utslipp_31['næring'].unique()).sort_values(0).rename(columns={0 : 'uts_sektor'})

# Apply emCode-associator function to utslipp
df_utslipp_31 = df_utslipp_31.pipe(emCode_associator_emissions_v2)

# Extract nulls
null = df_utslipp_31[df_utslipp_31.isna().any(axis=1)]

# Print nulls
print('Rows containing null values:')
print(null)

# Check amount of sectors
print('Number of sectors related to emissions before eliminating alle næringer & husholdninge, and merging petro related sectors:', df_utslipp_31['næring'].nunique())


"""
Husholdinger was extracted and needs to be treated as a special case as 
husholdninger arent counted in GDP

"""

# Extract husholdninger as separate df - for separate decomposition
df_hush_utslipp = df_utslipp_31.loc[df_utslipp_31['næring'] =='Husholdninger']

# Write household emissions to  to csv
df_hush_utslipp.to_csv('../Ferdige_data/utslipp_hush.csv')


# Turn emcode parent to int
df_utslipp_31['emCode_parent'] = df_utslipp_31['emCode_parent'].astype(int)

# Eliminate rows where parent code = 0, i.e. alle næringer + husholdninger
df_utslipp_31 = df_utslipp_31[df_utslipp_31.emCode_parent != 0]

# Eliminate rørtransport, which is already integrated in sector 2.04
df_utslipp_31 = df_utslipp_31[df_utslipp_31["næring"].str.contains("Rørtransport")==False]

# Check amount of sectors
print('Number of sectors related to emissions post elimination:', df_utslipp_31['næring'].nunique())


#%%

"""
Creation of aggregate sector emissions (10 sectors), write to csv

"""

# Create macro sector df
df_utslipp_10 = df_utslipp_31.groupby(['emCode_parent', 'år', 'statistikkvariabel']).sum('value').reset_index()

# Associate parent codes and sector columns
koder = pd.read_csv(r'./Klassifisering/emcodeparent_to_næring_final.csv', sep=',', encoding='ANSI')
d = dict(zip(koder.emCode_parent, koder.næring_11))

# Map sector names to parent codes
df_utslipp_10['næring'] = df_utslipp_10['emCode_parent'].map(d)

# Write to csv
df_utslipp_10.to_csv('../Ferdige_data/utslipp_sektorer_10.csv')


#%%

""" 
Import of GDP data

Amount of sectors: 66

Range: 1990 - 2019

Variable: Bruttoprodukt i faste priser

Unit: millNOK, 2015 prices

Source: SSB table 09170

Footnotes from SSB
 
De enkelte tallene i faste priser summerer seg ikke opp til del- og totalsummene på grunn av kjedingsavvik. 
 Tall fra og med 2020 er foreløpige. 

Reason for using 66 sectors: 34 sector format lumps sectors together compared to emissions sectors.
Thus 66 sectors allow for correct aggregation provided correct aggregation key is made

...luckily i made one :)
    
    
Q to self: implications of kjedingsavvik? should i do the analysis with nominal numbers? If I use fixed prices and the individual elements do not add up I introduce a residual
the decomposition - wont this be larger than the effect of using nominal numbers?
    
"""

# Set URLs for data and metadata
url = 'https://data.ssb.no/api/v0/no/table/09170/'
urlmeta = 'https://data.ssb.no/api/v0/no/console/meta/table/09170/'
file = './JSON queries/ssbapi_table_09170_A64.json'

# Import utslipp by aggregate sector, associate NACE
df_bnp = query_to_df(url, urlmeta, file)

# Checks for nulls, stores to separate dataframe
nulls_bnp = df_bnp.loc[df_bnp['value'].isna()]

# Check amount of sectors
print('Number of sectors related to GDP pre aggregation:', df_bnp['næring'].nunique())

# Apply emCodes to BNP
df_bnp = df_bnp.pipe(emCode_associator_GDPtoemCode_v2)

# Collect nulls
nulls = df_bnp.loc[df_bnp['emCode'].str.contains('nan')]

# Drop nulls from main dataframe
df_bnp = df_bnp.dropna()

# Aggregate df_bnp by emCode and year
df_bnp = df_bnp.groupby(['emCode', 'år']).sum('value').reset_index().pipe(emCode_associator_rev_gdp).set_index('år')

# Check amount of sectors
print('Number of sectors related to GDP post aggregation:', df_bnp['næring'].nunique())

# Write to csv



#%%

"""
Collection of national energy use data from SSB

Metadata information

Sector standard: SN - Miljøregnskap

Amount of sectors: 34

Range: 2010 - 2020

Unit: GWh, energibruk i alt

Justfication for energibruk i alt:
Energibruk i alt omfatter summen av alle typer forbruk. Det skilles i energiregnskapet (ER) ikke mellom ulike formål 
(omvandling, transport, råstoff og annet sluttforbruk) slik som i EB.

I.e. energibruk i alt yields total fossil energy dependency for each sector - even though that sector doesn't
use those fossil fuels to produce energy. After all, fossil energy use is fossil energy use even though
it is not used to power something else.

Avklaring som må gjøres: Forteller det en bedre historie å gjøre om enheten fra GWh til tonn olje-ekvivalenter? GWh impiserer
tross alt bruk av energi og ikke bruk av petro-produkter

Source: SSB table 11558

URL: https://www.ssb.no/statbank/table/11558/

Footnotes from SSB:
    
     Produksjon: Avfall produseres av hele økonomien og plasseres ikke etter næring (men noen unntak i industri). 
     
     Forbruk i næringer og husholdninger til energiformål: Omfatter alle typer forbruk av energiprodukter. 
     
     Dvs. energiprodukter benyttet til omvandling, transport, råstoff og annet sluttforbruk. 
     
     Energiprodukter benyttet som råstoff i industrien, energiprodukter benyttet til omvandling i 
     oljeraffinerier og forbruk av smøremidler og bitumen er ekskludert. 
     
     EP05 Biobrensler: Inkluderer bioandelene for bensin (EP0465), autodiesel (EP0467112) og jetparafin (EP04661). 
     EP052 Flytende biobrensler: Inkluderer bioandelene for bensin (EP0465), autodiesel (EP0467112) og jetparafin (EP04661). 
     EP0469 Oljeprodukter ikke nevnt andre steder: Inkluderer raffineriråstoff (EP043), 
     tilsetningsstoffer i raffineri (EP044), 
     andre hydrokarboner og hydrogener (EP045) og raffinerigass (EP0461). 
    
    Tall for 1990-2019 er revidert 27.10.2020. 
    
    Som følge av endring i nivåtallene fra statistikken Sal av petroleumsprodukt, 
    vil tallene bli rettet ved neste publisering 21. juni 2021.
    
    : = Vises ikke av konfidensialitetshensyn. Tall publiseres ikke for å unngå å identifisere personer eller virksomheter. 
    . = Ikke mulig å oppgi tall. Tall finnes ikke på dette tidspunktet fordi kategorien ikke var i bruk da tallene ble samlet inn.
    - = Null
    

Note regarding sector breakdown in energy accounts:
    Utvinning av olje og gass grupperes sammen med rørtransport i energiregnskapet. Dvs. det som er emCode 2.02, 2.03 og 8.01 i måten BNP
    og utslipp er gruppert på. Jeg har laget en ny emCode, emCode 2.04 som omfatter alle tre. Dette gjør at olje og gass må
    grupperes sammen i utslipps- og bnp- regnskap.

Treatment of missing values in energy usage:
    
    - It is impossible to assume anything about the energy usage where SSB withholds data. Cases must be imputed.
    - Zero usage (-) is imputed to 0
    - Missing data depends on the nature of the missing data. It might be that missing data makes it
      necessary to drop years from the aalysis

"""

"""
Note to self: Identifiser næringer som må aggregeres slik at vi får 31 næringer innenfor energi
"""

# # Set URLs for data and metadata
# url = 'https://data.ssb.no/api/v0/no/table/11558/'
# urlmeta = 'https://data.ssb.no/api/v0/no/console/meta/table/11558/'
# file = './JSON queries/ssbapi_table_11558_energinasjonalt.json'

# Collect emission data
# df_energy = query_to_df(url, urlmeta, file).pipe(emCode_associator_emissions_v2)

# Import data
df_energy = pd.read_csv(r'./Rådata/energibalanse_gwh.csv', sep=';', header=1, encoding='ANSI')

# Apply emCodes to energy data
df_energy = df_energy.pipe(emCode_associator_emissions_v2)

#Check for witheld data
df_energy_witheld = df_energy[df_energy['Mengde (GWh)'] == ':'].groupby('år').count()
df_energy_missing = df_energy[df_energy['Mengde (GWh)'] == '.'].groupby('år').count()
df_energy_zero = df_energy[df_energy['Mengde (GWh)'] == '-'].groupby('år').count()

# Print amounts of missing data
print('Amount of withheld datapoints per year:')
print(df_energy_witheld['Mengde (GWh)'])

print('')
print('Total amount of withheld datapoints:', df_energy_witheld['Mengde (GWh)'].sum())
print('')
print('Total amount of missing datapoints:', df_energy_missing['Mengde (GWh)'].sum())
print('')
print('Total amount of zero datapoints:', df_energy_zero['Mengde (GWh)'].sum())

# Turn zero datapoints into zero
df_energy = df_energy.replace('-', '0')

"""
Conclusion: No withheld data :)

"""



#%%

# Check for patterns in missing data

# Extract subset where data is missing
df_energy_missing = df_energy[df_energy['Mengde (GWh)'] == '.']

# Group missing data by næring
print(df_energy_missing.groupby('næring').count())


sns.set(rc = {'figure.figsize':(16, 8)}, style="whitegrid", palette='tab10')

# Group missing data by energy product - plot missing data. Yields amounts of missing data across energyproducts
mis_enprod = df_energy_missing.groupby('energiproduktregnskap').count().reset_index()
plot = sns.barplot(y='Mengde (GWh)', x = 'energiproduktregnskap', data = mis_enprod, hue='energiproduktregnskap')
plot.set(xticklabels=[])
plot.set_ylabel("Missing obs", fontsize = 12)
plot.set_xlabel("Energy product", fontsize = 12)
plt.show()

"""
Looks like most of the missing data is within 6 categories:
    - Andre energiprodukter (we can assume energy usage from this product to be very, very low) *check if thats the case - if it is, set to zero
    - Avfall
    - Kull og kullprodukter
    - Biobrensler
    - Naturgass

"""


#%%

"""
Visualisation of missing data by energy product

Note: Add sector breakdown as well...


"""


def dot_to_nan_func(df_energy):
    
    # Replace . with nan in energy column ...in a roundabout way
    df_energy = df_energy.replace('.', '999999999')
    df_energy['Mengde (GWh)'] = df_energy['Mengde (GWh)'].astype('int')
    df_energy['Mengde (GWh)'] = df_energy['Mengde (GWh)'].astype('Int64')
    df_energy = df_energy.replace(999999999, np.nan)
    
    return df_energy

# apply replacer function to df
df_energy = dot_to_nan_func(df_energy)

def heatmapper_func_energy(df_energy):
    
    """
    Yields heatmap showing where missing values occur
    34 = no missing values
    High number = few missing values
    Low number = many missing values
    
    Returns df showing value count in Mengde (GWh) column
    """
    
    # Extract subset for heatmat plot
    df = df_energy
    # df['Mengde (GWh)'] = df['Mengde (GWh)'].astype('int')
    
    # Create aggregated df that counts amounts of entries. If a cell says 34 it means no missing values
    df = df[['energiproduktregnskap', 'Mengde (GWh)', 'år']].groupby(['energiproduktregnskap', 'år']).count().reset_index()
    
    # Pivot df for right shape for heatmap
    df = df.pivot(columns='år', values = 'Mengde (GWh)', index='energiproduktregnskap')
    
    sns.set(palette='viridis')
    
    # Make heatmap of data
    sns.heatmap(df, cbar=True, annot=True)
    plt.show()
    
    return df

# Apply heatmap function to see missing data by energy source
df_energyprod = heatmapper_func_energy(df_energy)

def heatmapper_func_energy_v2(df_energy):
    
    """
    Yields heatmap showing where missing values occur
    9 = no missing values
    High number = few missing values
    Low number = many missing values
    
    Returns df showing value count in Mengde (GWh) column
    """
    df = df_energy

    df = df[['næring', 'Mengde (GWh)', 'år']].groupby(['næring', 'år']).count().reset_index()

    # Pivot df for right shape for heatmap
    df = df.pivot(columns='år', values = 'Mengde (GWh)', index='næring')
                  
    # Make heatmap of data
    sns.heatmap(df, cbar=True, annot=True) # Consider different colour scheme ...blue?
    plt.show()
    
    return df

df_energysec = heatmapper_func_energy_v2(df_energy)


def heatmapper_func_energy_næring(df_energy):
    
    """
    Yields heatmap showing where missing values occur
    9 = no missing values
    High number = few missing values
    Low number = many missing values
    
    Returns df showing value count in Mengde (GWh) column
    """
    df = df_energy[df_energy['energiproduktregnskap'].str.contains('Naturgass')]

    df = df[['næring', 'Mengde (GWh)', 'år']].groupby(['næring', 'år']).count().reset_index()

    # Pivot df for right shape for heatmap
    df = df.pivot(columns='år', values = 'Mengde (GWh)', index='næring')
                  
    # Make heatmap of data
    sns.heatmap(df, cbar=True, annot=True) # Consider different colour scheme ...blue?
    plt.show()
    
    return df

df_energy_secprod = heatmapper_func_energy_næring(df_energy)

"""
Conclusion:
    What the F is going on. Why no missing data between 2010 - 2016?
    Why gradually complete data for naturgass?
    Why more incomplete data after 2016?
    
    Have i aggregated the data correctly? Note: Check for errors a domani
    
Thought:
    Naturgass is really the only energy source where missing data matters as that's the only
    energy source that contributes more than 1% towards the energy consumed
    
    Could do a check to see if data from each sector and year adds up to total.
    
    Sectors in which natural gas data is variable
    
"""

#%%

"""
Check for importance of energy sectors - helps me decide which
sectors are important with regards to missing data

"""

# Check how much each factor is of the total energy known mix
sub = df_energy.loc[df_energy['emCode'] == '0.0']
sub = sub.replace(['-', '.', np.nan], 0)
sub['Mengde (GWh)'] = sub['Mengde (GWh)'].astype('int')
sub = sub.groupby(['energiproduktregnskap']).sum('Mengde (GWh)').reset_index()
sub = sub[sub['energiproduktregnskap'].str.contains('Alle')==False]
# plot = sns.lineplot(y='Mengde (GWh)', x = 'år', data = sub, hue='energiproduktregnskap',linewidth=2)

class pie:
    
    """
    Flexible pie input charter
    
    Input: Long format df with values and næring in columns
    
    Output: 
    """
    
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set(rc = {'figure.figsize':(16, 12)}, style="whitegrid", palette='tab10')
    
    def __init__(self, df, title, titlesize):
        self.df = df
        self.title = title
        self.titlesize = titlesize
        
    
    def pie_charter(self):
        labels = list(self.df['energiproduktregnskap'])
        
        # Create piechart
        color = sns.color_palette('Blues_r')
        sns.set_theme(style='whitegrid')
        plt.pie(self.df['Mengde (GWh)'], colors = color, autopct='%.0f%%')
        plt.suptitle(self.title, size = self.titlesize)
        plt.legend(labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.figure(figsize=(16, 12))
        # plt.tight_layout()
        plt.show()

plot = pie(sub, 'Cumulative energy usage per energy source', 30).pie_charter()

# Bar plot of distribution of GWh from different energy sources
plot = sns.barplot(y='Mengde (GWh)', x = 'energiproduktregnskap', data = sub, hue='energiproduktregnskap')
plot.set(xticklabels=[])
plot.set_ylabel =('GWh')
plt.tight_layout()
plt.show()

# Get percentages of power usage
sub['%'] = (sub['Mengde (GWh)'] / sub['Mengde (GWh)'].sum()) * 100
sub.drop('år', axis=1, inplace=True)

print('Cumulative consumption per energy product 1990 - 2020')
print(sub)

# Check for sectoral patterns of missing data

"""
Note to self: Fix graph of which sectors 

"""


ngas = df_energy_missing[df_energy_missing['energiproduktregnskap'].str.contains('Naturgass')].groupby('næring').count().reset_index()
ngas['missing obs'] = ngas['år']
plot = sns.barplot(y='missing obs', x = 'næring', data = ngas, hue='næring')
plot.set(xticklabels=[])
plt.tight_layout()
plot.set_ylabel = ('Missing obs (n)')
plt.show()

# # Make heatmap of missing data
# ngas = df_energy_missing[df_energy_missing['energiproduktregnskap'].str.contains('Naturgass')].groupby(['næring', 'år']).count()
# sns.heatmap(ngas['Mengde (GWh)'], cbar=True)
# plt.show()



"""
Conclusion:
    - Andre energiprodukter conributes near zero to energy consumed - can be assumed 0 where data is missing
    - Avfall is near 1% of total energymix, can be safely assumed to be 0 where data is missing
    - Biobrensel is 2% of energy consumed since 1990, can be safely imputed to 0
    - Kull og kullprodukter constitutes 2% of energy consumed since 1990, can be safely imputed to 0
    - Naturgass constitutes 9% of total energy consumed since 1990. It is problematic that this variable has missing.
      This category must be investigated further. If the missing data is systemic I will either have to drop years from the analysis
      or I will have to assume that the usage is 0 if it turns out that it is within sectors where methane is unlikely to be used as an energy source

"""


#%%

"""
Cleanup of energydata

Operations
- Imputed 0 where energy data was missing (bold assumption... will need to be revisited)
- Extraction of missing data
- Extracted fossil fuels as separate dataframe
    - Identify total usage of fossil fuels (GWh)
    - Identify petro usage per sector
    - Avfall has a (very roughly) estimated content of fossil carbon at 20% 
        Source: https://www.ssb.no/natur-og-miljo/artikler-og-publikasjoner/beregning-av-co2-faktor-for-utslipp-fra-fossil-del-av-avfall-brent-i-forbrenningsanlegg
        page 12 
    - Thus: Fossil fuels include 20% of the energy from avfall. Missing datapoints has assumed usage of 0.
- Extracted aggregated total energy usage 
- Extracted total energy usage per sector per year
- Aggregated rørtransport to 
Will use fossil fuels per sector year and total energy usage per sector per year for decomposition


"""

# Assume zero usage where data is missing - cannot assume anything else
# Rename column, impute energy usage to 0
df_energy = df_energy.rename(columns={'Mengde (GWh)' : 'value'})
df_energy['value'] = df_energy['value'].fillna(0)

# Apply parent codes to df_energy
df_energy = df_energy.pipe(emCode_child_to_parent)

# Check amount of sectors
print('Number of sectors related to energy post aggregation:', df_energy['næring'].nunique())


def energy_fossil_extractor (df_energy):
    
    """
    Input: df_energy
    
    Operations:
        - Extract subset of fossil fuels
        - Multiplies energy usage from avfall by 0.2, this is estimated fossil component of avfall
            -    Source: https://www.ssb.no/natur-og-miljo/artikler-og-publikasjoner/beregning-av-co2-faktor-for-utslipp-fra-fossil-del-av-avfall-brent-i-forbrenningsanlegg
                page 12 
            - Thus: Fossil fuels include 20% of the energy from avfall. 
            - Missing datapoints has assumed usage of 0
        - Appends fossil portion of energy produced from avfall to fossil energy dataframe
        - Removes total energy from avfall per sector
        
    Output: df with amount of GWhs stemming from fossils accross all sectors
    
    """
    
    # Extract fossil energy sources
    df_energy_foss = df_energy[df_energy['energiproduktregnskap'].str.contains('Kull|Naturgass|Olje|Avfall')]
    
    # Turns value column into int
    df_energy_foss['value'] = df_energy_foss['value'].astype('Int64')
    
    # Pull out avfall as subset
    avfall = df_energy_foss[df_energy_foss['energiproduktregnskap'].str.contains('Avfall')]
    
    # Multiply energy from avfall by 0.2 -> to account for fossil degree in waste
    avfall['value'] = avfall.value.fillna(0) * 0.2
    
    # Rename avfall to name indicating fossil energy fraction
    avfall['energiproduktregnskap'] = 'Avfall fossil fraksjon (ktonn)'
    
    # Drop total energy use from avfall from fossil dataframe, append fossil fraction
    df_energy_foss = df_energy_foss[df_energy_foss['energiproduktregnskap'].str.contains('Avfall')==False]
    df_energy_foss = pd.concat([df_energy_foss, avfall])
    
    # Aggregate fossil usage per sector
    df_energy_foss = df_energy_foss.groupby(['næring', 'år']).sum('value').reset_index()
    
    # rename value column
    df_energy = df_energy.rename(columns = {'value' : 'GWh_Fossil'})
    
    # Calculate total fossil energy usage per sector
    df_energy_foss = df_energy.groupby(['energiregnskapspost', 'emCode', 'emCode_parent', 'næring', 'år']).sum('GWh_Fossil').reset_index()
    df_energy_foss['energiproduktregnskap'] = 'Fossil energibruk'

    return df_energy_foss

# Apply fossil energy subsetter function to df_energy
df_energy_foss = energy_fossil_extractor(df_energy)

# Extract total energy usage from sectors aside from husholdninger
df_energy_foss_tot = df_energy_foss[df_energy_foss['emCode'].str.contains('0.1')]

# Extract aggregate sectors from fossil energy used
df_energy_foss_10 = df_energy_foss.pipe(
    emCode_child_to_parent).groupby(['emCode_parent', 'år']).sum('GWh_Fossil').reset_index().pipe(emCode_parent_to_sector)


# Create dataframe with fossil energy usage per sector, aggregate by emCode to assure correct aggregation
df_energy_foss_31 = df_energy_foss.pipe(sector_removal_func).groupby(
    ['emCode', 'år']
    ).sum(
        'GWh_Fossil'
        ).reset_index()

# Apply name associated on em Codes in dataframe
df_energy_foss_31 = df_energy_foss_31.pipe(emCode_associator_rev_gdp)

# Map names to emCodes
df_energy_foss_31['næring'] = df_energy_foss_31['emCode'].map(d)

# Extract fossil and total energy usage in households
df_energy_hush_tot = df_energy[(df_energy['næring'].str.contains('Husholdninger')) & (df_energy['energiproduktregnskap'].str.contains('Alle energiprodukter'))]
df_energy_hush_foss = df_energy_foss[df_energy['næring'].str.contains('Husholdninger')]

# Isolate data where all energy sources are included - 
df_energy_31 = df_energy[df_energy['energiproduktregnskap'] == 'Alle energiprodukter'].pipe(sector_removal_func)

# Dropping rørtrnsport from df, values are already incorporated into emCode 2.04 sector by default
df_energy_31 = df_energy_31[df_energy_31['næring'].str.contains('Rørtransport')==False]

df_energy_31 = df_energy_31.pipe(emCode_associator_rev_gdp)

# Check amount of sectors
print('Number of sectors related to total energy prior to treatment:', df_energy_31['næring'].nunique())

# Check sectors present
df_energy_31['næring'].unique()

#%%

# Check for sector equality within energy, emissions and gdp
l1 = list(df_energy_31['næring'].unique())
l2 = list(df_bnp['næring'].unique())
l3 = list(df_utslipp_31['næring'].unique())

# Definition of function. Lists should be equal in lenght and the same in sorted order. If not, returns False
def checkEqual(l1, l2, l3):
    
    """
    Input: 3 invidual lists
    
    Makes list of unique values
    
    Asserts that lenght and sorted order of lists are the same
    
    Returns: True/False
    """
    return len(l1) == len(l2) == len(l3) and sorted(l1) == sorted(l2) == sorted(l3)

# Print results
print('Sectors within energy, gdp and emissions data are the same:', checkEqual(l1, l2, l3))

#%%



# Prepare enegy sector df for aggregtion
df_energy_10 = df_energy_31.groupby(['emCode_parent', 'år']).sum('value').reset_index()

# Create dataframes with aggregate sectors based on emCode_parent aggregation 
df_energy_10 = df_energy_10.pipe(emCode_parent_to_sector).groupby(['næring', 'år', 'emCode_parent']).sum('value').reset_index()
df_bnp_10 = df_bnp.pipe(emCode_child_to_parent).pipe(emCode_parent_to_sector).groupby(['næring', 'år', 'emCode_parent']).sum('value').reset_index()


#%%

# Create lists of unique sectors in næring, bnp and utslipp aggregates
l1, l2, l3 = sorted(list(df_energy_10['næring'].unique())), sorted(list(df_utslipp_10['næring'].unique())), sorted(list(df_bnp_10['næring'].unique()))

test = pd.DataFrame(zip(l1, l2, l3))

# Check for sector consistency
print('Sectors within energy, gdp and emissions data are the same:', checkEqual(l1, l2, l3))

#%%

l1 = df_energy_foss_31['næring'].unique()
l2 = df_utslipp_31['næring'].unique()

set(l1).difference(l2)

#%%

# Committ imported, aggregated data to CSV files

# All energy, 31 sectors and 11
df_energy_10.to_csv('../Ferdige_data/energi_sektorer_10.csv', index = False)
df_energy_31.to_csv('../Ferdige_data/energi_sektorer_31.csv', index = False)

# Fossil energy, 31 sectors and 11
df_energy_foss_10.to_csv('../Ferdige_data/energi_foss_10.csv', index = False)
df_energy_foss_31.to_csv('../Ferdige_data/energi_foss_31.csv', index = False)

# Household energy, fossil energy use and emissions
df_energy_hush_tot.to_csv(r'../Ferdige_data/energy_hush_tot.csv', index = False)
df_energy_hush_foss.to_csv(r'../Ferdige_data/energy_hush_foss.csv', index = False)
df_hush_utslipp.to_csv('../Ferdige_data/utslipp_hush.csv', index = False)

# GDP, 31 sectors and 11
df_bnp_10.to_csv('../Ferdige_data/bnp_sektorer_10.csv', index = False)
df_bnp.to_csv('../Ferdige_data/bnp_sektorer_31.csv')

# Emissions, 31 sectors and 11
df_utslipp_10.to_csv('../Ferdige_data/utslipp_sektorer_10.csv', index = False)
df_utslipp_31.to_csv('../Ferdige_data/utslipp_sektorer_31.csv')



