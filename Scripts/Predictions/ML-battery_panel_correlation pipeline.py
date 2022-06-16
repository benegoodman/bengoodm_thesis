# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:41:12 2022

@author: bened
"""

import os
import pandas as pd
import numpy as np

import seaborn as sns

sns.set()

os.chdir(r'C:/Users/bengo/OneDrive - Norwegian University of Life Sciences/Fag/Masteroppgave/Data/Machine learning data/ML_ready')

#%%

em = pd.read_csv('emissions_master.csv')

df1 = pd.read_csv('master_data_t1.csv')
df2 = pd.read_csv('master_data_t2.csv')
df3 = pd.read_csv('master_data_t3.csv')
df4 = pd.read_csv('master_data_t4.csv')
df5 = pd.read_csv('master_data_t5.csv')

em = em[['År', 'region', 'l_tCO2e']].rename(columns={'l_tCO2e' : 'past emissions l_tCO2e'})
em['past emissions l_tCO2e'] = np.exp(em['past emissions l_tCO2e']) 

"""
Assignment of observations to training and validation groups

Currently not used - we'll see if it comes in handy
"""
def year_remover(df, cutoff):
    
    df = df.loc[df['År'] <= cutoff]
    
    return df

# Past emissions
em5 = year_remover(em, 2014).set_index(['region', 'År']).rename(columns={'l_tCO2e' : 'past emissions ltCO2e'})

# Remove years 
df1 = year_remover(df1, 2018)
df2 = year_remover(df2, 2017)
df3 = year_remover(df3, 2016)
df4 = year_remover(df4, 2015)
df5 = year_remover(df5, 2014).set_index(['region', 'År'])

# # Add inn past emissions
df5= pd.concat([df5, em5], axis=1).dropna(axis=0)



#%%

# Apply min-max normalisation formula
def min_max_norm(df):
    Xn = df
    Xn = (Xn-Xn.min())/(Xn.max()-Xn.min())
    return Xn

# Standardise data
def standardise_func(Xn):
    Xn = (Xn-Xn.mean())/Xn.std()
    return Xn

# apply formula to X
feats = df5.iloc[:,1:].pipe(min_max_norm).dropna(axis=1)

df5 = pd.DataFrame(df5.iloc[:,0]).join(feats)


#%%

# Clustering
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

def optimum_clusters(df):
    
    X = df
    
    # Finding optimal levels of clusters
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Plotting of distances
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    
    # plt.ylim(0, 60000)
    return plt.plot(distances)
    
df5.pipe(optimum_clusters)


def db_scan(X):
    
    X = X
    
    # hyper parameters for dbscan
    epsilon = 0.024
    min_samples = 340
    
    # Do DBscan for clusters
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    return n_clusters_, n_noise_, labels


db_res = df5.pipe(db_scan)

#%%

def k_means_optimum(X):
    
    Xn = X
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    
    sil = []
    kmax = 50
    
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k, n_init=10).fit(Xn)
        labels = kmeans.labels_
        sil.append(silhouette_score(Xn, labels, metric = 'euclidean')) # Calculates average silhouette score
    
    sns.set(rc={"figure.figsize":(6, 5)})
    plt.plot(list(range(0, len(sil))), sil, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Silhouette score') 
    plt.title('Silhouette analysis for opt. k-clusters - panel data', size=16)
    plt.show()
    
    return labels, sil
    
k_means = k_means_optimum(df5)

"""
Conclusion: 3 clusters
"""

#%%

from sklearn.cluster import KMeans

# # Run clustering again with n cluster, add to dataset
# kmeans = KMeans(n_clusters = 1, n_init=10).fit(df5)
# labels = kmeans.labels_



#%%

"""
Panel-test-train module

"""

def test_train(df):
    
    df = df.reset_index()
    
    # Get unique regions, assign to new df
    ids = df.drop_duplicates(subset='region')
    ids = ids[['region']]
    
    # Create random column
    np.random.seed(42)
    ids['rand'] = (np.random.randint(0, 10000, ids.shape[0]))/10000
    ids = ids[['region', 'rand']]
    
    # Assign groupings
    ids['group'] = np.where(ids['rand'] <= 0.75, 'Train', 'Val')
    
    # Sort regions in same order
    df = df.sort_values('region', ascending=True)
    ids = ids.sort_values('region', ascending=True)
    
    # Append grouping to df 
    df = df.merge(ids, on=['region'], how='inner')
    
    print()
    print('Number of obs in groups')
    print(df.groupby('group')['rand'].count())

    return df

# Apply to training set
df5 = test_train(df5).set_index(['region', 'År'])

idx = df5.index

#%%

def variable_setter(df, y):
    
    # Select target and X variables - training set
    y_train = df[df['group'] == 'Train'][y]
    X_train = df[df['group'] == 'Train'].iloc[:,3:57]
    
    # Select target and X variables - validation set
    y_test = df[df['group'] == 'Val'][y]
    X_test = df[df['group'] == 'Val'].iloc[:,3:57]
    
    # Drop group variable from original dataset
    df = df.drop(['group', 'rand'], 1)
    
    # Drop group variable
    # X_train, X_test = X_train.drop('rand',1), X_test.drop('rand', 1)
    
    return  X_train, X_test, y_train, y_test, df

data = df5.pipe(variable_setter, 'tCO2e_t5')

# unpack touple with data
X_train, X_test, y_train, y_test, df5 = data

# Drop label for testing/training group
X_train, X_test = X_train.drop(['group', 'rand'], 1), X_test.drop(['group', 'rand'], 1)

#%%
from sklearn.feature_selection import r_regression
from sklearn.feature_selection import SelectKBest

# Take correlation
cor = r_regression(X_train, y_train)

# Selects 15 highest correlating features
feats = SelectKBest(score_func=r_regression, k=15).fit(X_train, y_train)

# Get scores + feature names
f_support = feats.get_support() # correlation scores
f_scores = feats.scores_
f_names = feats.get_feature_names_out() # Feature names

# Save scores to dataframe
f_sel = pd.DataFrame(f_scores, index=X_train.columns).reset_index().rename(columns={0 : 'Correlation'})


# Filter out columns with low correlation
f_sel = f_sel[f_sel['index'].isin(f_names)]


# Get features as list
feats = list(f_sel['index'])

# f_sel = f_sel.set_index('index')

# Plot f-correlations
sns.set(rc={"figure.figsize":(6, 6)}, palette='mako', style='whitegrid')
sns.barplot(data = f_sel, x='Correlation', y='index', palette='mako')
plt.suptitle('Selected features by Pearson correlation')
plt.show()

#%%

"""
Correlation feature selector
"""
def fcorr_selector(X_train, y_train):
    
    from sklearn.feature_selection import f_regression
    from sklearn.feature_selection import SelectKBest
    
    # Take correlation
    cor = f_regression(X_train, y_train)
    
    # Selects 15 highest correlating features
    feats = SelectKBest(score_func=f_regression, k=15).fit(X_train, y_train)
    
    # Get scores + feature names
    f_support = feats.get_support() # correlation scores
    f_scores = feats.scores_
    f_names = feats.get_feature_names_out() # Feature names
    
    # Save scores to dataframe
    f_sel = pd.DataFrame(f_scores, index=X_train.columns).reset_index().rename(columns={0 : 'F-correlation'})
    
    # Filter out columns with low correlation
    f_sel = f_sel[f_sel['index'].isin(f_names)]
    
    # Get features as list
    feats = list(f_sel['index'])

    # f_sel = f_sel.set_index('index')

    # Plot f-correlations
    sns.set(rc={"figure.figsize":(6, 6)}, palette='mako', style='whitegrid')
    sns.barplot(data = f_sel, x='F-correlation', y='index', palette='mako')
    plt.suptitle('Selected features by F-score')
    plt.show()
    
    return feats

feats = fcorr_selector(X_train, y_train)


# Apply feature selection to training and test data
X_train, X_test = X_train[feats], X_test[feats]


# Apply feature selection to test and training set
X_train, X_test = X_train[feats], X_test[feats]

# Add in past emissions - indexes make sure data is joined correctly
# em5 = min_max_norm(em5)
# X_train, X_test = X_train.join(em5), X_test.join(em5)

#%%

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import shap

# Define regressor
XG = XGBRegressor(learning_rate=0.1, gamma = 0.1 , max_depth=100, min_child_weight=10, n_estimators=2000)

# Fit model
XG.fit(X_train, y_train)
score = XG.score(X_train, y_train) #r2 score in the training set

# Get Accuracy measures
yhat = XG.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, yhat))
mape = mean_absolute_percentage_error(y_test, yhat)
r2 = XG.score(X_test, y_test)

print('XG Boost performance')
print('RMSE: ', rmse)
print('MAPE: ', mape)
print('r2: ', r2)

# Plot results
sns.set(rc={"figure.figsize":(18, 6)}, style='whitegrid')
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label = 'Emissions 2020')
plt.plot(x_ax, yhat, label = 'Predicted values')
plt.title('XGboost predicted carbon emissions, 2014-2020', size=18)
plt.ylabel('log tonnes of CO2e')
plt.xlabel('Municipalities')
plt.legend()
plt.show()
plt.close()

# plot exponentiated results
sns.set(rc={"figure.figsize":(18, 6)}, style='whitegrid')
x_ax = range(len(y_test))
plt.plot(x_ax, np.exp(y_test), label = 'Emissions 2020')
plt.plot(x_ax, np.exp(yhat), label = 'Predicted values')
plt.title('XGboost predicted carbon emissions, 2014-2020', size=18)
plt.ylabel('tonnes of CO2e')
plt.xlabel('Municipalities')
plt.legend()
plt.show()
plt.close()

# Get shap scores
explainer = shap.TreeExplainer(XG)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")

shap.summary_plot(shap_values, X_test)


# Get feature importances
feature_importance = XG.get_booster().get_score(importance_type='weight')
feat_imp = pd.DataFrame.from_dict(
    feature_importance, orient='index'
    ).rename(
        columns= {0 : 'Feature importance'}
        ).reset_index()
feat_imp['index'] = feat_imp['index'].str.replace('([_])', ' ')
feat_imp = feat_imp.set_index('index')

# Plot results
sns.set(rc={"figure.figsize":(6, 6)}, style='whitegrid')
sns.barplot(y=feat_imp.index, x=feat_imp['Feature importance'], palette='mako')
plt.suptitle('XG-Boost model feature importance, weight score', size = 16)
plt.ylabel('Variable')

#%%

"""
Naive model placeholder

"""

# import statsmodels.api as sm
# from linearmodels.panel import PooledOLS
# import math
# from sklearn.metrics import mean_squared_error, r2_score

# # Run pooled OLS on training data
# ols = PooledOLS(y_train, X_train)
# ols_result = ols.fit(cov_type='clustered', cluster_entity=True)


# # get predicted values
# ols_y = np.array(ols_result.predict(X_test))

# # # Calculate rmse
# # ols_MSE = np.square(np.subtract(y_test,ols_y[:,0])).mean() 
# # ols_RMSE = math.sqrt(ols_MSE)

# # Calculate mape
# def mean_absolute_percentage_error(y_true, y_pred): 
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ols_mape = mean_absolute_percentage_error(y_test, ols_y)

# # obtain r2 on test data
# ols_r2 = r2_score(y_test, ols_y)

# print('Pooled OLS performance:')
# print('Pooled OLS RMSE: ', ols_RMSE)
# print('Pooled OLS MAPE: ', ols_mape)
# print('Pooled OLS R2: ', ols_r2)

# # Set plot size
# sns.set(rc={"figure.figsize":(18, 6)}, style='whitegrid')

# # plot results
# x_ax = range(len(y_test))
# plt.plot(x_ax, y_test, label = 'Emissions 2020')
# plt.plot(x_ax, ols_y, label = 'POLS predicted')
# plt.title('Linear regression predicted carbon emissions, panel 5yr lookahead', size=18)
# plt.ylabel('log 1000-tonnes of CO2e')
# plt.legend()
# plt.show()

#%%

"""
Naive ols

"""
#Linear regression forecast
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set plotsize
sns.set(rc={"figure.figsize":(18, 6)}, style='whitegrid')

# Create training and test data
X_ols = np.array(X_train['l_tCO2e']).reshape(-1,1)
X_ols_test = np.array(X_test['l_tCO2e']).reshape(-1,1)

# Fit naive model
regr = LinearRegression()
regr.fit(X_ols, y_train)

# Predicted values
pred_y = regr.predict(X_ols_test)

# Get prediction error and mape
n_r2 = r2_score(y_test, pred_y)
n_rmse = np.sqrt(mean_squared_error(y_test, pred_y))
n_mape = mean_absolute_percentage_error(y_test, pred_y)


# Print results
print('Naive r2:', n_r2)
print('Naive rmse:', n_rmse)
print('Naive mape:', n_mape)

# plot results
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label = 'Emissions 2020')
plt.plot(x_ax, pred_y, label = 'Naive OLS predicted')
plt.title('Linear regression predicted carbon emissions, panel 5yr lookahead', size=18)
plt.ylabel('log mktCO2e')
plt.legend()
plt.show()

#%%

#Linear regression forecast
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Define andfit models
regr = LinearRegression()
regr.fit(X_train, y_train)

# Get predicted values
pred_y = regr.predict(X_test)


# Get accuracy measures
yhat = regr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, yhat))
mape = mean_absolute_percentage_error(y_test, yhat)

# Print measures
# print("Coefficients: \n", regr.coef_)
print()
print('Linear regression performance: ')
print('RMSE: ', rmse)
print('MAPE :', mape)
print("R2 scrore: %.2f" % r2_score(y_test, pred_y))

sns.set(rc={"figure.figsize":(18, 6)}, style='whitegrid')

# plot results
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label = 'Emissions 2020')
plt.plot(x_ax, pred_y, label = 'Lin-reg predicted')
plt.title('Linear regression predicted carbon emissions, panel 5yr lookahead', size=18)
plt.ylabel('log 1000-tonnes of CO2e')
plt.legend()
plt.show()

#%%

# grid search hyperparameters for the elastic net
from numpy import arange
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

# define parameters to search for 
grid = dict() # parameters are stored in a dict
grid['alpha'] = arange(0, 0.5, 0.01) # parameters to try for alpha
grid['l1_ratio'] = arange(0, 0.5, 0.01) # parametersto try for l1 ratio
grid['tol'] = [10]

# grid search hyperparameters for the elastic net
from numpy import arange
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet

def el_net_GS(X_train, X_test, y_train, y_test, grid):
    
    """
    Grid needs to be of type dict - best defined outside of the function
    """
    
    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test = y_test
    
    # define model
    elnet = ElasticNet()
    
    # define search object - using regular test/split
    search = GridSearchCV(elnet, grid, scoring='neg_mean_squared_error', n_jobs=-1)
    
    # perform the search
    el_results = search.fit(X_train, y_train)
    
    # Print best parameters
    print('Best parameters: ', el_results.best_params_)
    
    return el_results

# el_net_GS(X_train, X_test, y_train, y_test, grid)

#%%

def el_net(X_train, X_test, y_train, y_test):

    # Rerun elasticnet model with best parameters
    # define model
    elnet = ElasticNet(alpha= 0.05, l1_ratio=0.01, tol=10)
    el_res = elnet.fit(X_train, y_train)
    
    # Predict future values, get prediction error
    yhat = elnet.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, yhat))
    r2 = elnet.score(X_test, y_test) #r2 score in the test set
    mape = mean_absolute_percentage_error(y_test, yhat)
    
    # Print prediction errors
    print('Elastic-net scores:')
    print('RMSE: ', rmse)
    print('MAPE: ', mape)
    print('R2 score:', r2)
    
    sns.set(rc={"figure.figsize":(18, 6)}, style='whitegrid')
    
    # Plot results
    # sns.set(rc={"figure.figsize":(12, 4)}, style='whitegrid', palette='deep')
    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label = 'Emissions per municipality 2019')
    plt.plot(x_ax, yhat, label = 'Elastic-net predicted values')
    plt.title('Elastic-net predicted carbon emissions, panel 5 yr lookahead', size=18)
    plt.ylabel('log 1000-tonnes of CO2e')
    plt.xlabel('Municipalities')
    plt.legend()
    plt.show()

el_net(X_train, X_test, y_train, y_test)

#%%

#Re run winning model

# Define regressor
XG = XGBRegressor(learning_rate=0.1, gamma = 0.1 , max_depth=100, min_child_weight=10, n_estimators=800)

# Fit model
XG.fit(X_train, y_train)
score = XG.score(X_train, y_train) #r2 score in the training set

# Get Accuracy measures
yhat = XG.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, yhat))
mape = mean_absolute_percentage_error(y_test, yhat)
r2 = XG.score(X_test, y_test)

xghat = np.exp(yhat)

print('XG Boost performance')
print('RMSE: ', rmse)
print('MAPE: ', mape)
print('r2: ', r2)

# plot winning model
sns.set(rc={"figure.figsize":(18, 6)}, style='whitegrid')
x_ax = range(len(y_test))
plt.plot(x_ax, np.exp(y_test), label = 'Emissions per municipality 2019')
plt.plot(x_ax, np.exp(yhat), label = 'Elastic-net predicted values')
plt.title('Elastic-net predicted carbon emissions, panel 5 yr lookahead', size=18)
plt.ylabel('1000-tonnes of CO2e')
plt.xlabel('Municipalities')
plt.legend()
plt.show()

# Get feature importances
feature_importance = XG.get_booster().get_score(importance_type='weight')
feat_imp = pd.DataFrame.from_dict(
    feature_importance, orient='index'
    ).rename(
        columns= {0 : 'Feature importance'}
        ).reset_index()
feat_imp['index'] = feat_imp['index'].str.replace('([_])', ' ')
feat_imp = feat_imp.set_index('index')

# Plot results
sns.set(rc={"figure.figsize":(6, 4)}, style='whitegrid')
sns.barplot(y=feat_imp.index, x=feat_imp['Feature importance'], palette='mako')
plt.suptitle('XG-Boost model feature importance F-scores', size = 16)
plt.ylabel('Variable')

#%%

"""
Forecast for 2025

"""

# Import data from 2020
df20 = pd.read_csv(r"C:\Users\bengo\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\Machine learning data\treated data\master_data.csv")
df20 = df20.loc[df20['År'] > 2012].dropna(axis=0)

# Extract amount of kommuner in analysis
print('Amount of municipalities in final predictions:')
print(df20['region'].nunique())

# 
df20 = df20.set_index(['region', 'År'])

# Set index for emissions data
em = em.set_index(['region', 'År'])

# Join in historical emissions
df20 = df20.join(em['past emissions l_tCO2e'])

# Add reduce all but selected features
X_new = df20[feats]

# Normalise variables
X_new = min_max_norm(X_new)

#%%

# Create index
idx = X_new.index

# Predict 5 years ahead
yhat_xg = pd.DataFrame(np.exp(XG.predict(X_new)))

# Set index to predicted values
yhat_xg.index = idx

# count amount of kommuner in ananlysis
# yhat_xg.count()

# predicted values using regression
yhat2 = pd.DataFrame(np.exp(regr.predict(X_new)))
yhat2.index = idx

# XG forecasts -> winning ones
yhat_xg = yhat_xg.reset_index()
yhat_xg_2025 = yhat_xg.loc[yhat_xg['År'] == 2020]


# Make aggreagate predictions
aggpred = yhat_xg.groupby('År').sum()

# linreg
yhat_new = yhat2.reset_index()
# yhat_new = yhat_new.loc[yhat_new['År'] == 2020]
yhat_new = yhat_new.rename(columns={0 : 'Predicted Emissions lin-reg'})
linpred = yhat_new.groupby('År').sum()

# plot winning model
sns.set(rc={"figure.figsize":(10, 5)}, style='white')
x_ax = range(len(yhat_xg))
plt.plot(x_ax, yhat_xg.iloc[:,2], label = 'Emissions per municipality 2025')
plt.title('Lin-reg predicted carbon emissions for 2025', size=18)
plt.ylabel('tCO2e')
plt.xlabel('Municipalities')
plt.legend()
plt.show()


yhat_new = yhat_xg.set_index('region')
yhat_new['År'] = 2025
yhat_new[0].nlargest(10)

#Write predictions to file
yhat_xg.to_csv('XG_Corr_panel_predictions.csv')

#%%

# Import total historical emissions from municipalities
# tot_kom = pd.read_csv(r"C:\Users\bengo\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\Miljødirektoratet\tot_kom_utslipp.csv",)

tot_kom = pd.read_excel(r'C:\Users\bengo\OneDrive - Universitetet i Oslo\NMBU post-master\Fag\Masteroppgave\Data\Miljødirektoratet\utslippsstatistikk_alle_kommuner.xlsx',
                        sheet_name='Oversikt - sektorfordelt')

tot_kom = em.reset_index().groupby('År').sum()

tot_kom['Aggregated regional emissions'] = tot_kom['Utslipp (tonn CO2-ekvivalenter)']

tot_kom['']

# Define year column
aggpred['Year'] = list(range(2018,2026))
linpred['Year'] = list(range(2018,2026))
aggpred = aggpred.rename(columns={0 : 'Predicted Emissions XG'})

sns.set(style='darkgrid')
sns.lineplot(data=tot_kom, x='År', y='Utslipp (tonn CO2-ekvivalenter)')
sns.lineplot(data=aggpred, x='Year', y = 'Predicted Emissions XG')
# sns.lineplot(data=linpred, x='Year', y = 'Predicted Emissions lin-reg')
plt.suptitle('Predicted aggregated emissions for 2018-2020 vs actual emissions', size = 16)
plt.legend(['Aggregated regional emissions'
            , 'Predicted emissions (Panel-Corr-XG)'
            , 'Predict emissions (Panel-Corr-Linreg)'])
plt.ylabel('tCO2e')
plt.xlabel('Year')

#%%



