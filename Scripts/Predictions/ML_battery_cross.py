# -*- coding: utf-8 -*-
"""
Created on Tue May  3 23:25:35 2022

@author: bened
"""

import os
import pandas as pd
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

#Set figure style
sns.set(style='whitegrid', palette='mako')

# Desktop directory
os.chdir(r'C:/Users/bened/OneDrive - Norwegian University of Life Sciences/Fag/Masteroppgave/Data/Machine learning data/ML_ready')

# Laptop directory
# os.chdir(r'C:/Users/bengo/OneDrive - Norwegian University of Life Sciences/Fag/Masteroppgave/Data/Machine learning data/ML_ready')


# Import data
em = pd.read_csv('emissions_master.csv')
df1 = pd.read_csv('master_data_t1.csv')
df2 = pd.read_csv('master_data_t2.csv')
df3 = pd.read_csv('master_data_t3.csv')
df4 = pd.read_csv('master_data_t4.csv')
df5 = pd.read_csv('master_data_t5.csv')



#%%

# Helper functions

def long_to_wide(df):
    
    df = df.reset_index()
    
    # Melt into long
    df = df.melt(id_vars=['region', 'År'])
    
    #Pivot to wide
    df = df.pivot_table(columns=['variable', 'År'], index='region')
    
    # MErge multileve columns
    df = df.droplevel(0, axis=1)
    df.columns = df.columns.map('{0[0]} {0[1]}'.format) 
    
    return df

# Adder function for historical emissions
def hist_em_adder(df, em):
     
    df = df.reset_index()
    
    # Add historical emissions
    df, em = df.set_index(['region', 'År']), em.set_index(['region', 'År'])
    df = df.join(em['tCO2e'])
    
    df = df.reset_index()
    
    return df

# Used for plotting predictor importance from RF feature selector
def barplot_feat_imp(df):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df = df
    
    df = pd.DataFrame(df)
    sns.set(style='whitegrid', rc = {'figure.figsize':(10,8)}) # need to set size
    plot = sns.barplot(y = df.iloc[:,0], x = df.iloc[:,1], palette='mako')
    # plot.bar_label(plot.containers[0])
    plot.set(ylabel ='Sector', xlabel='Feature importance')
    # plt.legend(list(df.columns))
    plt.suptitle('Feature importance using mean decrease in impurity', size = 18)

    plt.show()


#%%

# Pivot dataframe with emissions to wide for use for baseline models and regressions
em2 = em.pivot(index='region', columns='År', values='l_tCO2e')
em2 = em2.add_prefix('mtCO2e_')


#%%

"""
Trial with prediction 5 years ahead - 5 years chosen because its a mix between far enough
ahead to be interesting, whilst still not diminishing the data too much.

"""

# Add historical emissions
# df5 = hist_em_adder(df5, em) # Turns on usage of historical emissions

# Make wide version of dataset
dfw = df5.pipe(long_to_wide)

# cor = dfw.corr()['tCO2e_t5 2014']
# cor = cor[cor > 0.45]

y = dfw['tCO2e_t5 2014']

# Get rid of future values - prevents dataleakage
X = dfw[dfw.columns.drop(list(dfw.filter(regex='tCO2e_')))]

# Concat y and x together
df = pd.concat([y, X], axis =1)


349 #%%

# Standardise data
def standardise_func(Xn):
    Xn = (Xn-Xn.mean())/Xn.std()
    return Xn

# Apply min-max normalisation formula
def min_max_norm(df):
    Xn = df
    Xn = (Xn-Xn.min())/(Xn.max()-Xn.min())
    return Xn

Xn = X.pipe(min_max_norm).dropna(axis=1)

#%%

"""
Trying clustering once more...

"""

# Clustering
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

def optimum_clusters(df):
    
    X = df
    
    # Finding optimal levels of clusters
    neigh = NearestNeighbors(n_neighbors=100)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Plotting of distances
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    
    # plt.ylim(0, 600)
    plt.suptitle('Elbow-plot')
    return plt.plot(distances)
    
Xn.pipe(optimum_clusters)


def db_scan(X):
    
    X = X
    
    # hyper parameters for dbscan
    epsilon = 1.5
    min_samples = 5
    
    # Do DBscan for clusters
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    return n_clusters_, n_noise_, labels


db_res = Xn.pipe(db_scan)

print(db_res[0])

"""
Results indicate 2 clusters

Trying K-means as well to verify
"""

#%%

"""
Attempting K-means clustering

"""
def k_means_optimum(X):
    
    Xn = X
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import seaborn as sns
    sns.set()
    
    sil = []
    kmax = 50
    
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k, n_init=6).fit(Xn)
        labels = kmeans.labels_
        sil.append(silhouette_score(df, labels, metric = 'euclidean')) # Calculates average silhouette score
    sns.set(rc = {'figure.figsize':(6,5)})
    plt.plot(list(range(0, len(sil))), sil, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Silhouette score') 
    plt.title('Silhouette analysis for opt. k-clusters - wide form data', size=18)
    sns.set
    plt.show()
    
# Apply k-means function
k_means_optimum(X)

"""
Conclusion: No clusters, it's all just one big blob. Which perhaps goes to show that the data at hand is shite
"""

#%%

"""
Add in last features, then cross validate
"""

# Adding in clusters as separate feature
# Xn['cluster'] = db_res[2]
# X['cluster'] = db_res[2]

# normalise historical emissions
em_n = min_max_norm(em2)

#%%


"""
Part 2: Machine learnining predictions

"""

#Split into test and training sets
from sklearn.model_selection import train_test_split

# Define test and train variables
X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.3)

#%%

"""
Random forest feature selector

"""
# Import modules
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline    
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error


# Get feature names
feature_names = list(X_train.columns)

# Automatic feature selector 
sel = SelectFromModel(RandomForestRegressor(n_estimators = 40))

# Fit selector
sel.fit(X_train, y_train)


# List of selected features
selected_feat = X_train.columns[(sel.get_support())]
len(selected_feat)


# importances
imp = sel.estimator_.feature_importances_
std = np.std([sel.estimator_.feature_importances_ for tree in sel.estimator_.feature_importances_], axis=0)
sel_f = pd.DataFrame(imp, index=feature_names).reset_index().rename(columns={0 : 'imp'})

# Filter out unused features
sel_f = sel_f[sel_f['index'].isin(selected_feat)]

# Plot selected predictors
barplot_feat_imp(sel_f)

# Make list of selected predictors 
feats = list(sel_f['index'])
# feats.append('cluster')

# Keep only selected features
X_train, X_test = X_train[feats], X_test[feats]



#%%

# add in emissions prior to 2015
X_train = X_train.reset_index().merge(
      em_n[['mtCO2e_2014', 'mtCO2e_2013', 'mtCO2e_2012']], 
    on='region', how='left').set_index('region') 
    
X_test = X_test.reset_index().merge(
      em_n[['mtCO2e_2014', 'mtCO2e_2013', 'mtCO2e_2012']], 
    on='region', how='left').set_index('region') 

#%%

# Import metrics from sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

"""
Calculation of baseline model prediction
I.e. make a guess based on mean

Useful for seeing if our model is better or worse than baseline guess

Note: Estimates from this model are different from the naive AR 4 model which 
"""
def baseline_pred(y_train):
    
    import scipy
    
    # "Learn" the mean from the training data
    mean_train = np.mean(y_train)
    
    # Create mean -> as good as just guessing the expected value, i.e no prediction
    baseline_predictions = np.ones(y_test.shape) *  mean_train
    
    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(bX, bY)
    
    # Calculate test scores
    rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline_predictions))
    MAPE_baseline = mean_absolute_error(y_test, baseline_predictions)
    r2 = r2_score(y_test, baseline_predictions)
    
    print("Baseline RMSE is {:.2f}".format(rmse_baseline))
    print('Baseline MAPE is {:.2f}'.format(MAPE_baseline))
    print('Baseline r2 is {:.2f}'.format(r2))
    
    # print(r_value)
    
    
    return rmse_baseline, MAPE_baseline

baseline = baseline_pred(y_train)
    
#%%

"""
Naive vector autoregressive models
"""

import statsmodels.formula.api as smf

# Desktop
os.chdir(r'C:\Users\bened\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\Machine learning data\treated data')

# Laptop
# os.chdir(r'C:\Users\bengo\OneDrive - Norwegian University of Life Sciences\Fag\Masteroppgave\Data\Machine learning data\treated data')

# Import dataframe with emissions
em = pd.read_csv('emissions_master.csv')
em = em.pivot_table(index='region', columns='År', values='l_tCO2e')
em = em.add_prefix('mtCO2e_')

# emX = em[['mtCO2e_2015', 'mtCO2e_2014', 'mtCO2e_2013', 'mtCO2e_2012']]
# emy = em['mtCO2e_2020']

# emX_train, emX_test, emy_train, emy_test = train_test_split(emX, y, test_size=0.3)


# # Run regression - naive model
# naive_mod = smf.ols(em_ytrain)
# naive_res = naive_mod.fit()

# Run trend model
ar4_mod = smf.ols('mtCO2e_2020 ~ mtCO2e_2015', data = em)
ar4_res = ar4_mod.fit()

# # Get predicted values
# n_pred = naive_res.predict()

# Get predicted values
ar4_pred = ar4_res.predict()
ar4_r2 = ar4_res.rsquared # get r-squared
ar_rmse = np.sqrt(ar4_res.mse_total) # extract MSE from statsmodels

# manual calculation of MAPE for ar4 model
def mape_func(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate mape
ar_mape = mean_absolute_error(em['mtCO2e_2020'], ar4_pred) 

print('AR RMSE:', ar_rmse)
print('AR MAPE:', ar_mape)
print('AR r2:', ar4_r2)


#%%
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor

"""
Running this cell will take a VERY long time on a powerful computer due to extensive
gridsearch and CPU hungry algorithm
"""

# # Define model
# XG = XGBRegressor()

# # Redo feature selector outside of function for feature selection
# feature_selector = SelectFromModel(
#     RandomForestRegressor(n_jobs=-1, n_estimators=40), threshold="mean"
# )

# # Make pipeline
# pipeline = make_pipeline(
#    feature_selector, XGBRegressor(n_jobs=-1)
#     )


# # Define parameter search space dict
# grid = {
#         'min_child_weight': 5,
#         'gamma': [0.5],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'max_depth': list(range(40, 50, 2)),
#         'n_estimators': list(range(40, 200, 10))
#         }



# # define search object - using regular test/split
# grid_search = GridSearchCV(
#    XG,
#     param_grid=grid,
#     scoring = 'neg_mean_squared_error',
#     n_jobs = -1,
#     verbose=True
# )

# # perform the search
# results = grid_search.fit(X_train, y_train)


# # Print best parameters
# print('Best parameters: ', results.best_params_)


#%%

sns.set(rc={"figure.figsize":(18, 6)}, style='whitegrid', palette='deep')
    
# Rerun model with best params
# define model
XG = XGBRegressor(learning_rate=0.05, gamma = 0.05 , max_depth=60, min_child_weight=6, n_estimators=200)
XG_res = XG.fit(X_train, y_train)

# For the model and get r2 score
XG.fit(X_train, y_train)
# score = XG.score(X_train, y_train) #r2 score in the training set

# Predict future values, get prediction error
XGyhat = XG.predict(X_test)
XGrmse = np.sqrt(mean_squared_error(y_test, XGyhat))
XGmape = mean_absolute_percentage_error(y_test, XGyhat)
XGr2 = XG.score(X_test, y_test) #r2 score in the test set
# mae = mean_absolute_error(y_test, XGyhat)

# Print prediction errors
print('XG Boost scores:')
print('RMSE: ', XGrmse)
print('MAPE: ', XGmape)
print('R2 score:', XGr2)

# Plot results
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label = 'Emissions 2020')
plt.plot(x_ax, XGyhat, label = 'XG-Boost predicted emissions')
plt.title('XGboost predicted carbon emissions, 5 year lookahead', size=18)
plt.ylabel('log tonnes CO2e')
plt.xlabel('Municipalities')
plt.legend()
plt.show()

#%%

#Linear regression forecast
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def lin_reg(X_train, X_test, y_train, y_test):
    
    X_train = X_train
    X_test = X_test
    y_train = y_train
    y_test = y_test
    
    # Linear regression model has no hyperparameters to tune, hence no gridsearch
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    
    # # The coefficients
    # print("Coefficients: \n", mod_regr.coef_)
    
    # Predict future values, get prediction error
    yhat = regr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, yhat))
    r2 = regr.score(X_test, y_test)
    mape = mean_absolute_percentage_error(y_test, yhat)
    
    # Print prediction errors
    print('Linear regression scores:')
    print('RMSE: ', rmse)
    print('MAPE: ', mape)
    print('R2 score:', r2)
    
    # Plot results
    
    x_ax = range(len(y_test))
    plt.plot(x_ax, np.exp(y_test), label = 'Emissions 2020')
    plt.plot(x_ax, np.exp(yhat), label = 'Lin-reg predicted values')
    plt.title('Linear regression predicted carbon emissions, 5 year lookahead', size=18)
    plt.ylabel('tonnes CO2e')
    plt.xlabel('Municipalities')
    plt.legend()
    plt.show()
    
    return yhat, rmse, r2, mape

# Run model
lin_reg(X_train, X_test, y_train, y_test)


#%%


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
    elnet = ElasticNet(alpha= 0.04, l1_ratio=0.47, tol=10)
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
    
    # Plot results
    # sns.set(rc={"figure.figsize":(12, 4)}, style='whitegrid', palette='deep')
    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label = 'Emissions per municipality 2020')
    plt.plot(x_ax, yhat, label = 'Elastic-net predicted values')
    plt.title('Elastic-net predicted carbon emissions, 5 year lookahead', size=18)
    plt.ylabel('log tonnes of CO2e')
    plt.xlabel('Municipalities')
    plt.legend()
    plt.show()

el_net(X_train, X_test, y_train, y_test)

#%%

# Rerun winning model outside of functions

# define model
XG = XGBRegressor(learning_rate=0.05, gamma = 0.05 , max_depth=60, min_child_weight=6, n_estimators=200)
XG_res = XG.fit(X_train, y_train)

# For the model and get r2 score
XG.fit(X_train, y_train)
# score = XG.score(X_train, y_train) #r2 score in the training set

# Predict future values, get prediction error
XGyhat = XG.predict(X_test)
XGrmse = np.sqrt(mean_squared_error(y_test, XGyhat))
XGmape = mean_absolute_percentage_error(y_test, XGyhat)
XGr2 = XG.score(X_test, y_test) #r2 score in the test set
# mae = mean_absolute_error(y_test, XGyhat)

# Print prediction errors
print('XG Boost scores:')
print('RMSE: ', XGrmse)
print('MAPE: ', XGmape)
print('R2 score:', XGr2)

# Plot results
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label = 'Emissions 2019')
plt.plot(x_ax, XGyhat, label = 'XG-Boost predicted emissions')
plt.title('Predicted carbon emissions vs actual emissions', size=24)
plt.ylabel('log tonnes CO2e')
plt.xlabel('Municipalities')
plt.legend(loc='upper left')
plt.show()

# Get feature importance, save to dataframe
feature_importance = XG.get_booster().get_score(importance_type='weight')
feat_imp = pd.DataFrame.from_dict(
    feature_importance, orient='index'
    ).rename(
        columns= {0 : 'Feature importance'}
        ).reset_index()
feat_imp['index'] = feat_imp['index'].str.replace('([_])', ' ')
feat_imp = feat_imp.set_index('index')

# Plot results
sns.set(rc={"figure.figsize":(9, 10)}, style='whitegrid')
sns.barplot(y=feat_imp.index, x=feat_imp['Feature importance'], palette='mako')
plt.suptitle('XG-Boost model feature importance F-scores', size = 16)
plt.ylabel('Variable')


#%%

from sklearn.pipeline import Pipeline

# Redefine winnin model
XG_model = XGBRegressor(learning_rate=0.05, gamma = 0.05 , max_depth=60, min_child_weight=6, n_estimators=200)

# Running winning model
pipeline = Pipeline([
    ('selector', SelectFromModel(RandomForestRegressor(n_estimators = 40))), 
    ('model', XG_model)])

pipeline.fit(X_train, y_train)
pipeline.predict(X_test)



