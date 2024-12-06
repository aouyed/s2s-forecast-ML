
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import  make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import StackingRegressor

TEST_SIZE=0.5

def preprocess_ds(lead):
    #ds_era_p=xr.open_dataset('../data/raw/ERAI_raw_data_P.nc')
    #ds_era_t=xr.open_dataset('../data/raw/ERAI_raw_data_T.nc')
    ds=xr.open_dataset('../data/raw/GEFS_and_Observations_of_P_T_SWE_SM.nc')
    ds_prism=ds[['PRISM_P','PRISM_T']]
    ds_prism=ds_prism.loc[{'basin':0}].copy()
    ds_prism=ds_prism.drop('basin')
    ds_prism=ds_prism.rename({'PRISM_P':'P','PRISM_T': 'T'})
    
    ds_gefs=ds[['GEFS_P','GEFS_T']]
    ds_gefs= ds_gefs.loc[{'basin':0}].copy()
    ds_gefs= ds_gefs.drop('basin')
    ds_gefs=ds_gefs.rename({'GEFS_P':'P','GEFS_T': 'T'})
    ds_gefs=ds_gefs.loc[{'lead':lead}].copy()
    ds_gefs['P']=ds_gefs['P'].mean(dim='ensemble')
    ds_gefs['T']=ds_gefs['T'].mean(dim='ensemble')
    #ds_gefs['T']=ds_gefs['T'].mean(dim='ensemble')
    ds_gefs=ds_gefs.drop('ensemble')
    
   


    return ds_prism, ds_gefs


def stacking():
    level0 = list()
    level0.append(('linear',LinearRegression()))
    level0.append(('rf', RandomForestRegressor()))
    
    level1=RandomForestRegressor()
    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model




def preprocess_df(ds_era_az, ds_gefs):
    df_x_raw=ds_gefs.to_dataframe()
    df_x=df_x_raw
    df_y=ds_era_az.to_dataframe()
    df_x=df_x.drop(['lead'],axis=1)
    df_x.loc[:,'month']=pd.DatetimeIndex(df_x.reset_index()['time']).month
    #df_x=teleconnect(df_x)
    return df_x, df_y

def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores

def teleconnect(df_x):
    df_x.loc[:,'year']=pd.DatetimeIndex(df_x.reset_index()['time']).year
    df_i=pd.read_csv('../data/raw/ersst5.nino.mth.81-10.ascii', sep='   ', engine='python')
    df_i=df_i.drop(['NINO3.4','NINO1+2','NINO3','NINO4'],axis=1)
    df_i['month']=df_i['month']+1
    df_i.loc[df_i.month==13,'month']=1
    df_x= pd.merge(df_x,df_i,how='left',on=['month','year'])  
    df_i=pd.read_csv('../data/raw/norm.nao.monthly.b5001.current.ascii.txt', sep='   ',engine='python')
    df_i['month']=df_i['month']+1   
    df_i.loc[df_i.month==13,'month']=1
    df_x= pd.merge(df_x,df_i,how='left',on=['month','year'])  
    df_x=df_x.drop('year',axis=1)
    return df_x
 
def ml_calculator( X_train, X_test, y_train, y_test, regressor):
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    y_train = regressor.predict(X_train)
    y_pred=pd.DataFrame(y_pred, columns=['P','T'])
    y_train=pd.DataFrame(y_train, columns=['P','T'])
    y_test_trend=y_test.copy()
    y_pred_trend=y_pred.copy()
    print('rmse for P: ' + str(np.sqrt(mean_squared_error(y_test['P'], y_pred['P']))))
    print('rmse for T: ' + str(np.sqrt(mean_squared_error(y_test['T'], y_pred['T']))))
    return y_pred, y_train

def climatology(ds_era_az, ds_gefs):
    climatology_mean = ds_era_az.groupby("time.week").mean("time")
    anomalies = ds_era_az.groupby("time.week") - climatology_mean
    ds_era_az_base=ds_era_az.copy()
    ds_gefs_base=ds_gefs.copy()
    ds_era_az_base.loc[:,'P_anom']=ds_era_az['P'].groupby("time.week")-climatology_mean['P']
    ds_era_az_base.loc[:,'P_climatology']=-ds_era_az_base['P_anom']+ds_era_az_base['P']

    ds_era_az_base.loc[:,'T_anom']=ds_era_az['T'].groupby("time.week")-climatology_mean['T']
    ds_era_az_base.loc[:,'T_climatology']=-ds_era_az_base['T_anom']+ds_era_az_base['T']

    ds_gefs_base.loc[:,'P_anom']=ds_gefs_base['P'].groupby("time.week")-climatology_mean['P']
    ds_gefs_base.loc[:,'T_anom']=ds_gefs_base['T'].groupby("time.week")-climatology_mean['T']

    df_y_base=ds_era_az_base.to_dataframe()
    df_x_base=ds_gefs_base.to_dataframe()
    df_x_base[:,'month']=pd.DatetimeIndex(df_x_base.reset_index()['time']).month
    
    return df_x_base, df_y_base[['P','T']]


def get_models():
    models={}
    models['rf']=RandomForestRegressor()
    models['linear']=LinearRegression()
    #models['stacking']=stacking()
    return models

def main():
   for lead in [0]:
       print('lead week: ' + str(lead))
       ds_era_az, ds_gefs= preprocess_ds(lead)
       X,y= preprocess_df(ds_era_az, ds_gefs)
       models = get_models()
       # evaluate the models and store results
       results, names = list(), list()
       for name, model in models.items():
           scores = evaluate_model(model, X, y)
           results.append(scores)
           names.append(name)
           print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
       
       

       


def main2():
   for lead in [1]:
       print('lead week: ' + str(lead))
       ds_era_az, ds_gefs= preprocess_ds(lead)
       df_x,df_y= preprocess_df(ds_era_az, ds_gefs)
       X_train, X_test, y_train, y_test =train_test_split(
           df_x, df_y, test_size=TEST_SIZE)
       print(X_train.columns)
       
       regressor =RandomForestRegressor() 
       print('stats with teleconnection')
       X_train_t=teleconnect(X_train)
       X_test_t=teleconnect(X_test)
       y_pred,y_train= ml_calculator( X_train_t.drop(['P','T'],axis=1), X_test_t.drop(['P','T'],axis=1), y_train, y_test, regressor)
       #X_train=X_train[['lead','month']]
  
       
       #regressor = RandomForestRegressor()
       #regressor=KNeighborsRegressor(n_neighbors=2)
       regressor=LinearRegression()
       X_train=X_train.reset_index()
       X_train.loc[:,['P','T']]=y_train[['P','T']]
       X_train=X_train.set_index('time')
       print('stats')
       y_pred, y_train= ml_calculator( X_train, X_test, y_train, y_test, regressor)
      

    
    




if __name__ == '__main__':
    main()
