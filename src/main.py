
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import processors as pr
from bias_correction import BiasCorrection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import  make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor


TEST_SIZE=0.3
LEAD=4


    
def one_stage(lead, regressor, rmses, teleconnect, reg_s):
    print('one_stage')
    print('lead week: ' + str(lead))
    ds_era_az, ds_gefs= pr.preprocess_ds(lead)
    df_x, df_y, rmses=pr.climatology(ds_era_az, ds_gefs,lead,  rmses)
    df_x= pr.preprocess_df(df_x)
    if teleconnect:
        df_x=pr.teleconnect(df_x)
    X_train, X_test, y_train, y_test =train_test_split( df_x, df_y, test_size=TEST_SIZE,random_state=1)
    y_pred,y_train, rmses= pr.ml_calculator( X_train, X_test, y_train, y_test, regressor, lead, rmses)
    print('stats')
    rmses['lead'].append(lead)
    rmses['tele'].append(teleconnect)
    rmses['model'].append(reg_s)
    return rmses



def main():
    rmses={'lead':[],'model':[],'tele':[],'P':[],'T':[], 'P_clim': [], 'T_clim':[],'P_model':[], 'T_model':[]}
    leads=[0,1,2,3,4]
    models={}
    models['rf']=RandomForestRegressor()
    models['linear']= LinearRegression()
    for lead in leads: 
        for tele in [True, False]:
            for reg_s in models:
                #bias_stage(rmses)
                rmses=one_stage(lead, models[reg_s], rmses, tele, reg_s)
    df=pd.DataFrame(data=rmses)
    breakpoint()
    print(df)
    df.to_csv('../data/processed/df_results.csv')


if __name__ == '__main__':
    main()
