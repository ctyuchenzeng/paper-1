
import pyreadr
#import os.path
#import os
#import requests
import pdb
import wget
import pandas as pd
import numpy as np
#import datetime
#import sys
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

import datetime

import os
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.model_selection import RandomizedSearchCV
from xgboost.sklearn import XGBRegressor

import datetime as dt
import pickle
import shap
from sklearn.model_selection import train_test_split


script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

#################################################################################
# 1) Download required air quality future_data

download_path = "/mnt/iusers01/fatpou01/sees01/h29141yz/test_wtv"
Path(download_path).mkdir(parents=True, exist_ok=True)



meta_data_url = "https://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData"
data_url = "https://uk-air.defra.gov.uk/openair/R_data/"


meata_data_filename = '/mnt/iusers01/fatpou01/sees01/h29141yz/test_wtv/AURN_metadata.RData'
#meata_data_filename = 'MAN3_2023.RData'

if os.path.isfile(meata_data_filename) is True:
    print("Meta data file already exists in this directory, will use this")
else:
    print("Downloading Meta data file")
    wget.download(meta_data_url)

# Read the RData file into a Pandas dataframe
metadata = pyreadr.read_r(meata_data_filename)

# Downloading site data for a specific year or years
years = [2018,2019,2020,2021,2022,2023]
#current_year = datetime.datetime.now().year

# If a single year is passed then convert to a list with a single value
if type(years) is int:
    years = [years]
#datetime.datetime.now().year
current_year = datetime.datetime.now().year
years = sorted(years)

# List authorities manually, or fit to all?
manual_selection = True#This means we need to list our authorities.
                        # If in doubt, please check the name of the list_authorities in the metadata file above
                        # We again use a list of authorities as set out below

save_to_csv = save_to_csv = True # We Concatenate dataframes into one file. If you would like to save this to a csv file, set to True
#dont_update = True
# I also provide the ability to plot diurnal comparions for O3, NO2, Ox, PM2.5 and PM10 where available
plot_diurnal_comparison = True

site_data_dict=dict()
site_data_dict_name=dict()

if manual_selection is True:
    list_authorities = ['Manchester']
else:
    list_authorities = metadata['AURN_metadata'].local_authority.unique().tolist()

if plot_diurnal_comparison is True:
    site_data_dict=dict()
    site_data_dict_name=dict()

list_a=list(filter(lambda x: x == x, list_authorities))
# Now cycle through each authority and thus each site within
for local_authority in list(filter(lambda x: x == x, list_authorities)):
#for local_authority in list_authorities:

    # Does the authority data exist?
    
    data_path = download_path+ "/" + local_authority + "/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    print(metadata.keys())

    #pdb.set_trace()
    subset_df = metadata['AURN_metadata'][metadata['AURN_metadata'].local_authority == local_authority]

    # Check to see if your requested years will work and if not, change it
    # to do this lets create two new columns of datetimes for earliest and latest
    datetime_start=pd.to_datetime(subset_df['start_date'].values, format='%Y-%m-%d').year
    #datetime_start=pd.to_datetime(subset_df['start_date'].values, format='%Y/%m/%d').year
    #Problem with the end date is it could be ongoing. In which case, convert that entry into a date and to_datetime
    now = datetime.datetime.now()
    datetime_end_temp=subset_df['end_date'].values
    step=0
    for i in datetime_end_temp:
        if i == 'ongoing':
            datetime_end_temp[step]=str(now.year)+'-'+str(now.month)+'-'+str(now.day)
        step+=1
    datetime_end = pd.to_datetime(datetime_end_temp).year

    earliest_year = np.min(datetime_start)
    latest_year = np.max(datetime_end)

# Need to check valid year range
    proceed = True

    if latest_year < np.min(years):
        print("Invalid end year, out of range for ", local_authority)
        proceed = False
    if earliest_year > np.max(years):
        print("Invalid start year, out of range for ", local_authority)
        proceed = False

    ## Check year range now requested
# If we find we have requested an invalid date range then create a neew list of years
    years_temp = years
    try:
        if np.min(years) < earliest_year:
            print("Invalid start year. The earliest you can select for ", local_authority, " is ", str(earliest_year))
            years_temp = years_temp[np.where(np.array(years_temp) == earliest_year)[0][0]::]
            # sys.exit()
        if np.max(years) > latest_year:
            print("Invalid end year. The latest you can select for ", local_authority, " is ", str(latest_year))
            years_temp = years_temp[0:np.where(np.array(years_temp) == latest_year)[0][0]]
    except:
        print("No valid year range")
        proceed = False
        # sys.exit()
    

    # Create dictionary of all site data from entire download session
    clean_site_data=True

   
if proceed is True:

    for site in subset_df['site_id'].unique():

        site_type = metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site]['location_type'].unique()[0]

        # Do we want to check if the site has 'all' of the pollutants?
        # if check_all is True:
        #    if all(elem in ['O3', 'NO2', 'NO', 'PM2.5', 'temp', 'ws', 'wd'] for elem in metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site]['parameter'].values)
        station_name = metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site]['site_name'].values[0]
        # Create a list of dataframes per each yearly downloaded_data
        # Concatenate these at the end
        downloaded_site_data = []

        for year in years_temp:
            # pdb.set_trace()
            try:
                download_url = "https://uk-air.defra.gov.uk/openair/R_data/" + site + "_" + str(year) + ".RData"
                downloaded_file = site + "_" + str(year) + ".RData"

                # Check to see if file exists or not. Special case for current year as updates on hourly basis
                filename_path = download_path + "/" + local_authority + "/" + downloaded_file

                # pdb.set_trace()
                if os.path.isfile(filename_path) is True and year != current_year:
                    print("Data file already exists", station_name, " in ", str(year))
                else:
                    if os.path.isfile(filename_path) is True and year == current_year:
                        # Remove downloaded .Rdata file [make this optional]
                        os.remove(filename_path)
                        print("Updating file for ", station_name, " in ", str(year))
                    print("Downloading data file for ", station_name, " in ", str(year))
                    wget.download(download_url, out=download_path + "/" + local_authority + "/")

                # Read the RData file into a Pandas dataframe
                downloaded_data = pyreadr.read_r(filename_path)
                # Add coordinates as reference data [will change this to be not entire column]
                downloaded_data[site + "_" + str(year)]['latitude'] = \
                metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site].latitude.values[0]
                downloaded_data[site + "_" + str(year)]['longitude'] = \
                metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site].longitude.values[0]
                downloaded_data[site + "_" + str(year)]['location_type'] = \
                    metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site].location_type.values[0]
                # Append to dataframe list
                downloaded_site_data.append(downloaded_data[site + "_" + str(year)])

            except:
                print("Couldnt download data from ", year, " for ", station_name)

        if len(downloaded_site_data) == 0:
            print("No data could be downloaded for ", station_name)
        # final_dataframe = pd.DataFrame()
        else:
            final_dataframe = pd.concat(downloaded_site_data, axis=0, ignore_index=True)

            final_dataframe['datetime'] = pd.to_datetime(final_dataframe['date'])
            final_dataframe = final_dataframe.sort_values(by='datetime', ascending=True)
            final_dataframe = final_dataframe.set_index('datetime')

            # Add a new column into the dataframe for Ox
            try:
                final_dataframe['Ox'] = final_dataframe['NO2'] + final_dataframe['O3']
            except:
                print("Could not create Ox entry for ", site)

            # Add a new column into the dataframe for Ox
            try:
                final_dataframe['NOx'] = final_dataframe['NO2'] + final_dataframe['NO']
            except:
                print("Could not create NOx entry for ", site)
            # Now save the data frame to a .csv file

            # Now clean the dataframe for missing entries - make this optional!!
            if clean_site_data is True:
                # It might be that not all sites record all pollutants here. In which case I think
                # we just need to cycle through each potential pollutant
                for entry in ['O3', 'NO2', 'NO', 'PM2.5', 'Ox', 'NOx', 'temp', 'ws', 'wd']:
                    if entry in final_dataframe.columns.values:
                        # pdb.set_trace()
                        final_dataframe = final_dataframe.dropna(subset=[entry])
            # Now save the dataframe as a .csv file
            if save_to_csv is True:
                print("Creating .csv file for ", station_name)
                final_dataframe.to_excel(download_path + "/" + local_authority + "/" + site + '.xlsx', index=False, header=True)
                #final_dataframe.to_excel(download_path + "/" + local_authority + "/" + site + '.xlsx', index=False, header=True)

            # Append entire dataframe to all site catalogue dictionary
            site_data_dict[site] = final_dataframe
            site_data_dict_name[site] = metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site]['site_name'].values[0]

# Lets also now convert the wind speed into horizontal and vertical
df = site_data_dict['MAN3']
df['U']=np.cos(np.radians(df['wd']))*df['ws']
df['V']=np.sin(np.radians(df['wd']))*df['ws']
#train_dataset['ds'] = (pd.to_datetime(combined_df['Sdate']))

#load in boundary layer height
bl_df=pd.read_csv('/mnt/iusers01/fatpou01/sees01/h29141yz/test_wtv/BL_picc.csv')
bl_df['datetime'] = pd.to_datetime(bl_df['time'], format='%d/%m/%Y %H:%M')
#bl_df['datetime'] = pd.to_datetime(bl_df['time'])
bl_df=bl_df.set_index('datetime')

#add traffic flow
tr_df=pd.read_csv('/mnt/iusers01/fatpou01/sees01/h29141yz/test_wtv/traffic.csv')
tr_df['datetime'] = pd.to_datetime(tr_df['datetime'], format='%Y/%m/%d %H:%M')
tr_df=tr_df.set_index('datetime')

# Now we fit an XGBoost model to our data
forest_training=pd.merge(df[['NO2', 'NO', 'PM2.5', 'Ox', 'NOx','temp', 'ws', 'wd','U','V']],bl_df[['t2m','msl','u10','v10','blh','d2m']],how='inner',left_index=True,right_index=True)
forest_training=pd.merge(forest_training[['NO2', 'NO', 'PM2.5', 'Ox', 'NOx','temp', 'ws', 'wd','U','V','t2m','msl','u10','v10','blh','d2m']],tr_df[['HGV1005']],how='inner',left_index=True,right_index=True)
#forest_training=pd.merge(forest_training[['NO2', 'NO', 'PM2.5', 'Ox', 'NOx','temp', 'ws', 'wd','U','V','t2m','msl','u10','v10','blh','d2m']],tr_df[['HGV1005','Others1005']],how='inner',left_index=True,right_index=True)
#forest_training=pd.merge(forest_training[['NO2', 'NO', 'PM2.5', 'Ox', 'NOx','temp', 'ws', 'wd','U','V','t2m','msl','u10','v10','blh','d2m']],tr_df[['1005','HGV1005','Others1005']],how='inner',left_index=True,right_index=True)
#forest_training=pd.merge(forest_training[['NO2', 'NO', 'PM2.5', 'Ox', 'NOx','temp', 'ws', 'wd','U','V','t2m','msl','u10','v10','blh','d2m']],tr_df[['1004','1005','1011','1013','1024','1048','1061','1320','1332','9050004393']],how='inner',left_index=True,right_index=True)
#forest_training=pd.merge(forest_training[['NO2', 'NO', 'PM2.5', 'Ox', 'NOx','temp', 'ws', 'wd','U','V','t2m','msl','u10','v10','blh','d2m']],tr_df[['flow']],how='inner',left_index=True,right_index=True)
#forest_training=pd.merge(df[['NO2', 'NO', 'PM2.5', 'Ox', 'NOx','temp', 'ws', 'wd','U','V']],bl_df[['t2m','msl','u10','v10','blh','d2m']], how='inner', left_index=True, right_index=True)
#pdb.set_trace()
#forest_training=forest_training.set_index('datetime')




# now we re-index [where needed] a copy of our dataframe since our Prophet
# mdeol requires a continuous dataset

#pd.merge(historical_data[['trend','weekly','daily','yhat']],train_dataset[['Traffic Volume','Modelled Wind Direction','Modelled Wind Speed','Modelled Temperature','y','PM2.5']], how='inner', left_index=True, right_index=True)
forest_training['hour']=forest_training.index.hour
forest_training['day']=forest_training.index.weekday
forest_training['month']=forest_training.index.month
forest_training['unix']=(forest_training.index - dt.datetime(1970,1,1)).total_seconds()
forest_training['day_of_year']=forest_training.index.day_of_year

day = 24*60*60
year = (365.2425)*day
hour_gap_seconds=60*60*3
forest_training['Day sin'] = np.sin(forest_training['unix'] * (2 * np.pi / day))
forest_training['Day cos'] = np.cos(forest_training['unix'] * (2 * np.pi / day))
forest_training['Year sin'] = np.sin(forest_training['unix'] * (2 * np.pi / year))
forest_training['Year cos'] = np.cos(forest_training['unix'] * (2 * np.pi / year))

#pdb.set_trace()

start_date=forest_training.index.values[0]
end_date=forest_training.index.values[-1]
date_index2 = pd.date_range(start_date, end_date, freq='H')
forest_training_fill = forest_training.reindex(date_index2)
forest_training_fill['hour']=forest_training_fill.index.hour
forest_training_fill['day']=forest_training_fill.index.weekday
forest_training_fill['month']=forest_training_fill.index.month
forest_training_fill['unix']=(forest_training_fill.index - dt.datetime(1970,1,1)).total_seconds()
forest_training_fill['day_of_year']=forest_training_fill.index.day_of_year
forest_training_fill['Day sin'] = np.sin(forest_training_fill['unix'] * (2 * np.pi / day))
forest_training_fill['Day cos'] = np.cos(forest_training_fill['unix'] * (2 * np.pi / day))
forest_training_fill['Year sin'] = np.sin(forest_training_fill['unix'] * (2 * np.pi / year))
forest_training_fill['Year cos'] = np.cos(forest_training_fill['unix'] * (2 * np.pi / year))
# creating bool series True for NaN values
#bool_series = pd.isnull(forest_training_fill["NO2"])
#forest_training_fill[bool_series]

fit_xgboost = True

if fit_xgboost is True:
    #random_grid = {'criterion':criterion,'n_estimators': n_estimators,'max_features': max_features,'max_depth': max_depth,'min_samples_split': min_samples_split,'min_samples_leaf': min_samples_leaf,'bootstrap': bootstrap}
    parameters = {
            'colsample_bytree':[1.0,0.9],
            'gamma':[0,0.03,0.1,0.3],
            'min_child_weight':[1.0,1.5,6,10],
            'learning_rate':[0.3,0.1,0.07,0.01],
            'max_depth':[3,5,8,10],
            'n_estimators':[10000,5000],
            'reg_alpha':[1e-5, 1e-2,  0.75],
            'reg_lambda':[1e-5, 1e-2, 0.45],
            'subsample':[0.8,0.95]  }
    #Lets now fit a XGBOOST model to all the data we have
    xgb_new = XGBRegressor()
    gsearch_new = RandomizedSearchCV(estimator=xgb_new, param_distributions=parameters, n_iter=30, scoring='neg_mean_squared_error', n_jobs=20, cv=4, verbose=2, random_state=1001)
    training_features_new = ['temp','U','V','t2m','msl','u10','v10','blh','d2m','HGV1005','hour','day','month','unix','day_of_year']
    #training_features_new = ['temp','U','V','t2m','msl','u10','v10','blh','d2m','HGV1005','Others1005','hour','day','month','unix','day_of_year']
    #training_features_new = ['temp','U','V','t2m','msl','u10','v10','blh','d2m','1005','HGV1005','Others1005','hour','day','month','unix','day_of_year']
    #training_features_new = ['temp','U','V','t2m','msl','u10','v10','blh','d2m','1004','1005','1011','1013','1024','1048','1061','1320','1332','9050004393','hour','day','month','unix','day_of_year']
    #training_features_new = ['temp','U','V','t2m','msl','u10','v10','blh','d2m','hour','day','month','unix','day_of_year']
    #gsearch_new.fit(np.array(forest_training[training_features_new]), np.array(forest_training['NO2']))
    #split
    Xt,Xv,yt,yv=train_test_split(np.array(forest_training[training_features_new]), np.array(forest_training['NOx']),test_size=0.2,random_state=10)
    gsearch_new.fit(Xt,yt)
    #gsearch_new.fit(np.array(forest_training[training_features_new]), np.array(forest_training['NOx']))
    colsample_bytree_best=gsearch_new.best_params_['colsample_bytree']
    gamma_best=gsearch_new.best_params_['gamma']
    min_child_weight_best=gsearch_new.best_params_['min_child_weight']
    learning_rate_best=gsearch_new.best_params_['learning_rate']
    max_depth_best=gsearch_new.best_params_['max_depth']
    n_estimators_best=gsearch_new.best_params_['n_estimators']
    reg_alpha_best=gsearch_new.best_params_['reg_alpha']
    reg_lambda_best=gsearch_new.best_params_['reg_lambda']
    subsample_best=gsearch_new.best_params_['subsample']
    best_xgb_model_new = XGBRegressor(colsample_bytree=colsample_bytree_best,gamma=gamma_best,learning_rate=learning_rate_best,max_depth=max_depth_best,min_child_weight=min_child_weight_best,n_estimators=n_estimators_best,reg_alpha=reg_alpha_best,reg_lambda=reg_lambda_best,subsample=subsample_best,seed=42)
    #best_xgb_model_new.fit(np.array(forest_training[training_features_new]), np.array(forest_training['NO2']))
    best_xgb_model_new.fit(Xt,yt)
    #best_xgb_model_new.fit(np.array(forest_training[training_features_new]), np.array(forest_training['NOx']))

    # save model to file
    #pickle.dump(best_xgb_model_new, open("/mnt/iusers01/fatpou01/sees01/h29141yz/test_w/NO2_predictor.pickle.dat", "wb"))
    pickle.dump(best_xgb_model_new, open("/mnt/iusers01/fatpou01/sees01/h29141yz/test_wtv/NOx+traffic+valid_predictor.pickle.dat", "wb"))

else:

    # load model from file
    #best_xgb_model_new = pickle.load(open("/mnt/iusers01/fatpou01/sees01/h29141yz/test_w/NO2_predictor.pickle.dat", "rb"))
    best_xgb_model_new = pickle.load(open("/mnt/iusers01/fatpou01/sees01/h29141yz/test_wtv/NOx+traffic+valid_predictor.pickle.dat", "rb"))


#Now we go through a create a weather normalised dataset, which we will add on to our original air quality dataframe
training_features_notime = ['temp','U','V','t2m','msl','u10','v10','blh','d2m','HGV1005','hour','day','month','day_of_year']
#training_features_notime = ['temp','U','V','t2m','msl','u10','v10','blh','d2m','HGV1005','Others1005','hour','day','month','day_of_year']
#training_features_notime = ['temp','U','V','t2m','msl','u10','v10','blh','d2m','1005','HGV1005','Others1005','hour','day','month','day_of_year']
#training_features_notime = ['temp','U','V','t2m','msl','u10','v10','blh','d2m','hour','day','month','day_of_year']
training_features_getaverages = ['temp','U','V','t2m','msl','u10','v10','blh','d2m','HGV1005']
#training_features_getaverages = ['temp','U','V','t2m','msl','u10','v10','blh','d2m','HGV1005','Others1005']
#training_features_getaverages = ['temp','U','V','t2m','msl','u10','v10','blh','d2m','1005','HGV1005','Others1005']
#training_features_getaverages = ['temp','U','V','t2m','msl','u10','v10','blh','d2m']
training_features_time = ['hour','day','month','unix','day_of_year']





# Make predictions on the test set
y_pred = best_xgb_model_new.predict(Xv)

# Evaluate regression metrics on the test set
mae = mean_absolute_error(yv, y_pred)
mse = mean_squared_error(yv, y_pred)
r2 = r2_score(yv, y_pred)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Cross-validation
cv_scores = cross_val_score(best_xgb_model_new, np.array(forest_training[training_features_new]),forest_training['NOx'], cv=5, scoring='neg_mean_squared_error')
#cv_scores = cross_val_score(best_xgb_model_new, data[1:15722,1:643],data[1:15722,643], cv=5, scoring='neg_mean_squared_error')
#cv_scores = cross_val_score(best_xgb_model_new, data[1:43894,1:643],data[1:43894,643], cv=5, scoring='neg_mean_squared_error')
#cv_scores = cross_val_score(model, data[1:15996,0:16],data[1:15996,16], cv=5, scoring='accuracy')
rmse_cv = np.sqrt(-cv_scores) #convert negative MSE to RMSE

print("Cross-Validation Scores:", cv_scores)
print(f"Mean Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores):.2f})")

Mean_RMSE1=np.mean(rmse_cv)
Mean_RMSE2=np.std(rmse_cv)

explainer=shap.TreeExplainer(best_xgb_model_new)
shap_values=explainer.shap_values(Xv)
print("shap_values:",shap_values)
print("Xv:",Xv)

#shap.dependence_plot("temp",shap_values,Xv,interaction_index="RH")
#shap.dependence_plot("temp",shap_values,Xv,interaction_index="month")
#shap.dependence_plot("u10",shap_values,Xv,interaction_index="blh_final")

# Example data (dictionary)
parameters = {'mae_Mean Absolute Error': mae, 'mse_Mean Squared Error': mse, 'r2_R-squared': r2, 'Cross-Validation RMSE Scores': rmse_cv, 'Mean RMSE1': Mean_RMSE1, 'Mean RMSE2': Mean_RMSE2}

# Convert dictionary to DataFrame
df_parameters = pd.DataFrame(list(parameters.items()), columns=['Parameter Name', 'Parameter Value'])

# Specify the file path where you want to save the CSV file
csv_file_path = '/mnt/iusers01/fatpou01/sees01/h29141yz/test_wtv/NOx+traffic+valid_parameters.csv'
#csv_file_path = '\mnt\iusers01\fatpou01\sees01\h29141yz\first_job\parameters.csv'

# Writing data to the CSV file
df_parameters.to_csv(csv_file_path, index=False)

print(f'Data has been written to {csv_file_path}')

# Specify the file path where you want to save the CSV file
csv_file_path1 = '/mnt/iusers01/fatpou01/sees01/h29141yz/test_wtv/NOx+traffic+valid_combined_array_Xv_shap.csv'
#csv_file_path1 = '\mnt\iusers01\fatpou01\sees01\h29141yz\first_job/combined_array_Xv_shap.csv'

# Combine arrays using np.concatenate
combined_array = np.concatenate((Xv, shap_values))

print("Combined array:", combined_array)

# Writing array data to the CSV file
np.savetxt(csv_file_path1, combined_array, delimiter=',')

print(f'Data has been written to {csv_file_path1}')

shap.summary_plot(shap_values,Xv)

plt.savefig('/mnt/iusers01/fatpou01/sees01/h29141yz/test_wtv/NOx+traffic+valid_plot.png')



