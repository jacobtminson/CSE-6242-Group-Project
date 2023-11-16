# ---------------------------------------------------------------------------------------------------------------------
# FOLDER STRUCTURE
# ---------------------------------------------------------------------------------------------------------------------

# team003final (root; current directory) 
#     \_ README.txt - a concise, short README.txt file, corresponding to the "user guide" 
#     \_ DOC - a folder with final documentations (team003report.pdf, team003poster.pdf) 
#     \_ CODE - subset of files from Code folder that are required for running model 
#         \_ data_ranked_list.py - main code file to run model 
#         \_ college_audit_ui.py - main code file to run UI (run this file after executing data_ranked_list.py)
#         \_ dtype.txt - a text file used in the data cleaning script to set the datatypes of the columns 
#         \_ location_info.csv - location data of all schools for UI 
#     \_ data - a folder with raw data files (457.4 MB) which are extracted from the 'All Data Files' zip file 
#               downloaded from College Scorecard (https://collegescorecard.ed.gov/data/)  
#     \_ image - a folder with output image files of our model's performance (scatterplot.png) 

# ---------------------------------------------------------------------------------------------------------------------
# INSTALL 
# ---------------------------------------------------------------------------------------------------------------------

import subprocess
import sys

def install_libomp():
    # Check for the operating system and install libomp accordingly
    if sys.platform.startswith("linux"):
        # Use the appropriate package manager for Linux (e.g., apt for Debian/Ubuntu)
        subprocess.run(["sudo", "apt", "install", "libomp-dev"])
    elif sys.platform == "darwin":
        # Use Homebrew for macOS
        subprocess.run(["brew", "install", "libomp"])
    elif sys.platform == "win32":
        # Windows users may need to install libomp manually
        print("libomp not found. Please install libomp manually for Windows.")
    else:
        print("Unsupported operating system. Please install libomp manually.")

def install_lightgbm():
    # Install lightgbm using pip
    subprocess.run([sys.executable, "-m", "pip", "install", "lightgbm"])

def check_and_install_libraries():
    # Check if libomp is installed
    try:
        subprocess.run(["libomp"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("libomp not found. Installing libomp...")
        install_libomp()

    # Check if lightgbm is installed
    try:
        import lightgbm as lgb
    except ImportError:
        print("lightgbm not found. Installing lightgbm...")
        install_lightgbm()

check_and_install_libraries()

# Check and install required packages
required_packages = ['dash', 'matplotlib', 'pandas', 'numpy', 'scikit-learn', 'lightgbm']

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not found. Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package])

# Install scikit-learn if not present
try:
    import sklearn
except ImportError:
    print("scikit-learn not found. Installing scikit-learn...")
    subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn"])

# ---------------------------------------------------------------------------------------------------------------------
# IMPORT 
# ---------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV

# ---------------------------------------------------------------------------------------------------------------------
# RAW DATA - preprocessing
# ---------------------------------------------------------------------------------------------------------------------

# merge and clean

#get list of MERGED files in /data 
files = [
'data/MERGED2014_15_PP.csv',
'data/MERGED2015_16_PP.csv',
'data/MERGED2016_17_PP.csv',
'data/MERGED2017_18_PP.csv',
'data/MERGED2018_19_PP.csv',
'data/MERGED2019_20_PP.csv',
'data/MERGED2020_21_PP.csv',
'data/MERGED2021_22_PP.csv']

#concatenate all files into one dataframe and add a column for the file name
df = pd.concat([pd.read_csv(f).assign(year = int("20"+f.split("_")[1])) for f in files])

#move the year column to the first space
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

#save the dataframe as a csv
df.to_csv('CODE/all_merged_data.csv', index=False)

# read from raw data (downloaded from collegeboard) 

with open("CODE/dtype.txt") as f:
    dtype = eval(f.read())
    df2 = pd.read_csv("CODE/all_merged_data.csv", na_values="PrivacySuppressed", dtype=dtype)

#make a list of columns with > 90% null values
null_cols = df2.columns[df2.isnull().mean() > 0.9]

#drop columns with 90% null values
df2 = df2.drop(columns=null_cols)

df2.to_csv("CODE/cleanish_merged_data_no_privacy.csv", index=False)

merged_data_no_privacy = pd.read_csv("CODE/cleanish_merged_data_no_privacy.csv")

# combining public and private into one column for net price
merged_data_no_privacy["NPT4"] = merged_data_no_privacy[["NPT4_PUB","NPT4_PRIV"]].sum(axis=1, min_count=1)

# D_PCTPELL_PCTFLOAN represents number of undergrad students who received pell grant/federal loan
# UGDS represents number of undergraduate students seeking degree/certificate 
# and this variable seems to be a better representation of size

# UGNONDS is undergrads not seeking degree/certificate, so we combine this variable with UGDS to get total number of undergrads
merged_data_no_privacy["UG"] = merged_data_no_privacy[["UGDS","UGNONDS"]].sum(axis=1, min_count=1)
# total number of students will be total number of undergrads + total number of grads
merged_data_no_privacy["TOTAL_NUM_STUDENTS"] = merged_data_no_privacy[["GRADS","UG"]].sum(axis=1, min_count=1)

# dataset with the variables that we need:
merged_data_final_condensed = merged_data_no_privacy[["UNITID","year","INSTNM","STABBR","CITY","ZIP","LONGITUDE","LATITUDE",
                                                      "HIGHDEG","C100_4","ACTCM25","ACTCM75","ACTEN25","ACTEN75","ACTMT25",
                                                      "ACTMT75","ACTCMMID","ACTENMID","ACTMTMID","SATVR25","SATVR75","SATMT25",
                                                      "SATMT75","SATVRMID","SATMTMID","SAT_AVG","NPT4","ADM_RATE",
                                                      "MD_EARN_WNE_P6","MD_EARN_WNE_P8","MD_EARN_WNE_P10","DEBT_MDN",
                                                      "GRAD_DEBT_MDN","WDRAW_DEBT_MDN","TOTAL_NUM_STUDENTS","FAMINC",
                                                      "MD_FAMINC","MENONLY","WOMENONLY"]]

merged_data_final_condensed.to_csv("CODE/merged_data_final_condensed.csv", index=False)

# ---------------------------------------------------------------------------------------------------------------------
# processing most recent cohort fos data 
# ---------------------------------------------------------------------------------------------------------------------

# read data
MRCF_df = pd.read_csv("data/Most-Recent-Cohorts-Field-of-Study.csv")

# convert the ID column into integer, accounting for nan values
MRCF_df['UNITID'] = MRCF_df['UNITID'].astype("Int64")

# drop columns that are 90% nan
threshold = 0.9 * len(MRCF_df)
mrcf_with_privacy = MRCF_df.dropna(axis=1, thresh=threshold)
MRCF_df.replace('PrivacySuppressed', pd.NA, inplace=True)
mrcf_no_privacy = MRCF_df.dropna(axis=1, thresh=threshold)

mrcf_no_privacy.to_csv("CODE/most_recent_cohorts_fos_final.csv", index=False)

# ---------------------------------------------------------------------------------------------------------------------
# DATA for model 
# ---------------------------------------------------------------------------------------------------------------------

# input data
df_path =  "CODE/merged_data_final_condensed.csv"
rawdata = pd.read_csv(df_path)
newraw = rawdata.copy()

# Family income AVG_FAMINC = mean of FAMINC, MD_FAMINC 
newraw["AVG_FAMINC"] = newraw[["FAMINC","MD_FAMINC"]].mean(axis=1)#, min_count=1)
newraw[["AVG_FAMINC","FAMINC","MD_FAMINC"]]

# Salary EARN = mean of MD_EARN_WNE_P6, MD_EARN_WNE_P8, MD_EARN_WNE_P10 
newraw["EARN"] = newraw[["MD_EARN_WNE_P6","MD_EARN_WNE_P8","MD_EARN_WNE_P10"]].mean(axis=1)#, min_count=1)
newraw[["EARN","MD_EARN_WNE_P6","MD_EARN_WNE_P8","MD_EARN_WNE_P10"]]

# Rename some columns
newraw = newraw.rename(columns={"year": "YEAR", 'C100_4': 'GradRate', 'ADM_RATE': 'AdmRate', 
                                'EARN': 'Salary', 'DEBT_MDN': 'Debt','TOTAL_NUM_STUDENTS':'Size','NPT4':'NetPrice'})
data = newraw.copy()

# Some net price values are negative -> compute absolute value
data['NetPrice'] = np.abs(data['NetPrice'])
data[data['NetPrice']<0]

# Fill the missing values by the MEAN value (metrics)
fix_mean = ['GradRate', 'ACTCM25', 'ACTCM75', 'ACTEN25','ACTEN75', 'ACTMT25', 'ACTMT75', 'ACTCMMID', 'ACTENMID', 'ACTMTMID',
            'SATVR25', 'SATVR75', 'SATMT25', 'SATMT75', 'SATVRMID', 'SATMTMID','SAT_AVG', 'NetPrice', 'AdmRate', 
            'MD_EARN_WNE_P6', 'MD_EARN_WNE_P8','MD_EARN_WNE_P10', 'Debt', 'GRAD_DEBT_MDN', 'WDRAW_DEBT_MDN', 
            'Size','FAMINC', 'MD_FAMINC',  'AVG_FAMINC', 'Salary']

data[fix_mean] = data[fix_mean].fillna(data[fix_mean].mean())

# Fill the missing values by 0 (binary)
data[['WOMENONLY', 'MENONLY']] = data[['WOMENONLY', 'MENONLY']].fillna(0)

# Features to use
features_score = ["GradRate", "NetPrice", "AdmRate", "Salary", "Debt", "Size"]

# Variables
weights = {

    "GradRate": 0.16, # Graduation Rate
    "NetPrice": 0.12, # Net Price
    "AdmRate": 0.14, # Acceptance Rate
    "Salary": 0.24, # Salary
    "Debt": 0.12, # Debt
    "Size": 0.22 # Size
}

# Training data - only has 6 features mentioned above
X = data[features_score]

# Compute actual scores of school using weights
data['Score'] = (X * weights).sum(axis=1)

# Testing data (score) - replace 'Score' with your actual target column name
y = data['Score']

# Split the data into training and testing sets (adjust the test_size as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------------------------------------------------------------------------------------------------
# Hyperparameter tuning 
# ---------------------------------------------------------------------------------------------------------------------
  
# Grid search
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'num_leaves': [20, 30, 40],
    'min_child_samples': [10, 20, 30]
}

# Set the verbosity level to -1 for LightGBM
lgbm_params = {'verbose': -1}

# model = lgb.LGBMRegressor()
model = lgb.LGBMRegressor(**lgbm_params)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# RUN TIME: 30-60min
# best_params = {'learning_rate': 0.1, 'max_depth': 15, 'min_child_samples': 20, 'n_estimators': 300, 'num_leaves': 30}

# ---------------------------------------------------------------------------------------------------------------------
# Run model with best hyperparameters 
# ---------------------------------------------------------------------------------------------------------------------

# 10-fold CV (2014-22) 

# best_model = lgb.LGBMRegressor(**best_params)
# best_model.fit(X_train, y_train)
# y_pred = best_model.predict(X_test)

best_model_params = {**best_params, **lgbm_params}
best_model = lgb.LGBMRegressor(**best_model_params)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# ---------------------------------------------------------------------------------------------------------------------
# Evaluation 1 
# ---------------------------------------------------------------------------------------------------------------------

# Performance metrics

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# GRID SEARCH
# Mean Absolute Error (MAE): 82.04850125417036
# Mean Squared Error (MSE): 169805.87420303287
# Root Mean Squared Error (RMSE): 412.0750832106121
# R-squared (R2): 0.9821655925621944

# ---------------------------------------------------------------------------------------------------------------------
# Generate final list 
# ---------------------------------------------------------------------------------------------------------------------

ranking_df = pd.DataFrame({'YEAR': data.loc[X_test.index, 'YEAR'],'UNITID': data.loc[X_test.index, 'UNITID'],
                        'INSTNM': data.loc[X_test.index, 'INSTNM'],'Actual Score': data.loc[X_test.index, 'Score'], 
                        'Predicted Score': y_pred,'STABBR': data.loc[X_test.index, 'STABBR'],
                        'CITY': data.loc[X_test.index, 'CITY'],'ZIP': data.loc[X_test.index, 'ZIP'],
                        'HIGHDEG': data.loc[X_test.index, 'HIGHDEG'],
                        'GradRate': data.loc[X_test.index, 'GradRate'],'NetPrice': data.loc[X_test.index, 'NetPrice'],
                        'AdmRate': data.loc[X_test.index, 'AdmRate'],'Salary': data.loc[X_test.index, 'Salary'],
                        'Debt': data.loc[X_test.index, 'Debt'],'Size': data.loc[X_test.index, 'Size'],
                        'AVG_FAMINC': data.loc[X_test.index, 'AVG_FAMINC'],
                        'ACTCM75': data.loc[X_test.index, 'ACTCM75'],'ACTCM25': data.loc[X_test.index, 'ACTCM25'],
                        'ACTEN75': data.loc[X_test.index, 'ACTEN75'],'ACTEN25': data.loc[X_test.index, 'ACTEN25'],
                        'ACTMT75': data.loc[X_test.index, 'ACTMT75'],'ACTMT25': data.loc[X_test.index, 'ACTMT25'],
                        'ACTCMMID': data.loc[X_test.index, 'ACTCMMID'],'ACTENMID': data.loc[X_test.index, 'ACTENMID'],
                        'ACTMTMID': data.loc[X_test.index, 'ACTMTMID'],'SATVR75': data.loc[X_test.index, 'SATVR75'],
                        'SATVR25': data.loc[X_test.index, 'SATVR25'],'SATMT75': data.loc[X_test.index, 'SATMT75'],
                        'SATMT25': data.loc[X_test.index, 'SATMT25'],'SATVRMID': data.loc[X_test.index, 'SATVRMID'],
                        'SATMTMID': data.loc[X_test.index, 'SATMTMID'],'SAT_AVG': data.loc[X_test.index, 'SAT_AVG'],
                        'MENONLY': data.loc[X_test.index, 'MENONLY'],'WOMENONLY': data.loc[X_test.index, 'WOMENONLY'],
                        'FAMINC': data.loc[X_test.index, 'FAMINC'],'MD_FAMINC': data.loc[X_test.index, 'MD_FAMINC'],
                        'MD_EARN_WNE_P6': data.loc[X_test.index, 'MD_EARN_WNE_P6'], 'MD_EARN_WNE_P8': data.loc[X_test.index, 'MD_EARN_WNE_P8'],
                        'MD_EARN_WNE_P10': data.loc[X_test.index, 'MD_EARN_WNE_P10'],
                        'LATITUDE': data.loc[X_test.index, 'LATITUDE'],'LONGITUDE': data.loc[X_test.index, 'LONGITUDE']
                        })

# Rank universities based on predicted scores
ranked_universities = ranking_df.sort_values(by='Predicted Score', ascending=False)
ranked_universities.reset_index(drop=True, inplace=True)

# drop duplicates, and only keep highest scores
ranked_universities_unique = ranked_universities.drop_duplicates('INSTNM').sort_index()

# ---------------------------------------------------------------------------------------------------------------------
# Evaluation 2 - scatterplot
# ---------------------------------------------------------------------------------------------------------------------

# Scatterplot of predicted/actual scores from each school - using ranked_list 

true_value_unique = ranked_universities_unique['Actual Score']
predicted_value_unique = ranked_universities_unique['Predicted Score']

plt.scatter(true_value_unique, predicted_value_unique, c='red')
plt.yscale('log')
plt.xscale('log')

p1_ = max(max(predicted_value_unique), max(true_value_unique))
p2_ = min(min(predicted_value_unique), min(true_value_unique))
plt.title('Predicted vs. Actual Scores (2014-22, 10-fold CV)')
plt.plot([p1_, p2_], [p1_, p2_], 'b-')
plt.xlabel('Actual Scores', fontsize=15)
plt.ylabel('Predicted Scores', fontsize=15)
plt.axis('equal')
plt.savefig('image/scatterplot.png')
# plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Add Type of School (CONTROL) to our ranked_list
# ---------------------------------------------------------------------------------------------------------------------

# get type of school from fos and merge with ranked_list
fos = pd.read_csv("CODE/most_recent_cohorts_fos_final.csv")

# only keep unique schools to only get type of school 
unique_school = fos.drop_duplicates('INSTNM').sort_index()

# match by UNITID in ranked_list
ranked_unique_school_type_merged = ranked_universities_unique.merge(
    unique_school[['UNITID','INSTNM','CONTROL','MAIN','CREDDESC']], how='left')

# ---------------------------------------------------------------------------------------------------------------------
# FINAL RANKED_LIST - generate csv file
# ---------------------------------------------------------------------------------------------------------------------

ranked_unique_school_type_merged.to_csv("CODE/ranked_list_final.csv", index=False)

# ---------------------------------------------------------------------------------------------------------------------
# Historical data for historical trends (UI) - generate csv file
# ---------------------------------------------------------------------------------------------------------------------

historical = data.copy()
historical = historical[['YEAR','UNITID','INSTNM','NetPrice','AdmRate','Debt']]
historical.to_csv("CODE/historical_final.csv", index=False)
