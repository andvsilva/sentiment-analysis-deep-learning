###########################################################
### step  3 - training and prediction
#
# Add description for this code.
###########################################################

# libraries for this project
import json
import pandas as pd
import numpy as np
from numpy import mean, std
from IPython.display import HTML
import matplotlib.pyplot as plt
from icecream import ic
#import seaborn as sns
import os.path
import sys
import gc
import feather
from icecream import ic
from sys import getsizeof
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from matplotlib import pyplot
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

# Get start time 
start_time = time.time()

now = datetime.now()
 
print("date..............:", now)

# This make machine learning interpretable and understandable to everyone.
from shapash.explainer.smart_explainer import SmartExplainer

# Here we are going to implement some functions

print("Loading dataset - to modeling...")

# loading data feather format
df_processed = pd.read_feather('data-feather/featureselected.ftr')

print(df_processed)


# time of execution in minutes
time_exec_min = round( (time.time() - start_time)/60, 4)

print(f'time of execution (preprocessing): {time_exec_min} minutes')
print("the modeling is done.")
print("Now is time to check the results, good work.")
print("All Done.")
