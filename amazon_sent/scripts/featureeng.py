###########################################################
### step  2 - feature engineering
# 
# Add description for this code.
###########################################################

# libraries for this project
import json
import pandas as pd
import numpy as np
from datetime import datetime
from IPython.display import HTML
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
from matplotlib.backends.backend_pdf import PdfPages
import sys
import gc
import feather
import toolkit as tool
from icecream import ic
from sys import getsizeof
import time

# Get start time 
start_time = time.time()

now = datetime.now()
 
print("date..............:", now)

# Here we are going to implement some functions

print('*****************************************************')
print('Starting the feature engineering of the dataset.')
print('*****************************************************')

# Loadind data
print("Loading dataset - cleaned to feature engineering...")

df_featuresel = pd.read_feather('data-feather/cleaned.ftr')
print(df_featuresel)

print("saving the file format feather...")

# this is important to do before save in feather format.
df_featuresel = df_featuresel.reset_index(drop=True)

# saving in the feather format
df_featuresel.to_feather('data-feather/featureselected.ftr')

# time of execution in minutes
time_exec_min = round( (time.time() - start_time)/60, 4)

print(f'time of execution (preprocessing): {time_exec_min} minutes')
print("the feature engineering is done.")
print("The next step is to do the modeling.")
print("All Done.")

