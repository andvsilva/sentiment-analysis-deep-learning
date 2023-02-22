###########################################################
### step  1 - loading dataset, preprocessing dataset
# 
# Add description for this code.
# Author: Your name here / Thu Apr 22 11:14:38 2021
# Contact me: name@email.com
###########################################################

# libraries for this project
import pandas as pd
import sys
import itertools
from collections import Counter
import numpy as np
import gc # Garbage Collector interface
import time
from icecream import ic # Never use print() to debug again, ic a high level for debug
import snoop # print the lines of code being executed in a function/ great feature very useful to debug :)

import toolkit as tool # see the file toolkit.py for more info
import feather


# Get start time 
start_time = time.time()

print('**********************************************')
print('Cleaning the dataset...')
print('**********************************************')

df_covid = pd.read_csv('../dataset/datacovid.csv')
print(f"Shape dataset Full:.........observations/rows: {df_covid.shape[0]} and columns: {df_covid.shape[1]}")


# time of execution in minutes
time_exec_min = round( (time.time() - start_time)/60, 4)

print(f'time of execution (preprocessing): {time_exec_min} minutes')
print("the preprocessing is done.")
print("The next step is to do the feature engineering.")
print("All Done. :)")