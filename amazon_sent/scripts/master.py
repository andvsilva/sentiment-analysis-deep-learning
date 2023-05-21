####################################################################################
## This script is the workflow of the project

### Building...
### step  0 - Installing the libraries via pip3
### step  1 - loading dataset, preprocessing dataset
### step  2 - feature engineering
### step  3 - training and prediction

# Author: 

#####################################################################################

from platform import python_version
import sys
import os
import time
import toolkit as tool # see the file toolkit.py for more info

start_time = time.time()

# script master to run ALL the steps of the project. 
def main():

    # checking the python version:
    if sys.version_info<(3,6,0):
        py_version = python_version()
        print(f'The python version installed in your computer: {py_version}')
        sys.stderr.write("You need python 3.6 or later to run this script\n")
        sys.exit()
        
    else:
        py_version = python_version()
        print(f'python version: {py_version}')
        print('Successfully, go ahead to run the script.')
        
    # step 0    
    #os.system('pip3 install -r requirements.txt')

    # step 1
    os.system('python3 cleaning.py') # > logs/info-preprocess.dat
    
    # step 2
    os.system('python3 featureeng.py') # > logs/feature-eng.dat
    
    # step 3
    os.system('python3 modeling.py') # > logs/training.dat

    time_exec_min = round( (time.time() - start_time)/60, 4)
    
    print(f'time of execution (total pipeline): {time_exec_min} minutes')
    print('Done! please, check the results. :)')

if __name__ == "__main__":
    main()