***** THIS IS NOT OUR ACTUAL README.txt *****



CSE 6242 Group Project

Project Title : College Audit 
Team Name: Full-timers (Team Number #3)
Team Members: Hayung Suh, Nicolas Swanson, Satish Uppathil, Jacob Minson, Taeyeon Kim 

------------------------------------------------------------------------------------------

1. FOLDER STRUCTURE

team003final (root)
    \_ README.txt - a concise, short README.txt file, corresponding to the "user guide"
    \_ DOC - a folder with final documentations
        \_ team003report.pdf - report writeup in PDF format
        \_ team003poster.pdf - final poster
    \_ CODE - subset of files from Code folder that are required for running model
    \_ data - a folder with raw data files (457.4 MB) which are extracted from the 'All Data Files' zip file downloaded from College Scorecard (https://collegescorecard.ed.gov/data/) 
    \_ image - a folder with output image files of our model's performance 
    \_ dtype.txt - a text file used in the data cleaning script to set the datatypes of the columns

------------------------------------------------------------------------------------------

2. DESCRIPTION - Describe the package in a few paragraphs

**************************** NEED TO WRITE THIS SECTION ****************************

- data_ranked_list.ipynb is the file for running the model that preprocess College Scorecard's data 
and generates that ranked list of universities.
- collage_audit.py is the file that operates the interactive user interface for College Audit.

Make sure to set current working directory to the same level where this README.txt is located in.
For example: setwd("D:/CSE6242_Project/team003final") 



************************************************************************************
------------------------------------------------------------------------------------------

3. INSTALLATION - How to install and setup your code

**************************** NEED TO WRITE THIS SECTION ****************************
# INSTALL
# brew install libomp
# pip install lightgbm
# IMPORT
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import lightgbm as lgb




************************************************************************************

------------------------------------------------------------------------------------------

4. EXECUTION - How to run a demo on your code

**************************** NEED TO WRITE THIS SECTION ****************************

- Download all data files from college scorecard
- Extract the downloaded zip file and place the extracted folder ('/data') under team003final 
- According to the folder structure as mentioned above in 1. FOLDER STRUCTURE.
- Make sure to set current working directory to the same level where this README.txt is located.
    For example: setwd("D:/CSE6242_Project/team003final") 
- Run data_ranked_list.ipynb 
    (terminal: python ./CODE/data_ranked_list.py)
- After data_ranked_list.ipynb is finished running, run collage_audit.py 
    (terminal: python ./CODE/collage_audit.py)




************************************************************************************

------------------------------------------------------------------------------------------
