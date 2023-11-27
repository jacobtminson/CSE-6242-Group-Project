CSE 6242 Group Project
 
Project Title: College Audit 
Team Name: Full-timers (Team Number #3)
Team Members: Hayung Suh, Nicolas Swanson, Satish Uppathil, Jacob Minson, Taeyeon Kim 
 
------------------------------------------------------------------------------------------
 
1. FOLDER STRUCTURE
team003final (root; current directory) 
    \_ README.txt - a concise, short README.txt file, corresponding to the "user guide" 
    \_ DOC - a folder with final documentations (team003report.pdf, team003poster.pdf) 
    \_ CODE – a folder with subset of files that are required for running model and produced after executing model
        \_ data_ranked_list.py - main code file to run model 
        \_ college_audit_ui.py - main code file to run UI (run this file after executing data_ranked_list.py)
        \_ dtype.txt - a text file used in the data cleaning script to set the datatypes of the columns 
        \_ location_info.csv - location data of all schools for UI 
    \_ data - a folder with raw data files (457.4 MB) which are extracted from the 'All Data Files' zip file downloaded from College Scorecard (https://collegescorecard.ed.gov/data/) 
    \_ image - a folder with output image files of our model's performance (scatterplot.png)

------------------------------------------------------------------------------------------
 
2. DESCRIPTION – Description of the package (College Audit) in paragraphs
College Audit is an interactive visualization tool that offers personalized school recommendations based on user information and preferences. 
It analyzes large-scale data about colleges the United States government provides and recommends the top 10 colleges based on the user's filtering options. 
This process primarily consists of the three tasks below. 

- Data preprocessing: 
‘data_ranked_list.py’ is the file that pre-processes the raw datasets and operates the model for College Audit. 
The data is processed initially by combining each of the primary data sources (field of study and general information) into a year-over-year format, so that all information for a given dataset is combined. 
Next, the data columns that are more than 90% null are dropped, and a column for tracking the year of the data is created. After that, a few transformations on columns are completed (including joining redundant columns and doing calculations for totals of key columns).
The field of study data and general information were intentionally kept separate from one another, because combining them would have created a large amount of redundant information. 
Also joining the data would have significantly impacted the efficiency of the model. 
Finally, the location data file is created since much of the zip code data in the general information file was inaccurate and missing, and we needed accurate information for the map to function.

- Model:
With the preprocessed data, we predict the scores of each university which will be used to rank the schools and further apply personalized filtering. 
Specifically, the university weights are assigned by using various features that affect dropouts – graduation gate, net price, acceptance rate, salary, debt, size of school. 
Then using these calculated actual scores, we use gradient boosting to rank the universities according to their individual predicted rated scores. 
Before building the model, we tune the hyperparameter to get the optimal predicted scores for each university and improve performance using grid search and 10-fold cross validation. 
Finally, we train the model on training data (features) and make predictions on the testing data (actual score).
College Audit seeks to offer users an interactive experience, enabling them to explore numerous factors while also providing information about different majors and suitable school options that fit within their budgets. 
College Audit’s novelty will be focused on generating a universal ranked list of all universities from our dataset and then apply customized filters upon specific user’s information, rather than ranking universities again for every new user – for time and memory efficiency.

- UI:
'college_audit.py' is the file that operates the interactive user interface for College Audit. 
Upon execution, it generates a URL, which when accessed, opens the user interface in a local environment. 
This interface, organized into five sections, filtering options, the Top 10 Best Colleges list, Detailed Information, Historical trends of the best colleges, and Location information. 
After users input mandatory criteria like academic fields, state, and test scores (with additional optional criteria) and click the 'submit' button, The Top 10 Best College List showcases universities that meet the criteria. 
Users can find detailed descriptions of each option by hovering the mouse over the options. Once the list is available, users can simultaneously access various types of information about the best universities, tailored to their specific criteria. 
In Detailed Information, users can explore in-depth insights into each university’s ranking variables. Historical Trends offer data on admission rates, net prices, and debt by selecting a university from the list. 
The map section provides detailed geographical information about each listed university.

3. INSTALLATION - How to install and setup your code
This section guides you through the installation process of the College Audit. 
Follow these steps carefully and for additional help, watch our demo video: https://youtu.be/yo4UalQtiqY

- Code Installation 
    1) Download the Project Folder 
        • Download the file ‘team003final.zip’ from Canvas.
    2) Extract the Folder
        • Once downloaded, extract the ‘team003final.zip’ file, which will give a folder ‘\team003final’.

- Data Installation
    1)  Visit the College Scorecard website at https://collegescorecard.ed.gov/data/ and download ‘All Data Files’ package.
    2) Organize Data Files
       • Unzip ‘All Data Files’ and place the extracted folder directly under ‘\team003final’. 
       • Ensure the path for the data files is ‘\team003final\data’. Your ‘\team003final’ directory should contain three main items: ‘CODE’ folder, ‘data’ folder, ‘image’ folder, and ‘README.txt’.

- Setting up the Command Path
    1) Open a terminal window (e.g., Command Prompt, Terminal or similar)
    2) Navigate to the Project Directory
       • Change to the directory where you place ‘\team003final.’ 
       • For example, if the path to ‘\team003final’ is ‘C:\Users\team003final’, use the command: ‘cd C:\Users\team003final’. 
       • Additionally, many of the packages below will be used in the process of running the Python code, but the users don’t need to install them separately because the function to install the packages is included in our code. 

- Used Python Packages: matplotlib.pyplot, pandas, numpy, sklearn.metric, sklearn.model_selection, lightgbm, Dash, Plotly, math etc. 

------------------------------------------------------------------------------------------

4. EXECUTION - How to run a demo on your code
    1)	Make sure to set current working directory to the same level where this README.txt is located. 
        a.	For example: ‘pwd’ should give ‘C:\User\CSE6242\team003final’
    2)	Next, run ‘data_ranked_list.py’ for the model
        a.	For example, run on terminal: ‘python CODE\data_ranked_list.py’ 
        b.	Approximate runtime: 45-60 minutes
    3)	After ‘data_ranked_list.py’ finally executed, run ‘college_audit_dash.py’ for the UI
        a.	For example, run on terminal: ‘python CODE\ college_audit_dash.py’ 
    4)	Go to http://127.0.0.1:8050/ on internet server to display the UI dashboard. Make sure you are connected to the internet to operate the UI.

Our UI shows the information in a single page unified view of the results. 
Multiple sub-windows displaying various aspects of the results allows the user to simultaneously view details along with the big picture highlighting their strengths and weaknesses. 
The user inputs are taken in the ‘Filtering Options’ sidebar on the left. 
The inputs expected from the user are academic field (major), location, test scores, income, net price, debt, acceptance rate, school type, size, and women/men only. 
Detailed information on each input is displayed when the cursor is placed on each title (E.g., Academic Fields [Required]’).
Then, the top ten universities that best match the user's profile are displayed on the right with the list in the top right sub-window ‘Top 10 Universities for you’. 
The variables used to rank them and how each fare based on these metrics is shown in ‘Detail information for top 10’. 
The historical trend for any university in the top ten can be viewed by selecting one university. Finally, their locations are shown on a map.  

------------------------------------------------------------------------------------------

