
# I import and read the data
import py7zr
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing, feature_selection, ensemble, linear_model, decomposition
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

# Path to the .7z file
file_path = 'medical-sample.7z'

# Open the .7z file
with py7zr.SevenZipFile(file_path, mode='r') as z:
    all_files = z.getnames()
    csv_file_name = all_files[0]
    # Extract the CSV file to memory
    csv_file_dict = z.read([csv_file_name])
    csv_file = csv_file_dict[csv_file_name]

# Load the CSV data into a pandas DataFrame
medical_data_df = pd.read_csv(csv_file)


print(medical_data_df.shape)

# I check the amount of missing values in the data set
def display_missing_perc(medical_data_df):
    """
    This is a function that evaluates the percentage of NA values per column
    """
    for col in medical_data_df.columns.tolist():          
        missing_value = 100*(medical_data_df[col].isnull().sum()/len(medical_data_df[col]))
        missing_num = medical_data_df[col].isnull().sum()
        print(f'{col} column percentage of missing values: {missing_value} ; total missing: {missing_num}') # Here, I can also see the total number of missing values.
    print('\n')
display_missing_perc(medical_data_df)

#I remove any unnamed columns
medical_data_df = medical_data_df.loc[:, ~medical_data_df.columns.str.contains('^Unnamed')]
medical_data_df.dtypes

#This is function to visualize the NA values.
#it is a heatmap to see how many NA values per column.
def utils_recognize_type(dtf, col, max_cat=20):
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"
dic_cols = {col:utils_recognize_type(medical_data_df, col, max_cat=20) for col in medical_data_df.columns}
heatmap = medical_data_df.isnull()
for k,v in dic_cols.items():
    if v == "num":
        heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
    else:
        heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
sns.heatmap(heatmap, cbar=False).set_title('Dataset Overview')
plt.show()
print("\033[1;37;40m Categorical ", "\033[1;30;41m Numeric ", "\033[1;30;47m NaN ")

# Upload the scripts into a data frame
file_path = 'scripts-sample.7z'

# Open the .7z file
with py7zr.SevenZipFile(file_path, mode='r') as z:
    all_files = z.getnames()
    # Assuming there's only one CSV file in the .7z archive
    csv_file_name = all_files[0]
    # Extract the CSV file to memory
    csv_file_dict = z.read([csv_file_name])
    csv_file = csv_file_dict[csv_file_name]

# Load the CSV data into a pandas DataFrame
scripts_data_df = pd.read_csv(csv_file)

# I check the amount of missing values in the data set
def display_missing_perc(scripts_data_df):
    """
    This is a function that evaluates the percentage of NA values per column
    """
    for col in scripts_data_df.columns.tolist():          
        missing_value = 100*(scripts_data_df[col].isnull().sum()/len(scripts_data_df[col]))
        missing_num = scripts_data_df[col].isnull().sum()
        print(f'{col} column percentage of missing values: {missing_value} ; total missing: {missing_num}') # Here, I can also see the total number of missing values.
    print('\n')
display_missing_perc(scripts_data_df)

def utils_recognize_type(dtf, col, max_cat=20):
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"
dic_cols = {col:utils_recognize_type(scripts_data_df, col, max_cat=20) for col in scripts_data_df.columns}
heatmap = scripts_data_df.isnull()
for k,v in dic_cols.items():
    if v == "num":
        heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
    else:
        heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
sns.heatmap(heatmap, cbar=False).set_title('Dataset Overview')
plt.show()
print("\033[1;37;40m Categorical ", "\033[1;30;41m Numeric ", "\033[1;30;47m NaN ")

# I explore the patients age distribution
age = medical_data_df['patient_age'].value_counts()

# Create a bar plot
age.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Patients Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotates X-axis labels to horizontal
plt.show()

# I explore the patients gender distribution
gender = medical_data_df['patient_gender'].value_counts()

# Create a bar plot
gender.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Frequency of Male and Female patients')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotates X-axis labels to horizontal
plt.show()

# I explore the patients visit type: 
visit_type_df = medical_data_df['visit_type']

visit_type_counts = medical_data_df['visit_type'].value_counts()

# Using seaborn to create the bar plot for a nicer default style
sns.barplot(x=visit_type_counts.index, y=visit_type_counts.values)
plt.xticks(rotation=80)  # Rotate labels to make them readable if they're long
plt.xlabel('Visit Type')
plt.ylabel('Frequency')
plt.title('Frequency of Patient Visit Types')
plt.show()


# Explore the most common procedures: 

proc_code_counts = medical_data_df['proc_code'].value_counts()

# Directly obtain the top 10 most common procedures
top_10_proc_codes = proc_code_counts.head(10)

# Since proc_code_counts is a Series, keys are the index and values are the Series' values
keys = top_10_proc_codes.index
values = top_10_proc_codes.values

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(keys, values)
plt.xlabel('Procedure Code')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate labels to make them readable
plt.title('Top 10 Procedures by Count')
plt.show()

# Check if the code is in the dataset and find its rank
proc_code = 'J3490'

if proc_code in proc_code_counts.index:
    rank = proc_code_counts.index.get_loc(proc_code) + 1  # Adding 1 because index is 0-based
    count = proc_code_counts[proc_code]
    print(f"Procedure code {proc_code} is ranked {rank} with a count of {count}.")

# 1. Select 'diag_' columns
diag_columns = [col for col in medical_data_df.columns if col.startswith('diag_')]
df_diag = medical_data_df[diag_columns]

# 2. Combine values 
diag_list = df_diag.values.flatten().tolist()
diag_list = [x for x in diag_list if pd.notna(x)]

# 3. Process first three digits (same as before)
first_three_counts = {}
for diag_code in diag_list:
    first_three = diag_code[:3]
    first_three_counts[first_three] = first_three_counts.get(first_three, 0) + 1

import operator
# Sort the dictionary items by count (descending order)
sorted_counts = sorted(first_three_counts.items(), key=operator.itemgetter(1), reverse=True)

for items in sorted_counts:
    print(items)

# I explore the column of date_claim
# Ensure 'claim_date' is in datetime format
claim_date_df = medical_data_df['claim_date']
claim_date_df.head()

# Get the first 10 items
top_10_items = sorted_counts[:10]  

# Unpack the items into keys and values
keys, values = zip(*top_10_items)

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(keys, values)
plt.xlabel('Diagnosis Code') 
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate labels to make them readable
plt.title('Top 10 diagnosis by Count')
plt.show()

#Ozempic has code J3490
filtered_df = medical_data_df[medical_data_df['proc_code'] == "J3490"]

# 1. Select 'diag_' columns
diag_columns = [col for col in filtered_df.columns if col.startswith('diag_')]
df_diag = filtered_df[diag_columns]

# 2. Combine values 
combined_list = df_diag.values.flatten().tolist()
combined_list = [x for x in combined_list if pd.notna(x)]

# 3. Process first three digits (same as before)
first_three_counts = {}
for diag_code in combined_list:
    first_three = diag_code[:3]
    first_three_counts[first_three] = first_three_counts.get(first_three, 0) + 1

sorted_counts = sorted(first_three_counts.items(), key=operator.itemgetter(1), reverse=True)

#for items in sorted_counts:
#    print(items)

# Get the first 15 diagnoses
first_15_diagnoses = sorted_counts[:15]  

# Unpack the items into keys and values
keys, values = zip(*first_15_diagnoses)

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(keys, values)
plt.xlabel('Diagnosis Code') 
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate labels to make them readable
plt.title('Top 15 diagnosis for the use of Ozempic')
plt.show()

# Create a copy of the DataFrame when filtering to avoid SettingWithCopyWarning
filtered_df = medical_data_df[medical_data_df['proc_code'] == "J3490"].copy()

# Convert 'claim_date' to datetime format
filtered_df['claim_date'] = pd.to_datetime(filtered_df['claim_date'])


# Find the earliest 'claim_date' for the filtered rows
earliest_date = filtered_df['claim_date'].min()

# Display the earliest date
if pd.notnull(earliest_date):
    print(f"The earliest claim date for procedure code 'J3490' is: {earliest_date.strftime('%Y-%m-%d')}")
else:
    print("No claims match the specified criteria.")




# Aggregate data by 'claim_date'
prescription_counts = filtered_df.groupby('claim_date').size()

# Plot the time series
plt.figure(figsize=(12, 6))  # Set the figure size for better readability
prescription_counts.plot(kind='line', color='blue', marker='o', linestyle='-')
plt.title('Ozempic Prescriptions Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Prescriptions')
plt.grid(True)
plt.show()


# Step 1: Create a new DataFrame with each 'diag_' entry and its 'claim_date'
diag_long_df = filtered_df.melt(id_vars=['claim_date'], value_vars=diag_columns, var_name='DiagnosisType', value_name='DiagnosisCode')

# Remove rows with NaN diagnosis codes
diag_long_df = diag_long_df.dropna(subset=['DiagnosisCode'])

# Filter for rows where DiagnosisCode starts with 'E66'
e66_df = diag_long_df[diag_long_df['DiagnosisCode'].str.startswith('E66')]

# Convert 'claim_date' to datetime format, if not already
e66_df['claim_date'] = pd.to_datetime(e66_df['claim_date'])

# Aggregate the data by 'claim_date'
prescriptions_per_date = e66_df.groupby('claim_date').size()
# Example of checking the aggregated data
pd.set_option('display.max_rows', None)  # Set to None to display all rows
print(prescriptions_per_date)

# Plot the time series
plt.figure(figsize=(12, 6))  
prescriptions_per_date.plot(kind='line', color='blue', marker='o', linestyle='-')
plt.title('Ozempic Prescriptions for Obesity (E66) Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Prescriptions')
plt.grid(True)
plt.show()


# Resample the data to fill missing dates with zeros
prescriptions_per_date_filled = prescriptions_per_date.resample('D').asfreq().fillna(0)

# Plot the filled time series
plt.figure(figsize=(12, 6))
prescriptions_per_date_filled.plot(kind='line', color='blue', marker='o', linestyle='-')
plt.title('Ozempic Prescriptions for Diabetes (E11) Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Prescriptions')
plt.grid(True)
plt.show()


# Load the data
multiTimeline_df = pd.read_csv('/Users/Gustavo/Library/CloudStorage/Dropbox/UC Davis/2024 - Winter/ML/Assignments ML/HW4/A5/Ozempic_All_Time/multiTimeline.csv', skiprows=1)  # Adjust skiprows if needed

# Rename columns for clarity
multiTimeline_df.rename(columns={'Month': 'date', 'Ozempic: (United States)': 'search_interest'}, inplace=True)

# Convert 'date' to datetime and 'search_interest' to numeric, treating '<1' as a small number like 0.5
multiTimeline_df['date'] = pd.to_datetime(multiTimeline_df['date'])
multiTimeline_df['search_interest'] = pd.to_numeric(multiTimeline_df['search_interest'].str.replace('<1', '0.5'), errors='coerce')

# Plot the search interest data
plt.figure(figsize=(14, 7))
plt.plot(multiTimeline_df['date'], multiTimeline_df['search_interest'], color='green', marker='o', linestyle='-')
plt.title('Ozempic Search Interest Over Time')
plt.xlabel('Date')
plt.ylabel('Relative Search Interest')
plt.grid(True)
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

related_queries_df = pd.read_csv('//Users/Gustavo/Library/CloudStorage/Dropbox/UC Davis/2024 - Winter/ML/Assignments ML/HW4/A5/Ozempic_All_Time/relatedQueries.csv', skiprows=4, names=['query', 'score']) 

top_queries = related_queries_df.head(25)

# Convert the 'score' column to numeric, dropping non-numeric rows to avoid 'Breakout' values causing issues
top_queries['score'] = pd.to_numeric(top_queries['score'], errors='coerce')

# Now, sort the data by score for better visualization
top_queries_sorted = top_queries.sort_values(by='score', ascending=False)

# Create a bar plot for the top related queries
inverted_greens = sns.color_palette("Greens_r") 
plt.figure(figsize=(14, 10))
sns.barplot(x='score', y='query', data=top_queries_sorted, palette=inverted_greens)
plt.title('Top Related Queries for Ozempic')
plt.xlabel('Relative Popularity Score')
plt.ylabel('Search Query')
plt.show()



multiTimeline_df = pd.read_csv('/Users/Gustavo/Library/CloudStorage/Dropbox/UC Davis/2024 - Winter/ML/Assignments ML/HW4/A5/Ozempic_start_2017/geoMap.csv', skiprows=1) 

# 1. Select 'diag_' columns
#diag_columns = [col for col in df.columns if col.startswith('diag_')]
df_zip = medical_data_df['patient_short_zip']

# 2. Combine values 
combined_list = df_zip.values.flatten().tolist()
combined_list = [x for x in combined_list if pd.notna(x)]

# 3. Process first three digits (same as before)
zips = {}
for diag_code in combined_list:
    #first_three = diag_code[:3]
    zips[diag_code] = zips.get(diag_code, 0) + 1
sorted_counts = sorted(zips.items(), key=operator.itemgetter(1), reverse=True)

for items in sorted_counts:
    print(items)

# Create a copy of the DataFrame
medical_data_df_copy = medical_data_df.copy()

# Merging of the diag_ columns on the copy
diag_columns = [col for col in medical_data_df_copy.columns if col.startswith('diag_')]
df_diag = medical_data_df_copy[diag_columns]

# Merge the diag_ columns into a single column as before
medical_data_df_copy['combined_diag'] = df_diag.apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

medical_data_df_copy['has_E11'] = medical_data_df_copy['combined_diag'].str.contains('E11')
medical_data_df_copy['treated_with_J3490'] = medical_data_df_copy['proc_code'] == 'J3490'

# I filter DataFrame for patients with E11 
patients_with_E11 = medical_data_df_copy[medical_data_df_copy['has_E11']]

# I group by 'claim_date' and 'treated_with_J3490', then count unique 'journey_id' (hospital visits)
visits_by_date_E11 = patients_with_E11.groupby(['claim_date', 'treated_with_J3490']).agg({'journey_id': pd.Series.nunique}).reset_index()

# I Rename columns for clarity
visits_by_date_E11.rename(columns={'journey_id': 'hospital_visits'}, inplace=True)

# I convert 'claim_date' to datetime format for plotting
visits_by_date_E11['claim_date'] = pd.to_datetime(visits_by_date_E11['claim_date'])

# Aggregate data monthly
visits_by_month_E11 = visits_by_date_E11.set_index('claim_date').groupby([pd.Grouper(freq='M'), 'treated_with_J3490']).sum().reset_index()
# Ensure correct aggregation by 'claim_date' and 'treated_with_J3490'
visits_by_month_E11 = visits_by_date_E11.copy()

# Re-checking the aggregation logic for monthly data
visits_by_month_E11 = visits_by_month_E11.groupby(['claim_date', 'treated_with_J3490'])['hospital_visits'].sum().reset_index()

# Convert 'claim_date' to datetime
visits_by_month_E11['claim_date'] = pd.to_datetime(visits_by_month_E11['claim_date'])

# Ensure correct application of the rolling average within each treatment group
visits_by_month_E11['hospital_visits_smooth'] = visits_by_month_E11.groupby('treated_with_J3490')['hospital_visits'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Separate plots for clarity
for treatment_status in visits_by_month_E11['treated_with_J3490'].unique():
    subset = visits_by_month_E11[visits_by_month_E11['treated_with_J3490'] == treatment_status]
    
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=subset, x='claim_date', y='hospital_visits_smooth', label=('Treated' if treatment_status else 'Not Treated'))
    plt.title(f"Hospital Visits Over Time for Diabetes Patients {'Treated' if treatment_status else 'Not Treated'} with J3490 (Monthly, Smoothed)")
    plt.xlabel('Month')
    plt.ylabel('Number of Hospital Visits (3-month Rolling Average)')
    plt.grid(True)
    plt.show()


# Filter for 'E66' diagnoses in the copied DataFrame
medical_data_df_copy['has_E66'] = medical_data_df_copy['combined_diag'].str.contains('E66')

# Group by 'journey_id' and determine the number of unique patients with 'E66'
e66_patient_counts = medical_data_df_copy[medical_data_df_copy['has_E66']].groupby('journey_id').size()
print(f"Number of unique patients with 'E66' diagnosis: {e66_patient_counts.count()}")


medical_data_df_copy['treated_with_J3490'] = medical_data_df_copy['proc_code'] == 'J3490'

# Filter DataFrame for patients with E11
patients_with_E66 = medical_data_df_copy[medical_data_df_copy['has_E66']]

# Step 2: Group by 'claim_date' and 'treated_with_J3490', then count unique 'journey_id' (hospital visits)
visits_by_date_E66 = patients_with_E66.groupby(['claim_date', 'treated_with_J3490']).agg({'journey_id': pd.Series.nunique}).reset_index()

# Rename columns for clarity
visits_by_date_E66.rename(columns={'journey_id': 'hospital_visits'}, inplace=True)

# Convert 'claim_date' to datetime format for plotting
visits_by_date_E66['claim_date'] = pd.to_datetime(visits_by_date_E66['claim_date'])

# Aggregate data monthly
visits_by_month_E66 = visits_by_date_E66.set_index('claim_date').groupby([pd.Grouper(freq='M'), 'treated_with_J3490']).sum().reset_index()
# Ensure correct aggregation by 'claim_date' and 'treated_with_J3490'
visits_by_month_E66 = visits_by_date_E66.copy()

# Re-checking the aggregation logic for monthly data
visits_by_month_E66 = visits_by_month_E66.groupby(['claim_date', 'treated_with_J3490'])['hospital_visits'].sum().reset_index()

# Convert 'claim_date' to datetime (if not already done)
visits_by_month_E66['claim_date'] = pd.to_datetime(visits_by_month_E66['claim_date'])

# Ensure correct application of the rolling average within each treatment group
visits_by_month_E66['hospital_visits_smooth'] = visits_by_month_E66.groupby('treated_with_J3490')['hospital_visits'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Separate plots for clarity
for treatment_status in visits_by_month_E66['treated_with_J3490'].unique():
    subset = visits_by_month_E66[visits_by_month_E66['treated_with_J3490'] == treatment_status]
    
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=subset, x='claim_date', y='hospital_visits_smooth', label=('Treated' if treatment_status else 'Not Treated'))
    plt.title(f"Hospital Visits Over Time for Obesity Patients {'Treated' if treatment_status else 'Not Treated'} with J3490 (Monthly, Smoothed)")
    plt.xlabel('Month')
    plt.ylabel('Number of Hospital Visits (3-month Rolling Average)')
    plt.grid(True)
    plt.show()

new_medical_df = medical_data_df_copy.drop(['ref_npi','hcp_npi','hcp_taxonomy','hcp_specialty', 'hcp_npi_list', 'rev_center_code', 'proc_modifier','diag_1', 'diag_2', 'diag_3', 'diag_4', 'diag_5'], axis=1)
# Display all the numeric columns (both float and int64):
#medical_data_df.select_dtypes(include=['float', 'int64'])

# I check the amount of missing values in the data set
def display_missing_perc(new_medical_df):
    """
    This is a function that evaluates the percentage of NA values per column
    """
    for col in new_medical_df.columns.tolist():          
        missing_value = 100*(new_medical_df[col].isnull().sum()/len(new_medical_df[col]))
        missing_num = new_medical_df[col].isnull().sum()
        print(f'{col} column percentage of missing values: {missing_value} ; total missing: {missing_num}') # Here, I can also see the total number of missing values.
    print('\n')
display_missing_perc(new_medical_df)

print(new_medical_df.dtypes)

def fill_na_with_mode(new_medical_df, columns):
    """
    Fill NaN values for specified columns with the mode (most common value) of each column.
    """
    for col in columns:
        if new_medical_df[col].dtype == 'object':  # Ensure the column is of object type
            mode_value = new_medical_df[col].mode()[0]  # Get the mode of the column, which is the most common value
            new_medical_df[col].fillna(mode_value, inplace=True)  # Fill NaN values with the mode
        else:
            print(f"Column {col} is not of object type and will not be processed.")
    return new_medical_df

# List of columns to fill NaN values with their mode
columns_to_fill = ['visit_id', 'patient_gender', 'visit_type', 'payor_channel', 'diag_list', 'proc_code', 'hco_npi_list']

# Apply the function to  DataFrame
new_medical_df = fill_na_with_mode(new_medical_df, columns_to_fill)


def fill_na_with_median(new_medical_df, columns):
    """
    Fill NaN values for specified numerical columns with the median of each column.
    """
    for col in columns:
        if new_medical_df[col].dtype in ['float64', 'int64']:  # Check if the column is numerical
            median_value = new_medical_df[col].median()  # Calculate the median of the column
            new_medical_df[col].fillna(median_value, inplace=True)  # Fill NaN values with the median
        else:
            print(f"Column {col} is not of numerical type and will not be processed.")
    return new_medical_df

# List of numerical columns to fill NaN values with their median
numerical_columns_to_fill = ['patient_short_zip', 'patient_age', 'rev_center_units', 'proc_units', 'hco_npi']

# Apply the function to DataFrame
new_medical_df = fill_na_with_median(new_medical_df, numerical_columns_to_fill)


# I create a Series with the mode of 'place_of_service' for each 'zip'
mode_place_of_service_by_zip = new_medical_df.groupby('patient_short_zip')['place_of_service'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)

# I fill NaN values in 'place_of_service' with this mode series
new_medical_df['place_of_service'] = new_medical_df['place_of_service'].fillna(mode_place_of_service_by_zip)
# For any missing value remaining, I calculate the global mode for the 'place_of_service' column
global_mode_place_of_service = new_medical_df['place_of_service'].mode()[0]

# Then, I check if there are still missing values in 'place_of_service' and fill them with the global mode
new_medical_df['place_of_service'].fillna(global_mode_place_of_service, inplace=True)

# I create a Series with the mode of 'payor' for each 'zip'
mode_payor_by_zip = new_medical_df.groupby('patient_short_zip')['payor'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)

# I fill NaN values in 'payor' with this mode series
new_medical_df['payor'] = new_medical_df['payor'].fillna(mode_payor_by_zip)

# For any missing value remaining, I calculate the global mode for the 'payor' column
global_mode_payor = new_medical_df['payor'].mode()[0]

# Then, I check if there are still missing values in 'payor' and fill them with the global mode
new_medical_df['payor'].fillna(global_mode_payor, inplace=True)


display_missing_perc(new_medical_df)

# encoding object columns with label encoding
from sklearn.preprocessing import LabelEncoder
cols = ['episode_id', 'patient_gender','visit_id', 'encounter_id', 'claim_date','patient_state', 'place_of_service', 'visit_type', 'payor', 'payor_channel', 'diag_list', 'proc_code', 'combined_diag', 'hco_npi_list']
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(new_medical_df[c].values)) #important: with this it will learn the number inside the column. 
    new_medical_df[c] = lbl.transform(list(new_medical_df[c].values))

# shape        
print('Shape all_data: {}'.format(new_medical_df.shape))

# I convert bool columns
columns_to_convert = ['has_E11', 'treated_with_J3490', 'has_E66']
for column in columns_to_convert:
    new_medical_df[column] = new_medical_df[column].astype(int)
    
print(new_medical_df['has_E11'].value_counts())
print(new_medical_df['treated_with_J3490'].value_counts())
print(new_medical_df['has_E66'].value_counts())

new_scripts_df = scripts_data_df.drop(['patient_gender','patient_state','pharmacist_npi','primary_care_npi', 'ndc11_original', 'diagnosis_code', 'diagnosis_code_type', 'level_of_service', 'daw_code','prior_authorization_type_code', 'coupon_type', 'coupon_value_amount','reject_code_1', 'reject_code_3', 'reject_code_4','reject_code_5', 'end_date','group_id','quantity_prescribed_original','place_of_service', 'copay_coinsurance', 'plan_pay', 'is_service', 'unit_of_measure'], axis=1)

def display_missing_perc(new_scripts_df):
    """
    This is a function that evaluates the percentage of NA values per column
    """
    for col in new_scripts_df.columns.tolist():          
        missing_value = 100*(new_scripts_df[col].isnull().sum()/len(new_scripts_df[col]))
        missing_num = new_scripts_df[col].isnull().sum()
        print(f'{col} column percentage of missing values: {missing_value} ; total missing: {missing_num}') # Here, I can also see the total number of missing values.
    print('\n')
display_missing_perc(new_scripts_df)

print(new_scripts_df.dtypes)

def fill_na_with_mode(new_scripts_df, columns):
    """
    Fill NaN values for specified columns with the mode (most common value) of each column.
    """
    for col in columns:
        if new_scripts_df[col].dtype == 'object':  # Ensure the column is of object type
            mode_value = new_scripts_df[col].mode()[0]  # Get the mode of the column, which is the most common value
            new_scripts_df[col].fillna(mode_value, inplace=True)  # Fill NaN values with the mode
        else:
            print(f"Column {col} is not of object type and will not be processed.")
    return new_scripts_df

# List of columns to fill NaN values with their mode
columns_to_fill = ['patient_dob', 'prescriber_npi', 'date_authorized', 'is_compound_drug','pcn']

# Apply the function to your DataFrame
new_scripts_df = fill_na_with_mode(new_scripts_df, columns_to_fill)

# fill out missing numerical values with median
def fill_na_with_median(new_scripts_df, columns):
    """
    Fill NaN values for specified numerical columns with the median of each column.
    """
    for col in columns:
        if new_scripts_df[col].dtype in ['float64', 'int64']:  # Check if the column is numerical
            median_value = new_scripts_df[col].median()  # Calculate the median of the column
            new_scripts_df[col].fillna(median_value, inplace=True)  # Fill NaN values with the median
        else:
            print(f"Column {col} is not of numerical type and will not be processed.")
    return new_scripts_df

# List of numerical columns to fill NaN values with their median
numerical_columns_to_fill = ['patient_zip', 'patient_zip', 'number_of_refills_authorized', 'quantity_dispensed', 'pharmacy_submitted_cost', 'bin']

# Apply the function to your DataFrame
new_scripts_df = fill_na_with_median(new_scripts_df, numerical_columns_to_fill)

# I create a Series with the mode of 'patient_pay' for each 'zip'
mode_patient_pay_by_zip = new_scripts_df.groupby('patient_zip')['patient_pay'].transform(lambda x: x.median() if not x.empty else np.nan)

# I fill NaN values in 'patient_pay' with this mode series
new_scripts_df['patient_pay'] = new_scripts_df['patient_pay'].fillna(mode_patient_pay_by_zip)

# For any missing value remaining, I calculate the global median for the 'patient_pay' column
global_mode_patient_pay = new_scripts_df['patient_pay'].median()

# Then, I check if there are still missing values in 'patient_pay' and fill them with the global median
new_scripts_df['patient_pay'].fillna(global_mode_patient_pay, inplace=True)


# I create a Series with the mode of 'pharmacy_npi' for each 'zip'
mode_pharmacy_npi_by_zip = new_scripts_df.groupby('patient_zip')['pharmacy_npi'].transform(lambda x: x.median() if not x.empty else np.nan)

# I fill NaN values in 'pharmacy_npi' with this mode series
new_scripts_df['pharmacy_npi'] = new_scripts_df['pharmacy_npi'].fillna(mode_pharmacy_npi_by_zip)

# For any missing value remaining, I calculate the global median for the 'pharmacy_npi' column
global_mode_pharmacy_npi = new_scripts_df['pharmacy_npi'].median()

# Then, I check if there are still missing values in 'pharmacy_npi' and fill them with the global median
new_scripts_df['pharmacy_npi'].fillna(global_mode_pharmacy_npi, inplace=True)


display_missing_perc(new_scripts_df)

print(new_scripts_df.dtypes)

# encoding object columns with label encoding
from sklearn.preprocessing import LabelEncoder
cols = ['claim_id', 'patient_dob','prescriber_npi', 'date_of_service', 'date_authorized', 'transaction_type', 'date_prescription_written','pcn']
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(new_scripts_df[c].values)) #important: with this it will learn the number inside the column. 
    new_scripts_df[c] = lbl.transform(list(new_scripts_df[c].values))

# shape        
print('Shape all_data: {}'.format(new_scripts_df.shape))

# encoding bool column
new_scripts_df['is_compound_drug'] = new_scripts_df['is_compound_drug'].astype(int)
new_scripts_df['active'] = new_scripts_df['active'].astype(int)

merged_df = new_medical_df.merge(new_scripts_df, on='journey_id', how='inner')
print(merged_df.shape)

# Now i can encode 'journey_id' into 'journey_id_encoded' as its numerical representation.
# Initialize the LabelEncoder
lbl = LabelEncoder()

# Fit and transform the 'journey_id' column and assign it back to the DataFrame
merged_df['journey_id_encoded'] = lbl.fit_transform(merged_df['journey_id'])

print(merged_df.shape)


merged_df.drop('journey_id', axis=1, inplace=True)
print(merged_df.dtypes)

# Count the number of visits per journey_id
visit_counts = merged_df.groupby('journey_id_encoded')['visit_id'].nunique().reset_index(name='visit_count')

# Merge the visit counts with the original dataset to get the features for each individual
medical_visits_df = pd.merge(merged_df, visit_counts, on='journey_id_encoded', how='left')

# Drop duplicates to ensure each individual is represented once
df_unique = medical_visits_df.drop_duplicates(subset=['journey_id_encoded'])

X = df_unique[['has_E11', 'has_E66']]
y = df_unique['visit_count']


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

# Preparing the dataset for train-test split
X = df_unique.drop(columns=['journey_id_encoded', 'visit_count', 'treated_with_J3490']) # Drop the identifiers, outcome, and treatment indicator
y = df_unique['treated_with_J3490']   # Treatment indicator


# Performing the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit the LassoCV model
lasso_cv = LassoCV(alphas=None, cv=5, max_iter=10000, random_state=0)

# Fit the model on the scaled training data
lasso_cv.fit(X_train_scaled, y_train)

# Predict on training and testing data
train_preds_lasso = lasso_cv.predict(X_train_scaled)
test_preds_lasso = lasso_cv.predict(X_test_scaled)

# Best alpha
print(f"Best alpha: {lasso_cv.alpha_}")

# Coefficients
print(f"Coefficients: {lasso_cv.coef_}")


# Create a dictionary of variable names and their corresponding coefficients
coef_dict = {variable: coef for variable, coef in zip(X.columns, lasso_cv.coef_)}

# Print the variables with non-zero coefficients
for variable, coef in coef_dict.items():
    if coef != 0:
        print(f"{variable}: {coef}")


# Add the predicted probabilities back to the original dataset
df_unique['predicted_treatment'] = np.concatenate([train_preds_lasso, test_preds_lasso])

class SelectiveRegularizationLinearRegression:
    def __init__(self, alpha, apply_penalty, tolerance=1e-4, max_iterations=1000):
        self.alpha = alpha
        self.tolerance = tolerance
        self.apply_penalty = apply_penalty
        self.max_iterations = max_iterations
        self.w = None
        
    def _predicted_values(self, X, w):
        return np.matmul(X, w)

    def _rho_compute(self, y, X, w, j): #we evaluate covariate individually to see how much it adds to our prediction
        X_k = np.delete(X, j, 1)
        w_k = np.delete(w, j)
        predict_k = self._predicted_values(X_k, w_k)
        residual = y - predict_k
        rho_j = np.sum(X[:, j] * residual)
        return rho_j

    def _z_compute(self, X):
        return np.sum(X * X, axis=0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        if not self.w:
            self.w = np.zeros(n_features)
        z = self._z_compute(X)
        for iteration in range(self.max_iterations):
            max_step = 0
            for j in range(n_features):
                rho_j = self._rho_compute(y, X, self.w, j) # how much this particular covariate adds to the overall prediction
                w_j_old = self.w[j]
                if j == 0:
                    self.w[j] = rho_j / z[j]
                else:
                    if self.apply_penalty[j]:
                        if rho_j < -self.alpha * n_samples:
                            self.w[j] = (rho_j + self.alpha * n_samples) / z[j]
                        elif -self.alpha * n_samples <= rho_j <= self.alpha * n_samples:
                            self.w[j] = 0.
                        elif rho_j > self.alpha * n_samples:
                            self.w[j] = (rho_j - self.alpha * n_samples) / z[j]
                    else:
                        self.w[j] = rho_j / z[j]
                max_step = max(max_step, abs(self.w[j] - w_j_old))
            if max_step < self.tolerance:
                break
        return self

    def predict(self, X):
        if self.w is None:
            raise ValueError("Model is not fitted yet!")
        return self._predicted_values(X, self.w)

# Selecting independent variables and the dependent variable
X = df_unique[['treated_with_J3490', 'predicted_treatment', 'diag_list', 'proc_code', 'smart_allowed', 'claim_id', 'days_supply' ]].values  
y = df_unique['visit_count'].values

n_samples = len(X)
X = np.hstack((np.ones((n_samples, 1)), X))


apply_penalty = np.array([False, False, False, True, True, True, True, True]) #apply a penalty term to the last coefficients
alpha = 0.1  # L1 penalty term
max_iterations = 1000  


model = SelectiveRegularizationLinearRegression(alpha, apply_penalty, max_iterations)
model.fit(X, y)  # Fit the model

print("Learned coefficients:", model.w)  # Print out model coefficients


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# Removing constant features before the cross-validation loop
selector = VarianceThreshold(threshold=0)
X_new = selector.fit_transform(X)

# I use X_new in the cross-validation loop
kf = KFold(n_splits=5, shuffle=True, random_state=0)
rmse_list = []
r_squared_list = []
for train_index, test_index in kf.split(X_new):
    X_train, X_test = X_new[train_index], X_new[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SelectiveRegularizationLinearRegression(alpha, apply_penalty, max_iterations)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r_squared = r2_score(y_test, y_pred)
    r_squared_list.append(r_squared)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_list.append(rmse)

#Print average performance across folds
print(f"Average Model Performance:  RMSE: {np.mean(rmse_list)}")

from sklearn.linear_model import Lasso
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  

alphas = np.logspace(-6, 6, 50)
aicc_dic = {}
bic_dic = {}

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X, y)  # Use second-stage predictors and outcome variable

    # Make predictions on the data
    y_pred = lasso.predict(X)

    rss = np.sum((y - y_pred) ** 2)
    n = len(y)
    k = np.sum(lasso.coef_ != 0) + 1  # Number of non-zero parameters including the intercept

    # Calculate AICc
    aicc = n * np.log(rss / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)
    aicc_dic[alpha] = aicc
    
    # Calculate BIC
    bic = n * np.log(rss / n) + k * np.log(n)
    bic_dic[alpha] = bic

# Get the alpha that gives the minimum AICc and BIC
min_aicc_key = min(aicc_dic, key=aicc_dic.get)
min_bic_key = min(bic_dic, key=bic_dic.get)

print(f'Alpha with the lowest AICc: {min_aicc_key}, AICc: {aicc_dic[min_aicc_key]}')
print(f'Alpha with the lowest BIC: {min_bic_key}, BIC: {bic_dic[min_bic_key]}')

from sklearn.linear_model import LassoLarsIC

# Fit the LassoLarsIC model with AIC criterion
model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X_scaled, y)

# Retrieve the AIC value
aic = model_aic.criterion_

# The alpha value chosen
alpha_aic = model_aic.alpha_

print(f"AIC: {aic}")
print(f"Alpha: {alpha_aic}")



