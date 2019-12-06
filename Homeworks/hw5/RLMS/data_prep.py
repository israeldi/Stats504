"""
Download the individual-level data for the Russia Longitudinal Monitoring Survey:

https://dataverse.unc.edu/dataset.xhtml?persistentId=doi:10.15139/S3/12438

There are data files and codebook files, you will want one of each.
"""

import numpy as np
import pandas as pd

# Variables to keep

va = [
        "idind", # individual ID
        "J1", # current work status (1 = working)
        "age", # current age
        "year", # current year
        "J69_9C", # year of birth
        "educ", # education
        "status", # area type (city, etc.)
        "J8", # hours worked last 30 days
        "J10", # after tax wages
        "H5", # gender 1 = male, 2 = female
        "psu", # survey PSU and geographic area
        "OCCUP08", # occupation code
        "M71", # Smokes
        "M3", # Health Evaluation
        "M43", # Ever diagnosed with diabetes
        "M1", # Self Reported Weight
        "M2", # Self Reported Height
        "M20_66", # CHRONIC SPINAL DISEASE?
        "M20_65", # CHRONIC STOMACH DISEASE?
        "M20_64", # CHRONIC KIDNEY DISEASE?
        "M20_63", # CHRONIC LIVER DISEASE?
        "M20_62", # CHRONIC LUNG DISEASE?
        "M20_61", # CHRONIC HEART DISEASE?
        "M46", # EVER DIAGNOSED WITH HEART ATTACK?
        "marst", # Marital Status
     ]

# va = pd.read_csv('var_names.txt', sep=" ", header=None)
# va = list(va.iloc[:,0])

df = pd.read_csv("RLMS-HSE_IND_1994_2018_STATA.tab.gz", sep="\t", usecols=va)
# df = pd.read_csv("RLMS-HSE_IND_1994_2018_STATA.tab.gz", sep="\t")

NA_counts = df.isna().sum() / df.shape[0]
NA_counts.sort_values(inplace=True,ascending=False)
NA_counts2 = NA_counts[NA_counts < 0.5]
NA_counts2.index = map(str.upper, NA_counts2.index)

# Output variable names to use
'''
outFile = open('var_names.txt', 'w')
outFile.write("\n".join(str(item) for item in NA_counts2.index))
outFile.close()
'''

###################################################
m_cols = ["code", "description"]
db = pd.read_csv('RLMS_codebook.csv', sep=',', 
                 names=m_cols , encoding='latin-1')
db['code'] = db['code'].str.replace('.','_')

var_dict = dict(zip(db["code"], db["description"]))

descriptions = []
for var in NA_counts2.index:
    try:
        descriptions.append(var_dict[var])
    except:
        descriptions.append("Not found")
        
# Create Table of our variables and their descriptions
A = pd.DataFrame({"code": NA_counts2.index,
                  "description": descriptions,
                  "prct_missing": NA_counts2[0:]})
###################################################
        
# Drop people who are not working
df = df.loc[df.J1==1, :]

# Recode gender
df["Female"] = (df.H5 == 2).astype(np.int)

# Recode all other health variables
health_vars = ["M43", "M20_66", "M20_65", "M20_64", "M20_63", "M20_62", 
               "M71", "M20_61", "M46"]

for var in health_vars:
    df[var] = (df[var] == 1).astype(np.int)

# Drop the original versions of variables that we no longer need.
df = df.drop(columns=["J1", "H5"])

# Drop rows with missing values
dv = df.dropna()

# Center year at 2000, making it more centered
dv.year -= 2000

# Remove special codes
for v in ["J10", "J8", "educ", "age", "marst", 
"M71", "M3", "M43", "M1", "M2", "M20_66", "M20_65", 
"M20_64", "M20_63", "M20_62", "M20_61", "M46", "OCCUP08"]:
    ii = dv.loc[:, v] < 99999997
    dv = dv.loc[ii, :]
dv.dropna(inplace=True)

# Rename columns
dv.rename(columns={"J1": "work_status",
                   "J69_9C": "year_birth",
                   "J8": "hours_worked",
                   "J10": "after_tax_wages",
                   "M71": "smokes",
                   "M3": "health_eval",
                   "M43": "diabetes",
                   "M1": "weight",
                   "M2": "height",
                   "M20_66": "chr_spinal",
                   "M20_65": "chr_stomach",
                   "M20_64": "chr_kidney",
                   "M20_63": "chr_liver",
                   "M20_62": "chr_lung",
                   "M20_61": "chr_heart",
                   "M46": "heart_attack"
                   }, inplace=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
