"""
The data come from the IPUMS/NHGIS site: data2.nhgis.org

Be sure to download time series data

The population age data are designated: "Persons by Sex [2] by Age [18]"

The family income are designated: "Families by Income in Previous Year [5]"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Family income data
pa1 = "nhgis0002_csv/nhgis0002_ts_nominal_tract.csv"

# Age x sex population counts
pa2 = "nhgis0001_csv/nhgis0001_ts_nominal_tract.csv"

# Household Type
pa3 = "nhgis0003_csv/nhgis0003_ts_geog2010_tract.csv"

# Read the raw income and population data
df_inc = pd.read_csv(pa1, encoding="latin1")
df_pop = pd.read_csv(pa2, encoding="latin1")
df_hous = pd.read_csv(pa3, encoding="latin1")

# The population data are counts in the following age bands
age_bands = ["<5", "5-9", "10-14", "15-17", "18-19", "20", "21", "22-24", "25-29",
             "30-34", "35-44", "45-54", "55-59", "60-61", "62-64", "65-74", "75-84",
             "85+"]

# Person household counts
house_bands = ["1", "2", "3", "4", "5", "6", "7+"]


# Year codes
years1 = [1970, 1980, 1990, 2000, 125]
years2 = [1970, 1980, 1990, 2000, 2010]
years3 = [1990, 2000, 2010]

# Normalize counts within each year by dividing each count by total count in
# each age group
def norm(df, stem, years):
    
    for y in years:
        # Extract Age and Income Band columns
        vx = [na for na in df.columns if str(y) in na and 
              na.startswith(stem) 
              and not na.endswith(("M", "U", "L"))]
        
        # check that we pulled the 5 income band variables, or the 36 
        # population variables: 18 age variables for men, and 18 for women
        # , 2 household types per year
        assert(len(vx) in [13, 5, 36])
        
        tot = df.loc[:, vx].sum(1)
        df.loc[:, vx] = df.loc[:, vx].div(tot, axis=0)


# Normalize the population and income data
stem1 = "A88"
stem2 = "B58"
stem3 = "CS2"

norm(df_inc, stem1, years1)
norm(df_pop, stem2, years2)
norm(df_hous, stem3, years3)

# The variable names corresponding to either income or population data
incvars = [c for c in df_inc.columns if c.startswith(stem1) and not c.endswith("M")]
popvars = [c for c in df_pop.columns if c.startswith(stem2)]
housvars = [c for c in df_hous.columns if c.startswith(stem3) and not c.endswith(("U", "L"))]

# We lose a lot of data here, but OK for our purposes
dx_inc = df_inc[['NHGISCODE'] + incvars].dropna()
dx_pop = df_pop[['NHGISCODE'] + popvars].dropna()
dx_hous = df_hous[['GISJOIN'] + housvars].dropna()
dx_hous.rename(columns={'GISJOIN': 'NHGISCODE'}, inplace=True)

#dx = pd.merge(dx_pop, dx_inc, left_on="NHGISCODE", right_on="NHGISCODE")
dx = pd.merge(dx_pop, dx_inc, left_on="NHGISCODE", right_on="NHGISCODE")
dx = pd.merge(dx, dx_hous, left_on="NHGISCODE", right_on="NHGISCODE")

# Alternative ordering for income variables
io = []
for year in 1970, 1980, 1990, 2000, 125:
    for vn in "AA", "AB", "AC", "AD", "AE":
        io.append("A88%s%d" % (vn, year))

# Add one set of income loadings to the current plot.
def plot_inc(vp, ylabel, title):

    xl = []
    for y in range(1970, 2011, 10):
        xl.extend(["", ""])
        xl.append(str(y))
        xl.extend(["", ""])

    plt.clf()
    plt.axes([0.15, 0.15, 0.75, 0.8])
    plt.grid(True)
    plt.title(title)
    for j in range(5):
        plt.plot(range(5*j, 5*(j+1)), vp[5*j:5*(j+1)], '-o', color='black')
    plt.gca().set_xticks(range(25))
    g = plt.gca().set_xticklabels(xl)
    for u in g:
        u.set_size(10)
        u.set_rotation(-90)
    plt.ylabel(ylabel, size=15)

# Add one set of population loadings to the current plot.
def plot_pop(vp, ylabel, title, ylim):

    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.axes([0.12, 0.15, 0.7, 0.8])
    plt.grid(True)
    m = vp.shape[0]
    plt.plot(vp.iloc[0:m//2], label="Female")
    plt.plot(vp.iloc[m//2:], label="Male")
    plt.title(title)
    g = plt.gca().set_xticklabels(age_bands + age_bands)
    for u in g:
        u.set_size(10)
        u.set_rotation(-90)
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, loc="center right")
    leg.draw_frame(False)
    plt.xlabel("Age", size=15)
    plt.ylabel(ylabel, size=15)
    if ylim is not None:
        plt.ylim(ylim)
     

'''
# Plot House Types, 3 decades, 2 types each
def plot_hous(vp, ylabel, title, ylim):

    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.axes([0.12, 0.15, 0.7, 0.8])
    plt.grid(True)
    # m = vp.shape[0]
    plt.plot(vp.iloc[0:6], label="Family")
    plt.plot(vp.iloc[6:], label="Non-Family")
    plt.title(title)
    g = plt.gca().set_xticklabels(house_bands[1:] + house_bands)
    for u in g:
        u.set_size(10)
        u.set_rotation(-90)
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, loc="center right")
    leg.draw_frame(False)
    plt.xlabel("House Type", size=15)
    plt.ylabel(ylabel, size=15)
    if ylim is not None:
        plt.ylim(ylim)
'''

def plot_hous(ax, vp, ylabel, title, ylim):

    
    ax.grid(True)
    # m = vp.shape[0]
    ax.plot(vp.iloc[0:6], label="Family")
    ax.plot(vp.iloc[6:], label="Non-Family")
    ax.title.set_text(title)
    ax.set_xticklabels(house_bands[1:] + house_bands)
    #for u in g:
     #   u.set_size(10)
      #  u.set_rotation(-90)
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, loc="center right")
    leg.draw_frame(False)
    ax.set_xlabel("House Type", size=15)
    ax.set_ylabel(ylabel, size=15)
    if ylim is not None:
        plt.ylim(ylim)
        
    
    
    
    
    
    
    
    
    













    