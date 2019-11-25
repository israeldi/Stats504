#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:34:59 2019

@author:    israeldiego
Course:     Stats 504: HW1

Description: Conduct an analysis of BMI variation in the US population 
using the NHANES data. You should fit and interpret a regression model 
in which BMI is the dependent variable, and use some or 
all of age (RIDAGEYR), sex (RIAGENDR), ethnicity (RIDRETH1), and sleep 
duration (SLD012) as explanatory variables. The sleep variable is contained 
in the sleep file SLQ_I, which we did not consider in class.
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
# from data_prep import dx
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def plot_fit_by_age(result, fml):

    # Create a dataframe in which all variables are at the reference
    # level
    da = dx.iloc[0:100, :].copy()
    da["RIDAGEYR"] = np.linspace(18, 80, 100)
    da["RIDRETH1"] = "OH"
    
    fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(nrows=3, ncols=2,
                                                 figsize=(12, 15))
    fig.suptitle(fml, fontsize=16)
    i = 1
    for female in 0, 1:
        for sleep in 5, 8, 11:
            db = da.copy()
            db.Female = female
            db.SLD012 = sleep
    
            pr = result.predict(exog=db)
    
            la = "Female" if female == 1 else "Male"
            la += ", Sleep=%.0f" % sleep
            ax = eval("ax%s" % (str(i)))
            ax.plot(da.RIDAGEYR, pr, '-', label=la)
            
            # Filter original points and combine scatter plot with our predictions
            dcompare = dx[(dx["RIDRETH1"] == "OH") & (dx["RIDAGEYR"] >= 18) & 
                          (dx["RIDAGEYR"] <= 100) & (dx["Female"] == female) & 
                          (dx["SLD012"] < sleep + 2) & 
                          (dx["SLD012"] > sleep - 2)].copy()
            
            #ax.axes([0.1, 0.1, 0.66, 0.8])
            ax.grid(True)
            ax.plot(dcompare["RIDAGEYR"], dcompare["BMXBMI"], 'o')
            ax.set_xlabel("Age (years)", size=15)
            ax.set_ylabel("BMI", size=15)
            ax.legend(loc='upper left')
            i = i + 1
    pdf.savefig()
    

def plot_fit_by_age_all(result, fml, save_as_pdf = True, fig_name = ""):

    # Create a dataframe in which all variables are at the reference
    # level
    da = dx.iloc[0:100, :].copy()
    da["RIDAGEYR"] = np.linspace(18, 80, 100)
    da["RIDRETH1"] = "OH"
    
    plt.figure(figsize=(8, 5))
    plt.clf()
    plt.axes([0.1, 0.1, 0.66, 0.8])
    plt.grid(True)
    
    for female in 0, 1:
        for sleep in 5, 8, 11:
            db = da.copy()
            db.Female = female
            db.SLD012 = sleep
    
            pr = result.predict(exog=db)
    
            la = "Female" if female == 1 else "Male"
            la += ", Sleep=%.0f" % sleep
            plt.plot(da.RIDAGEYR, pr, '-', label=la)
            
            
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)

    plt.xlabel("Age (years)", size=15)
    plt.ylabel("BMI", size=15)
    plt.title(fml, size=11)
    plt.title(fml, fontdict={"fontsize": 9})
    if save_as_pdf:
        pdf.savefig()
    else:
        plt.savefig('./' + fig_name);
    return 
    
def plot_fit_by_age_race(result, fml, save_as_pdf = True, fig_name = ""):
    
    # Create a dataframe in which all variables are at the reference
    # level
    ethnicities = ["MA", "OH", "NHW", "NHB", "OR"]
    ethnic_dict = {"MA": "Mexican American",
                   "OH": "Hispanic", 
                   "NHW": "Non-Hispanic White", 
                   "NHB": "Non-Hispanic Black", 
                   "OR": "Other Non-Hispanic"}
    da = dx.iloc[0:100, :].copy()
    da["RIDAGEYR"] = np.linspace(18, 80, 100)
    
    fig, ((ax1, ax3), (ax2, ax4), (ax6, ax5)) = plt.subplots(nrows=3, ncols=2,
                                         figsize=(12, 15))
    
    fig.suptitle(fml, fontsize=16)
    
    i = 1
    for ethnicity in ethnicities:
        da["RIDRETH1"] = ethnicity
        for female in 0, 1:
            for sleep in 5, 8, 11:
                db = da.copy()
                db.Female = female
                db.SLD012 = sleep
                ax = eval("ax%s" % (str(i)))
        
                pr = result.predict(exog=db)
        
                la = "Female" if female == 1 else "Male"
                la += ", Sleep=%.0f" % sleep
                ax.plot(da.RIDAGEYR, pr, '-', label=la)
                
                ax.grid(True)
                ax.set_xlabel("Age (years)", size=15)
                ax.set_ylabel("BMI", size=15)
                ax.legend(loc='upper left')
                ax.title.set_text(ethnic_dict[ethnicity])
        i = i + 1
    ax6.axis('off')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
    if save_as_pdf:
        pdf.savefig()
    else:
        plt.savefig('./' + fig_name);
    return 
    
    
pdf = PdfPages("hw1_model.pdf")

# Open Data (run this after data_prep.py)
out_data_path = "./data/clean_nhanes_sleep.parq"
dx = pd.read_parquet(out_data_path)

# Main effects only (additive, linear model)
fml0 = "BMXBMI ~ RIDAGEYR + SLD012 + Female + RIDRETH1"
# fml0 = "BPXSY1 ~ RIDAGEYR + BMXBMI + Female + RIDRETH1"
model0 = sm.OLS.from_formula(fml0, data=dx)
result0 = model0.fit()
plot_fit_by_age(result0, fml0)
plot_fit_by_age_all(result0, fml0)


# Use AIC to select the degrees of freedom for age and BMI, each of
# which is modeled using spline basis functions.
aic = np.zeros((15, 15)) + np.nan
for age_df in range(3, 15):
    for sleep_df in range(3, 15):
        fml1 = "BMXBMI ~ bs(RIDAGEYR, age_df) + bs(SLD012, sleep_df) + Female + RIDRETH1"
        model1 = sm.OLS.from_formula(fml1, data=dx)
        result1 = model1.fit()
        aic[age_df, sleep_df] = result1.aic


# Make a heatmap of the AIC values, so we can see where the best fit
# is.
plt.clf()
plt.imshow(aic, interpolation="nearest")
plt.xlabel("Sleep df", size=15)
plt.ylabel("Age df", size=15)
plt.colorbar()
plt.xlim(3, 14)
plt.ylim(3, 14)
pdf.savefig()

# RESULT: from plot, choose sleep = 4 and age = 5

# degrees of freedom, as selected by AIC.
fml2 = "BMXBMI ~ bs(RIDAGEYR, 5) + bs(SLD012, 4) + Female + RIDRETH1"
model2 = sm.OLS.from_formula(fml2, data=dx)
result2 = model2.fit()
plot_fit_by_age(result2, fml2)
plot_fit_by_age_all(result2, fml2)

# Take model 5 above and allow the curves to vary by sex
fml7 = "BMXBMI ~ (bs(RIDAGEYR, 5) + bs(SLD012, 4)) * Female + RIDRETH1"
model7 = sm.OLS.from_formula(fml7, data=dx)
result7 = model7.fit()
plot_fit_by_age(result7, fml7)
plot_fit_by_age_all(result7, fml7, False, "regression_plot.png")

# Now include interactions between age and BMI, along with the other
# interactions, but the role of BMI is conditionally linear
fml8 = "BMXBMI ~ bs(RIDAGEYR, 5) * SLD012 * Female + RIDRETH1"
model8 = sm.OLS.from_formula(fml8, data=dx)
result8 = model8.fit()
plot_fit_by_age(result8, fml8)
plot_fit_by_age_all(result8, fml8)

# This is similar to the model above, except that BMI is now allowed
# to be nonlinear
fml9 = "BMXBMI ~ (bs(RIDAGEYR, 5) + bs(SLD012, 4)) * Female * RIDRETH1"
model9 = sm.OLS.from_formula(fml9, data=dx)
result9 = model9.fit()
plot_fit_by_age(result9, fml9)
plot_fit_by_age_all(result9, fml9)
plot_fit_by_age_all(result9, fml9, False, "trial_spline.png")
plot_fit_by_age_race(result9, fml9, False, "sleep6_spline_race.png")

# dx.groupby("RIDRETH1").count().iloc[:,1]

pdf.close()

# Building histogram of sleep hours
plt.figure(figsize=(8, 5))
plt.clf()
plt.hist(dx["SLD012"])
plt.title("Histogram for number of hours slept by NHANES participants")
plt.xlabel("Hours of sleep", size=15)
plt.ylabel("Frequency", size=15)
plt.figtext(.8, .8, "$\mu$ = %.1f" % (np.mean(dx["SLD012"])))
plt.figtext(.8, .7, "$\sigma$ = %.1f" % (np.std(dx["SLD012"])))
plt.savefig("sleep_hist.png")






























