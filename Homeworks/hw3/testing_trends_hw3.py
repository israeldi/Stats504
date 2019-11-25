"""
Use Poisson regression to decompose the crime rate data into
a long-term time trend, annual, and weekly cycles.  The crime
rates are modeled as the number of crimes per day of a given
type.  In the "spacetime" model the data are disaggregated
by spatial region.  In the non-spacetime models, the data are
aggregated to the whole city of Chicago.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import patsy
from chicago_data import cdat

# If False, sum over the communities to get a city-wide total for each
# crime type x day.
spacetime = False

# 2001 and 2002 data look incorrect
cdat = cdat.loc[cdat.Year >= 2003, :]

# Recode day of week with text labels
cdat.loc[:, "DayOfWeek"] = cdat.DayOfWeek.replace({0: "Mo", 1: "Tu", 2: "We", 3: "Th", 4: "Fr", 5: "Sa", 6: "Su"})

# Construct a Fourier basis for day of year
def fb(xm):
    for q in 1, 2, 3:
        xm.loc[:, "DayOfYear_sin_%d" % q] = np.sin(2 * np.pi * q * xm.DayOfYear / 365.25)
        xm.loc[:, "DayOfYear_cos_%d" % q] = np.cos(2 * np.pi * q * xm.DayOfYear / 365.25)

if not spacetime:
    # Collapse over spatial units
    a = {"Num": np.sum, "DayOfWeek": "first"}
    cdat = cdat.groupby(["PrimaryType", "Year", "DayOfYear"]).agg(a)
    cdat = cdat.reset_index()

fb(cdat)

# A few values of CommunityArea are missing.
cdat = cdat.dropna()

# Model terms for the day of year Fourier basis
doy = "(DayOfYear_sin_1 + DayOfYear_cos_1 + DayOfYear_sin_2 + DayOfYear_cos_2 + DayOfYear_sin_3 + DayOfYear_cos_3)"
#doy = "(DayOfYear_sin_1 + DayOfYear_cos_1 + DayOfYear_sin_2 + DayOfYear_cos_2)"

# With Interaction Terms:
pdf = PdfPages("testing_timetrends_time_ASS_BAT.pdf")
fml = "Num ~ bs(Year, 4) + C(PrimaryType) * C(DayOfWeek) * " + doy

opts = {"DayOfWeek": {"lw": 3}, "CommunityArea": {"color": "grey", "lw": 2, "alpha": 0.5}}


# Plot the fitted means curves for two variables, holding the others fixed.
def plot_fit(pt, cdat, dz):

    for tp in "Year", "DayOfYear":
        for vn in "DayOfWeek", "CommunityArea":

            if vn == "CommunityArea" and not spacetime:
                continue

            # Create a data set so we can plot the fitted regression function
            # E[Y|X=x] where certain components of x are held fixed and others
            # are varied systematically.
            p = 100
            dp = cdat.iloc[0:p, :].copy()
            dp.Year = 2015
            dp.DayOfWeek = "Su"
            dp.CommunityArea = 1
            dp.DayOfYear = 180
            dp.PrimaryType = pt
            fb(dp)


            if tp == "Year":
                dp.Year = np.linspace(2003, 2018, p)
            elif tp == "DayOfYear":
                dp.DayOfYear = np.linspace(2, 364, p)
                fb(dp)

            plt.clf()

            if vn == "DayOfWeek":
                plt.axes([0.15, 0.1, 0.72, 0.8])

            plt.grid(True)

            for u in dz[vn].unique():
                dp[vn] = u
                pr = result.predict(exog=dp)
                if vn == "DayOfWeek":
                    plt.plot(dp[tp], pr, '-', label=u, **opts[vn])
                else:
                    plt.plot(dp[tp], pr, '-', **opts[vn])

            plt.xlabel(tp, size=14)
            plt.ylabel("Expected number of reports per day", size=16)
            plt.title(pt)

            if vn == "DayOfWeek":
                ha, lb = plt.gca().get_legend_handles_labels()
                leg = plt.figlegend(ha, lb, "center right")
                leg.draw_frame(False)

            pdf.savefig()


# Loop over primary crime types, create a Poisson model for each type
# bic = []
'''
for pt, dz in cdat.groupby("PrimaryType"):

    # Create and fit the model
    model = sm.GLM.from_formula(fml, family=sm.families.Poisson(), data=dz)
    result = model.fit(scale='X2')
    # bic.append(result.bic)

    # Estimate the scale as if this was a quasi-Poisson model
    print("%-20s %5.1f" % (pt, result.scale))

    plot_fit(pt, cdat, dz)

pdf.close()
'''

#############################################################################

# Regression on all crime types
'''
crime_types = cdat["PrimaryType"].unique()

model = sm.GLM.from_formula(fml, family=sm.families.Poisson(), data=cdat)
result = model.fit(scale='X2')
crime_types_half1 = crime_types[0:2]
for pt in crime_types_half1:
    plot_fit(pt, cdat, cdat)
    print(pt)
'''

# crime_types = ["ASSAULT", "BATTERY"]
# dz = cdat[cdat['PrimaryType'].isin(crime_types)]
crime_types = cdat["PrimaryType"].unique()
model = sm.GLM.from_formula(fml, family=sm.families.Poisson(), data=cdat)
result = model.fit(scale='X2')
for pt in crime_types:
    plot_fit(pt, cdat, cdat)

pdf.close()





































