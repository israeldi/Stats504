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

pdf = PdfPages("trends_1.pdf")
fml1 = "Num ~ bs(Year, 4) + C(DayOfWeek) + " + doy
# With Interaction Terms:
fml2 = "Num ~ bs(Year, 4) + C(DayOfWeek) * " + doy

opts = {"DayOfWeek": {"lw": 3}, "CommunityArea": {"color": "grey", "lw": 2, "alpha": 0.5}}

# Get the empirical mean and variance of the response variable
# in a series of fitted value strata.
def get_meanvar(model, result):
    c = pd.qcut(result.fittedvalues, np.linspace(0.1, 0.9, 9))
    dd = pd.DataFrame({"c": c, "y": model.endog})
    mv = []
    for k,v in dd.groupby("c"):
        mv.append([v.y.mean(), v.y.var()])
    mv = np.asarray(mv)
    return mv

# Plot the empirical mean/variance relationship
def plot_empirical_meanvar(pt, mv):
    plt.clf()
    plt.title(pt)
    plt.grid(True)
    plt.plot(mv[:, 0], mv[:, 1], 'o', color='orange')
    mx = mv.max()
    mx *= 1.04
    b = np.dot(mv[:, 0], mv[:, 1]) / np.dot(mv[:, 0], mv[:, 0])
    plt.plot([0, mx], [0, b*mx], color='purple')
    plt.plot([0, mx], [0, mx], '-', color='black')
    plt.xlim(0, mx)
    plt.ylim(0, mx)
    plt.xlabel("Mean", size=15)
    plt.ylabel("Variance", size=15)
    pdf.savefig()

# Plot the fitted means curves for two variables, holding the others fixed.
def plot_fit(ax1, ax2, pt, cdat, dz, is_additive):
    
    i = 1
    for tp in "Year", "DayOfYear":
        for vn in "DayOfWeek", "CommunityArea":

            if vn == "CommunityArea" and not spacetime:
                continue
            
            ax = eval("ax%s" % (str(i)))
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

            ax.grid(True)

            for u in dz[vn].unique():
                dp[vn] = u
                pr = result.predict(exog=dp)
                if vn == "DayOfWeek":
                    ax.plot(dp[tp], pr, '-', label=u, **opts[vn])
                else:
                    ax.plot(dp[tp], pr, '-', **opts[vn])

            ax.set_xlabel(tp, size=14)
            ax.set_ylabel("Expected number of reports per day", size=16)
            if is_additive:
                ax.title.set_text(pt + ": Additive")
            else:
                ax.title.set_text(pt + ": Non-Additive")

            if vn == "DayOfWeek":
                ha, lb = plt.gca().get_legend_handles_labels()
                leg = plt.figlegend(ha, lb, "center right")
                leg.draw_frame(False)
        i = 2

def plot_resid_acor(ax, pt, result, is_additive):
    a = sm.tsa.acf(result.resid_pearson)

    ax.grid(True)
    ax.plot(a)
    if is_additive:
        ax.title.set_text(pt + ": Additive")
    else:
        ax.title.set_text(pt + ": Non-Additive")
    ax.set_xlabel("Lag (days)", size=15)
    ax.set_ylabel("Autocorrelation", size=15)
    ax.set_ylim(-0.2, 1)


# Loop over primary crime types, create a Poisson model for each type
# bic = []

i = 1
for pt, dz in cdat.groupby("PrimaryType"):
    fig, ((ax1, ax4),(ax2, ax5), (ax3, ax6)) = plt.subplots(nrows=3, ncols=2,
                                     figsize=(12, 15))

    fig.suptitle('Comparing additive vs non-additivity temporal variation', 
                 fontsize=18)
        
    # Create and fit the model
    model = sm.GLM.from_formula(fml1, family=sm.families.Poisson(), data=dz)
    result = model.fit(scale='X2')

    plot_fit(ax1, ax2, pt, cdat, dz, True)
    plot_resid_acor(ax3, pt, result, True)
    
    model = sm.GLM.from_formula(fml2, family=sm.families.Poisson(), data=dz)
    result = model.fit(scale='X2')

    plot_fit(ax4, ax5, pt, cdat, dz, False)
    plot_resid_acor(ax6, pt, result, False)
    plt.savefig("./plots/plot%s.png" % (i))
    i = i + 1
    # pdf.savefig()

# pdf.close()

'''

#############################################################################

# Get Correlations
counts = cdat.loc[:,["PrimaryType", "Num"]]
counts = counts.pivot(columns='PrimaryType', values='Num')
counts = counts.apply(lambda x: pd.Series(x.dropna().values)).fillna('')
corrs = counts.corr()

crime_types = ["ASSAULT", "BATTERY"]
dz = cdat[cdat['PrimaryType'].isin(crime_types)]
model = sm.GLM.from_formula(fml, family=sm.families.Poisson(), data=dz)
result = model.fit(scale='X2')
for pt in crime_types:
    plot_fit(pt, cdat, dz)

pdf.close()
'''




































