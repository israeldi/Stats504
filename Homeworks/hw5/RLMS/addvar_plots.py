import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
from data_prep import dx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages("addvar_plots.pdf")

dx["log_wage"] = np.log2(dx.age)
dx["log_hours"] = np.log2(dx.J8)


# Added variable plot for age
fml_wages_0 = "log_wage ~ C(status) + (bs(year, 5) + bs(educ, 5))*Female + C(OCCUP08) + I(np.log2(J8))"
fml_age_0 = "age ~ C(status) + (bs(year, 5) + bs(educ, 5))*Female + C(OCCUP08) + I(np.log2(J8))"

wage_adj = sm.OLS.from_formula(fml_wages_0, data=dx).fit().resid
age_adj = sm.OLS.from_formula(fml_age_0, data=dx).fit().resid

plt.clf()
plt.grid(True)
plt.plot(age_adj, wage_adj, 'o', alpha=0.5, rasterized=True)
plt.xlabel("Age residual", size=15)
plt.ylabel("Log wage residual", size=15)
pdf.savefig()


# Added variable plot for year
fml_wages_0 = "log_wage ~ C(status) + (bs(age, 5) + bs(educ, 5))*Female + C(OCCUP08) + I(np.log2(J8))"
fml_year_0 = "year ~ C(status) + (bs(age, 5) + bs(educ, 5))*Female + C(OCCUP08) + I(np.log2(J8))"

wage_adj = sm.OLS.from_formula(fml_wages_0, data=dx).fit().resid
year_adj = sm.OLS.from_formula(fml_year_0, data=dx).fit().resid

plt.clf()
plt.grid(True)
plt.plot(year_adj, wage_adj, 'o', alpha=0.5, rasterized=True)
plt.xlabel("Year residual", size=15)
plt.ylabel("Log wage residual", size=15)
pdf.savefig()

# Added variable plot for education
fml_wages_0 = "log_wage ~ C(status) + (bs(age, 5) + bs(year, 5))*Female + C(OCCUP08) + I(np.log2(J8))"
fml_educ_0 = "educ ~ C(status) + (bs(age, 5) + bs(year, 5))*Female + C(OCCUP08) + I(np.log2(J8))"

wage_adj = sm.OLS.from_formula(fml_wages_0, data=dx).fit().resid
educ_adj = sm.OLS.from_formula(fml_educ_0, data=dx).fit().resid

plt.clf()
plt.grid(True)
plt.plot(educ_adj, wage_adj, 'o', alpha=0.5, rasterized=True)
plt.xlabel("Education residual", size=15)
plt.ylabel("Log wage residual", size=15)
pdf.savefig()

pdf.close()

import os
os.system("cp rlms_addvar_plots.pdf ~kshedden")