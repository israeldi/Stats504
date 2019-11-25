import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
from data_prep import dx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

dx["age_cen"] = dx.age - dx.age.mean()

# Mean structure, with and without gender
fml = "I(np.log2(J10)) ~ C(status) + (bs(year, 5) + bs(age, 5) + bs(educ, 5))*Female + I(np.log2(J8))"
fml0 = "I(np.log2(J10)) ~ C(status) + bs(year, 5) + bs(age, 5) + bs(educ, 5) + I(np.log2(J8))"

# Random intercepts for subjects
model1 = sm.MixedLM.from_formula(fml, re_formula="1", groups="idind", data=dx)
result1 = model1.fit()

# Random intercepts for subjects, and random slopes for age within subjects
model2 = sm.MixedLM.from_formula(fml, re_formula="1 + age_cen", groups="idind", data=dx)
result2 = model2.fit()

out = open("lme.txt", "w")
out.write(result1.summary().as_text())
out.write("\n\n")
out.write(result2.summary().as_text())
out.close()