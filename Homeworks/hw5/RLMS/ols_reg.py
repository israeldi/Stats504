import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import lars_path
import patsy
from data_prep import dx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

fml = "I(np.log2(J10)) ~ 0 + C(status) + (bs(year, 5) + bs(age, 5) + bs(educ, 5))*Female + C(OCCUP08) + I(np.log2(J8))"

y, x = patsy.dmatrices(fml, dx, return_type='dataframe')
y -= y.mean()
x -= x.mean(0)
xnames = x.columns

y = np.asarray(y)[:, 0]
x = np.asarray(x)

a, b, coefs = lars_path(x, y)

coefs = pd.DataFrame(coefs, index=xnames)
s = (coefs != 0).sum(1)
ii = np.argsort(-s)
coefs = coefs.iloc[ii, :]
