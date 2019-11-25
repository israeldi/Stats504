# Principal Component Analysis of the US census data for
# family income and age/sex structure.  The unit of
# analysis is a census tract.  See the IPUMS/NHGIS
# codebooks for more information about the variables.

import numpy as np
import pandas as pd
from get_data import dx, incvars, popvars, io, age_bands, plot_inc, plot_pop
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# pdf = PdfPages("hw2_pca.pdf")

dy = dx.copy()
dy = dy.dropna()

def do_pca(x, ind_vars):

    dz = x.loc[:, ind_vars]
    ldz = np.log(dz + 0.01)

    ldz_mean = ldz.mean(0)
    ldz_c = ldz - ldz_mean

    u, s, vt = np.linalg.svd(ldz_c, 0)
    v = vt.T

    v = pd.DataFrame(v, index=ind_vars)

    return u, v, ldz_mean

# Do the PCA for the population data.
u_pop, v_pop, mn_pop = do_pca(dy, popvars)

# Plot the centroid and loadings for the population data.
for cx in range(4):
    for k in range(5):
        
        if cx == 0:
            vp = mn_pop.iloc[k:180:5]
            ylabel = "Mean"
        else:
            # Each column represents principal component
            # Each row represents the loading, so we take every 5th one, to get
            # the loading corresponding to that year
            vp = v_pop.iloc[k:180:5, cx - 1]
            ylabel = "Component %d loading" % cx

        ylim = (-0.2, 0.2) if cx > 0 else None

        plot_pop(vp, ylabel, "%d population structure" % (1970 + k*10), ylim)
        pdf.savefig()

# Do the PCA for the income data.
u_inc, v_inc, mn_inc = do_pca(dy, incvars)
pdf.close()


# Plot the loadings for the income data.
for cx in range(3):
    vp = v_inc.loc[io, cx].values
    title = "Income component loadings"
    ylabel = "Component %d loading" % (cx + 1)
    plot_inc(vp, ylabel, title)
    pdf.savefig()

# Plot the population scores against the income scores.i
for j in range(3):
    plt.clf()
    plt.grid(True)
    plt.plot(u_pop[:, j], u_inc[:, j], 'o', alpha=0.5, rasterized=True)
    plt.xlabel("Population layer %d scores" % (j +1), size=15)
    plt.ylabel("Income layer %d scores" % (j +1), size=15)
    pdf.savefig()

