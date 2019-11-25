# Principal Component Analysis of the US census data for
# family income and age/sex structure.  The unit of
# analysis is a census tract.  See the IPUMS/NHGIS
# codebooks for more information about the variables.

import numpy as np
import pandas as pd
from get_data_house import dx, housvars, incvars, popvars, io, age_bands, plot_inc, plot_pop, plot_hous
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages("hw2_pca.pdf")

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
# cx: number of loadings
# k: year
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


# Plot the loadings for the income data.
for cx in range(3):
    # io represents income band
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
   
###########################################################################
# Do the PCA for the house data.
# FIXME: GET PLOTS TO SHOW UP in correct year order
u_hous, v_hous, mn_hous = do_pca(dy, housvars)


fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(nrows=3, ncols=2,
                                     figsize=(12, 15))

fig.suptitle('Component Loadings for Household Type by Houshold Size', 
             fontsize=18)
i = 1
for cx in range(1, 3):
    for k in range(3):
        
        if cx == 0:
            vp = mn_hous.iloc[k:39:3]
            ylabel = "Mean"
        else:
            # Each column represents principal component
            # Each row represents the loading, so we take every 5th one, to get
            # the loading corresponding to that year
            vp = v_hous.iloc[k:39:3, cx - 1]
            ylabel = "Component %d loading" % cx

        # ylim = (-0.2, 0.2) if cx > 0 else None
        ylim = None
        ax = eval("ax%s" % (str(i)))
        plot_hous(ax, vp, ylabel, "%d House Structure" % (1990 + k*10), ylim)
        i = i + 1
        # pdf.savefig()
# plt.tight_layout()
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("component_loadings.png")
pdf.savefig()

'''
# Plot the loadings for the house data.
num_loadings = 3
for cx in range(num_loadings + 1):
    if cx == 0:
        vp = mn_hous
        title = "Mean Household Type"
        ylabel = "Mean"
    else:  
        # io represents income band
        vp = v_hous.iloc[:, cx - 1]
        title = "Household Type component loadings"
        ylabel = "Component %d loading" % (cx)
    plot_hous(vp, ylabel, title, None)
    pdf.savefig()
'''

# Plot Income, Population, and Household type scores
num_loadings = 1
'''
for j in range(num_loadings):
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.scatter(u_pop[:, j], u_inc[:, j], u_hous[:, j], c=u_hous[:, j], 
               cmap='viridis', linewidth=0.5);
    #plt.grid(True)
    #plt.plot(u_inc[:, j], u_hous[:, j], 'o', alpha=0.5, rasterized=True)
    #plt.xlabel("Income layer %d scores" % (j +1), size=15)
    #plt.ylabel("Household Type layer %d scores" % (j +1), size=15)
    pdf.savefig()
'''

# Get correlations of PCA scores for all three data sets
df = pd.DataFrame([])
df_names = ["inc", "pop", "hous"]
for df_name in df_names:
    df_temp = eval("u_" + df_name)
    for load_num in range(3):
        df[df_name + str(load_num+1)] = df_temp[:,load_num]

f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr('pearson'), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
corr = df.corr('pearson')
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
pdf.savefig()

###### Most correlation exists betwen inc1 vs home2, pop1 vs hous2, pop2 vs hous1
#############


# Plot the population scores against Household type scores
comp1 = 2
comp2 = 1
plt.clf()
plt.grid(True)
plt.plot(u_hous[:, comp1], u_inc[:, comp2], 'o', alpha=0.1, rasterized=True)
plt.xlabel("House layer %d scores" % (comp1), size=15)
plt.ylabel("Income layer %d scores" % (comp2), size=15)
plt.figtext(.8, .8, "$rho$ = %.1f" % (corr.loc["hous2", "inc1"]))
plt.axvline(x=0.0, color='black', linestyle='-')
plt.axhline(y=0.0, color='black', linestyle='-')
pdf.savefig()
    
# Plot the income scores against Household type scores

fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2,
                                     figsize=(20, 12))

fig.suptitle('Principal Component Scores for Household Type and Population Structure', 
             fontsize=18)
i = 1
comp1 = 2
comp2 = 1
# plt.clf()
i = 1
ax = eval("ax%s" % (str(i)))
ax.grid(True)
ax.plot(u_hous[:, comp1], u_pop[:, comp2], 'o', alpha=0.1, rasterized=True)
ax.set_xlabel("House layer %d scores" % (comp1), size=15)
ax.set_ylabel("Population layer %d scores" % (comp2), size=15)
ax.text(0.03, 0.03, "$rho$ = %.3f" % (corr.loc["hous2", "pop1"]),
        fontsize= 14)
ax.axvline(x=0.0, color='black', linestyle='-')
ax.axhline(y=0.0, color='black', linestyle='-')

# Plot the income scores against Household type scores
comp1 = 1
comp2 = 2
i = 2
ax = eval("ax%s" % (str(i)))
#plt.figure(figsize=(8, 5))
#plt.clf()
ax.grid(True)
ax.plot(u_hous[:, comp1], u_pop[:, comp2], 'o', alpha=0.1, rasterized=True)
ax.set_xlabel("House layer %d scores" % (comp1), size=15)
ax.set_ylabel("Population layer %d scores" % (comp2), size=15)
ax.text(0.03, 0.02, "$rho$ = %.3f" % (corr.loc["hous1", "pop2"]),
        fontsize= 14)
ax.axvline(x=0.0, color='black', linestyle='-')
ax.axhline(y=0.0, color='black', linestyle='-')
plt.savefig("component_scores.png")
pdf.savefig()

pdf.close()
































