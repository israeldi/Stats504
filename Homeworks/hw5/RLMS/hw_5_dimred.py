import numpy as np
import pandas as pd
from statsmodels.regression.dimred import SIR
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from data import df

df = pd.read_csv("chns.csv.gz", sep=',')

# Create a log income variable
df = df.loc[df.indinc>=0, :]
df["logindinc"] = np.log(1 + df.indinc)

# Use dimension reduction regression for each of the
# below outcome variables
resp = ["d3kcal", "d3carbo", "d3fat", "d3protn"]

# Drop variables that we don't need
xv = ["age", "female", "urban", "logindinc", "educ"]
dx = df.loc[:, resp + xv]

# Center the data
xmean = dx.loc[:, xv].mean(0)
dx.loc[:, xv] -= xmean


def kreg(y, xmat, s):
    """
    Generate a function that evaluates the kernel regression estimate
    of E[y|x] at a given point x, using the bandwidth parameter s.
    """
    
    # Gaussian Kernel function 
    def f(x):
        w = np.sum((xmat - x)**2, 1)
        w = np.exp(-w/s**2)
        w /= w.sum()
        return np.dot(y, w)

    return f

ages = np.linspace(18, 80, 100)

pdf = PdfPages("dimred.pdf")

# Base smoothing parameter for each value of ndim
spl = {1: 0.2, 2: 0.2, 3: 0.3, 4: 0.3, 5: 0.4}

for rv in resp:

    # Get the dimension reduction directions
    # Sliced Inverse Regressiongoodn
    
    m = SIR(dx[rv], dx.loc[:, xv])
    s = m.fit(slice_n=500)

    # Plot the eigenvalues
    plt.clf()
    plt.grid(True)
    plt.title(rv)
    plt.plot(s.eigs, '-o')
    plt.xlabel("Component", size=15)
    plt.ylabel("Eigenvalue", size=15)
    pdf.savefig()
    
    ndim = 1
    a = 0
    '''
    for ndim in 1, 2, 3, 4, 5:
        for a in 0, 0.3, 0.5:
    '''

    # Reduce the dimension of the covariates
    proj = s.params.iloc[:, 0:ndim]
    xmat = np.dot(dx.loc[:, xv], proj)

    # Get the local regression function
    sp = spl[ndim] + a
    f = kreg(dx[rv], xmat, s=sp)

    # Create a dataframe for prediction
    xp = dx.iloc[0:100, :].loc[:, xv].copy()
    xp["age"] = ages
    xp["logindinc"] = xmean.logindinc
    xp["educ"] = xmean.educ

    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.axes([0.11, 0.12, 0.68, 0.8])
    plt.grid(True)

    for female in 0, 1:
        for urban in 0, 1:

            # Prepare a dataframe for prediction
            xp.loc[:, "female"] = female
            xp.loc[:, "urban"] = urban

            # Transform the prediction dataframe the same as
            # the fitting dataframe
            xq = xp - xmean
            xq = np.dot(xq, proj)

            # Get the fitted values
            yp = [f(xq[i, :]) for i in range(100)]

            # A label for the line we will add here
            label = [["rural", "urban"][urban], ["male", "female"][female]]
            label = "%s %s" % tuple(label)

            # Add one line to the plot
            plt.plot(ages, yp, '-', label=label)

    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)

    plt.title("dim=%d, sp=%.2f" % (ndim, sp))
    plt.ylabel(rv, size=15)
    plt.xlabel("Age", size=15)

    pdf.savefig()

pdf.close()
