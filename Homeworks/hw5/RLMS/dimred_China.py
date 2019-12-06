import numpy as np
import pandas as pd
from statsmodels.regression.dimred import SIR
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from data import df

df1 = pd.read_csv("chns.csv.gz", sep=',')

# Create a log income variable
df1 = df1.loc[df1.indinc>=0, :]
df1["logindinc"] = np.log(1 + df1.indinc)

# Use dimension reduction regression for each of the
# below outcome variables
resp1 = ["d3kcal", "d3carbo", "d3fat", "d3protn"]

# Drop variables that we don't need
xv1 = ["age", "female", "urban", "logindinc", "educ"]
dx1 = df1.loc[:, resp1 + xv1]

# Center the data
xmean1 = dx1.loc[:, xv1].mean(0)
dx1.loc[:, xv1] -= xmean1


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

def plot_eigs(title, eigs):
    plt.clf()
    plt.grid(True)
    plt.title(title)
    plt.plot(eigs, '-o')
    plt.xlabel("Component", size=15)
    plt.ylabel("Eigenvalue", size=15)
    


ages1 = np.linspace(18, 80, 100)

# pdf = PdfPages("dimred.pdf")

# Base smoothing parameter for each value of ndim
spl1 = {1: 0.2, 2: 0.2, 3: 0.3, 4: 0.3, 5: 0.4}

for rv1 in resp1:

    # Get the dimension reduction directions
    # Sliced Inverse Regression
    
    m1 = SIR(dx1[rv1], dx1.loc[:, xv1])
    s1 = m1.fit(slice_n=500)

    # Plot the eigenvalues
    plot_eigs(rv1, s1.eigs)
    # pdf.savefig()
    
    ndim1 = 1
    a1 = 0
    '''
    for ndim in 1, 2, 3, 4, 5:
        for a in 0, 0.3, 0.5:
    '''

    # Reduce the dimension of the covariates
    proj1 = s1.params.iloc[:, 0:ndim1]
    xmat1 = np.dot(dx1.loc[:, xv1], proj1)

    # Get the local regression function
    sp1 = spl1[ndim1] + a1
    f1 = kreg(dx1[rv1], xmat1, s=sp1)

    # Create a dataframe for prediction
    xp1 = dx1.iloc[0:100, :].loc[:, xv1].copy()
    xp1["age"] = ages1
    xp1["logindinc"] = xmean1.logindinc
    xp1["educ"] = xmean1.educ

    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.axes([0.11, 0.12, 0.68, 0.8])
    plt.grid(True)

    for female in 0, 1:
        for urban in 0, 1:

            # Prepare a dataframe for prediction
            xp1.loc[:, "female"] = female
            xp1.loc[:, "urban"] = urban

            # Transform the prediction dataframe the same as
            # the fitting dataframe
            xq1 = xp1 - xmean1
            xq1 = np.dot(xq1, proj1)

            # Get the fitted values
            yp1 = [f1(xq1[i, :]) for i in range(100)]

            # A label for the line we will add here
            label1 = [["rural", "urban"][urban], ["male", "female"][female]]
            label1 = "%s %s" % tuple(label1)

            # Add one line to the plot
            plt.plot(ages1, yp1, '-', label=label1)

    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)

    plt.title("dim=%d, sp=%.2f" % (ndim1, sp1))
    plt.ylabel(rv1, size=15)
    plt.xlabel("Age", size=15)

    # pdf.savefig()

# pdf.close()
