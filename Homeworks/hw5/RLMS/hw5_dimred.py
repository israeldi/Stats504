import numpy as np
# import pandas as pd
from statsmodels.regression.dimred import SIR
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from data_prep import dv

# df = pd.read_csv("chns.csv.gz", sep=',')

df = dv.copy()

# Create a log income variable
df = df.loc[df.after_tax_wages >= 0, :]
df["logwage"] = np.log(1 + df.after_tax_wages)

# Use dimension reduction regression for each of the
# below outcome variables
resp = ["health_eval"]
# Drop variables that we don't need
xv = ['status', 'marst', 'OCCUP08', 'educ', 'age',
       'hours_worked', 'logwage', 'weight', 'height',
       'chr_heart', 'chr_lung', 'chr_liver', 'chr_kidney',
       'chr_stomach', 'chr_spinal', 'diabetes', 'heart_attack', 'smokes',
       'Female']
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
        # w = w.reshape((len(w), 1))
        return np.dot(np.transpose(y), w, out=None)

    return f

def plot_eigs(title, eigs):
    plt.clf()
    plt.grid(True)
    plt.title(title)
    plt.plot(eigs, '-o')
    plt.xlabel("Component", size=15)
    plt.ylabel("Eigenvalue", size=15)
    

ages = np.linspace(18, 80, 100)

pdf = PdfPages("dimred_hw5.pdf")

# Base smoothing parameter for each value of ndim
spl = {1: 0.2, 2: 0.2, 3: 0.3, 4: 0.3, 5: 0.4}

rv = resp

# Get the dimension reduction directions
# Sliced Inverse Regression

m = SIR(dx[rv], dx.loc[:, xv])
s = m.fit(slice_n=500)

# Plot the eigenvalues
plot_eigs(rv, s.eigs)
# pdf.savefig()
plt.savefig('plot_of_eigs.png')

ndim = 1
a = 0


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2,
                                     figsize=(12, 15))

fig.suptitle('RLMS Health Score Modeling based on health conditions', 
             fontsize=18)
i = 1
for ndim in 1, 2, 3:
    # for a in 0, 0.3, 0.5:

    # Reduce the dimension of the covariates
    proj = s.params.iloc[:, 0:ndim]
    xmat = np.dot(dx.loc[:, xv], proj)
    
    # Get the local regression function
    sp = spl[ndim] + a
    f = kreg(dx[rv], xmat, s=sp)
    
    # Create a dataframe for prediction
    xp = dx.iloc[0:100, :].loc[:, xv].copy()
    xp["age"] = ages
    
    fixed_vars = ["logwage", "educ", "OCCUP08", "status", "marst", "weight", 
                  "height", "hours_worked"]
    for var in fixed_vars:
        xp[var] = xmean[var]
    
    # Plot 
    '''
    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.axes([0.11, 0.12, 0.68, 0.8])
    plt.grid(True)
    '''
    health_vars = ["diabetes", "chr_spinal", "chr_stomach", "chr_kidney", 
                   "chr_liver", "chr_lung", "smokes", "chr_heart", "heart_attack"]

    for female in 0, 1:
        ax = eval("ax%s" % (str(i)))
        var_ind = 0 # Iterate through all health conditions
        for health_cond in health_vars:
            
            other_health = set(health_vars) - {health_cond}
            xp.loc[:, health_cond] = 1
            xp.loc[:, other_health] = 0
            
            # Prepare a dataframe for prediction
            xp.loc[:, "Female"] = female
            
            # Transform the prediction dataframe the same as
            # the fitting dataframe
            xq = xp - xmean
            xq = np.dot(xq, proj)
            
            # Get the fitted values
            yp = [f(xq[i, :]) for i in range(100)]
            
            # A label for the line we will add here
            label = [health_vars[var_ind], ["male", "female"][female]]
            label = "%s %s" % tuple(label)
            
            # Add one line to the plot
            ax.plot(ages, yp, '-', label=label)
            var_ind += 1
            if var_ind == len(health_vars):
                other_health = set(health_vars) - {health_cond}
                xp.loc[:, health_cond] = 0
                xp.loc[:, other_health] = 0
                
                # Prepare a dataframe for prediction
                xp.loc[:, "Female"] = female
                
                # Transform the prediction dataframe the same as
                # the fitting dataframe
                xq = xp - xmean
                xq = np.dot(xq, proj)
                
                # Get the fitted values
                yp = [f(xq[i, :]) for i in range(100)]
                
                # A label for the line we will add here
                label = ["None", ["male", "female"][female]]
                label = "%s %s" % tuple(label)
                
                # Add one line to the plot
                ax.plot(ages, yp, '-', label=label)
                var_ind += 1
        
        ha, lb = plt.gca().get_legend_handles_labels()
        #leg = ax.figlegend(ha, lb, "center right")
        #leg.draw_frame(False)
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # ax.legend(loc="center right")
        
        ax.set_title("dim=%d, sp=%.2f, %s" % (ndim, sp, 
                                              ["male", "female"][female]))
        ax.set_ylabel(rv, size=15)
        ax.set_xlabel("Age", size=15)
        ax.grid(True)
        i = i + 1

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(left=None, bottom=None, 
                    right=None, top=None, 
                    wspace=0.8, hspace=None)

# pdf.savefig()
plt.savefig('dim_health.png', bbox_inches="tight")

pdf.close()









