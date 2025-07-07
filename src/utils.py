import pandas as pd
import numpy as np
from matplotlib.pyplot import rcParams
from scipy.stats.qmc import LatinHypercube

def load_curves(src):
    # Load in Jean's curves data
    # In some of the curves files the last 4 columns headers are missing, namely:
    # DIC displacement in three components plus Euclidean norm
    # To get around this I read in the header separately and append the extra values if necessary
    curves_data = pd.read_csv(src, sep = "\\s+", index_col=False, skiprows=1,header=None)
    curves_headers = pd.read_csv(src, sep = "\\s+", index_col=False).columns.values
    if curves_data.shape[1] != len(curves_headers):
        print("Appending header for " + src)
        curves_headers = np.append(curves_headers, np.array(["Displacement_DIC[0]", "Displacement_DIC[1]", "Displacement_DIC[2]", "Displacement_DIC.two_norm()"]))
    
    curves_data.columns = curves_headers

    # Add a zero
    curves_data.loc[-1] = np.zeros(curves_data.shape[1])
    curves_data.index = curves_data.index+1
    curves_data.sort_index(inplace=True)

    return(curves_data)

def extract_failure_load(curves_data, failure_cols, load_col = "applied_load"):
    # Extract failure load in each mode using linear interpolation   
    failure_load = []
    # Loop over each failure mode
    for col in failure_cols:
        # Determine if failure occurs in this mode
        failure_init = curves_data[col]>=1.0
        if failure_init.any():
            # If failure occurs - find iteration at which failure occurs
            # Linearly interpolate failure load from previous iteration
            failure_ind = curves_data[failure_init].iloc[0].name
            postfailure_load = curves_data.iloc[failure_ind][load_col]
            prefailure_load = curves_data.iloc[failure_ind-1][load_col]
            postfailure_fi = curves_data.iloc[failure_ind][col]
            prefailure_fi = curves_data.iloc[failure_ind-1][col]
            # Linear interpolate assuming failure happens at failure index of 1
            alpha = (1 - prefailure_fi)/(postfailure_fi - prefailure_fi)
            failure_load.append(prefailure_load + alpha*(postfailure_load - prefailure_load))
        else:
            # Otherwise return a nan
            failure_load.append(np.nan)
            
    return(failure_load)

def uniform_LHS(input_df, N):
    # Takes an LHS of set of a set of inputs and transforms to lie between
    # The specified lower and upper bounds given in input_dict
    
    d = input_df.shape[1] # Number of inputs
    # Generate a set of N samples using the required Latin Hypercube sampler
    sampler = LatinHypercube(d=d, optimization = "random-cd")
    FLHS = sampler.random(N)
    # Get lower and upper bounds
    lb = input_df.iloc[0].values
    ub = input_df.iloc[1].values
    # Transform onto the required range
    xLHS = FLHS*(ub-lb) + lb
    xLHS = pd.DataFrame(xLHS, columns = input_df.columns)
    
    return(xLHS)

def set_plot_params():
    # Set commonly used plot parameters to desired values
    # Plotting parameters
    rcParams.update({'figure.figsize' : (8,6),
                    'font.size': 16,
                    'figure.titlesize' : 18,
                    'axes.labelsize': 18,
                    'xtick.labelsize': 15,
                    'ytick.labelsize': 15,
                    'legend.fontsize': 12}) 