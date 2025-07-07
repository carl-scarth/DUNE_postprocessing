import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy

# Add src directory to path
src_path = "src/"
sys.path.insert(0, src_path)

from utils import set_plot_params, load_curves, extract_failure_load
from CurvesPlot import *

# Load data from curves files
if __name__ == "__main__":
    set_plot_params() # Set up plot parameters
    color_cycle = [plt.cm.tab20(i) for i in range(20)] 
    color_cycle = color_cycle*50 # Set up colours to easily identify different samples from plots
    pd.set_option('display.max_rows', 10000)

    in_folder = "inputs/curves/" # Folder where inputs are stored
    
    # Define columns of curves files containing quantities of interest
    # Column containing applied load
    col_load = "applied_load"
    col_app_disp = "applied_displacement"
    # Columns containing displacement data
    cols_disp =  ["Displacement_DIC[2]",
                "D_xmax[0]"]
    # Columns containing failure indices
    cols_failure = ["Max_F_FT",
                "Max_F_FC",
                "Max_F_MT",
                "Max_F_MC",
                "Max_F_33",
                "max_damage_cohesive",
                "max_predamage_cohesive",
                ]
    cols_qoi = [col_load, col_app_disp] + cols_disp + cols_failure

    # Load in curves data for all files in the input folder, and extract 
    # failure load in each mode, if it exists
    failure_load = []
    i = 0
    for i, file in enumerate(os.listdir(in_folder)):
        curves_data = load_curves(os.path.join(in_folder,file))
        curves_data = curves_data[cols_qoi]
        curves_data["Increment"] = curves_data.index
        curves_data["Sample"] = i
        failure_load.append(extract_failure_load(curves_data, cols_failure))
        if i == 0:            
            curves_frame = curves_data
        else:
            curves_frame = pd.concat((curves_frame, curves_data), axis = 0)
    
    failure_load = pd.DataFrame(failure_load, columns = cols_failure)
    failure_load = failure_load/1000.0 # Convert to kN

    # Extract critical failure load
    crit_load = pd.concat((failure_load.min(axis=1), failure_load.idxmin(axis=1)), axis = 1)
    crit_load.columns = ["crit_failure_load","failure_mode"]
    failure_load = pd.concat((failure_load, crit_load), axis = 1)

    # Format curves data
    curves_frame.set_index(["Sample", "Increment"], inplace=True) # Create multi-index for curves data
    curves_frame[col_app_disp] = -curves_frame[col_app_disp] # Plot applied shortening
    curves_frame[col_load] = curves_frame[col_load]/1000.0 # Convert to kN
    
    # Initialise plots
    force_disp = CurvesPlot()
    force_disp_DIC = [CurvesPlot() for col in cols_disp]
    failure_plot = [CurvesPlot() for col in cols_failure]
    
    # Populate plots from each sample curves file
    for i, sample_df in curves_frame.groupby(level=0):
        # Plot Applied Load VS Applied Displacement
        force_disp.plot_fd(sample_df, col_load, col_app_disp, color = color_cycle[i])
        # Plot Applied Load VS "Measured" Displacement
        [point_plot.plot_fd(sample_df, col_load, column, color = color_cycle[i]) for column, point_plot in zip(cols_disp, force_disp_DIC)]
        # extract critical failure load and mode for current sample
        failure_load_i = copy.copy(failure_load.iloc[i])
        failure_load_i[failure_load_i.index != crit_load.iloc[i].failure_mode] = np.nan
        # Plot Applied Load VS Failure Index
        [fmode_plot.plot_failure_ind(sample_df, col_load, column, failure_load = failure_load_i, markersize = 12, color = color_cycle[i]) for column, fmode_plot in zip(cols_failure, failure_plot)]
    
    # Sets plot limits, add labels etc
    max_disp = [curves_frame[col].max() for col in cols_disp]
    max_app_disp = curves_frame[col_app_disp].max()
    max_force = curves_frame[col_load].max()
    force_disp.set_xlim([0, max_app_disp])
    force_disp.set_ylim([0, max_force])
    force_disp.set_title("Force VS Crosshead Displacement")
    [plot.set_xlim([0, disp_lim]) for plot, disp_lim in zip(force_disp_DIC, max_disp)]
    [plot.set_ylim([0, max_force]) for plot in force_disp_DIC]
    [plot.set_title(title) for plot, title in zip(force_disp_DIC, cols_disp)]
    for plot, title in zip(failure_plot, cols_failure):
        plot.set_xlim([0, max_force])
        plot.set_xlabel("Force (kN)")
        plot.plot_limit_state()
        plot.set_title(title)
    
    # Used to identify different samples vvv - not recommended for more than 10 samples
    #force_disp.plot_legend(simplify_labels=False)
    #[plot.plot_legend(simplify_labels=False) for plot in force_disp_DIC]
    #[plot.plot_legend(simplify_labels=False, loc = "upper left") for plot in failure_plot]

    # Print summary of results
    print(failure_load)
    failure_load.to_csv("outputs/curves_outputs.csv")
    plt.show()