import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

sys.path.insert(0, "src/")

from hashin_post import *
from utils import load_curves, uniform_LHS

if __name__ == "__main__":
    # Define folder where vtks and (if used) curves files are stored
    vtk_folder = "inputs/VTKs/"
    curves_folder = "inputs/curves/"
    append_curves = True # Append force and displacement data from curves files?

    # Load in curves data if needed
    if append_curves:
        curves_list = [file for file in os.listdir(curves_folder)]
        # Define columns of interest from curves file
        curves_cols = ["applied_displacement", "applied_load", "Displacement_DIC[2]", "D_xmax[0]", "max_damage_cohesive", "max_predamage_cohesive"]
    
    # Define list of inputs to be varied, and their lower and upper bounds
    inputs = {
            'X_T' : [2000.0, 3000.0],
            'X_C' : [1000.0, 2000.0],
            'Y_T' : [40.0,   180.0],
            'Y_C' : [140.0,  440.0],
            'S_L' : [70.0,   150.0],
            'S_T' : [70.0,   150.0]
        }
    inputs = pd.DataFrame(inputs, index = ["LB","UB"])

    N_init_sam = len([folder for folder in os.listdir(vtk_folder)]) # Get number of initial samples from vtks folder
    N_per_sam = 5  # Number of additional samples required per initial sample
    N = N_init_sam * N_per_sam # Overall number of samples
    
    # Generate Latin Hypercube
    x_sam = uniform_LHS(inputs, N)

    # Create second index referring to initial sample number
    x_sam["init_sam"] = [i for i in range(N_init_sam) for j in range(N_per_sam)]
    # Plot histograms of inputs
    x_sam.hist(bins = 25)
    x_sam.to_csv("outputs/hashin_UQ_inputs.csv", index = False) # Write to file

    # Loop over each inital sample, load in vtk output and re-calculate Hashin using
    # input values from the LHS. VTKs for each sample are stored in separate sub-folders
    out_frame = None
    for i, subfolder in enumerate(os.listdir(vtk_folder)):
        print(subfolder)
        # Load the relevant curves file if neeeded
        if append_curves:
            identifier = subfolder.split("_")[-1] # Text used to find the curves file matching the current vtk folder
            curves_file = [file for file in curves_list if os.path.splitext(file)[0].split("_")[-1] == identifier][0]
            curves_i = load_curves(os.path.join(curves_folder, curves_file))
            curves_i = curves_i[curves_cols]

        src = os.path.join(vtk_folder, subfolder)
        # Get all vtks in the subfolder and automatically detect number of subdomains and iterations
        vtks = [os.path.splitext(file)[0] for file in os.listdir(src) if os.path.splitext(file)[1] == ".vtk"]
        n_subdomains = max([int(vtk.split("_")[0]) for vtk in vtks]) + 1 
        n_iter = max([int(vtk.split("_")[2].strip("iter-")) for vtk in vtks]) + 1    
        # Run the Hashin post-processing script for each new sample matched to the current initial sample
        for sample in x_sam[x_sam["init_sam"] == i].iterrows():
            print("Sample " + str(sample[0]))
            x_i = sample[1] 
            # Run Hashin post-processing script
            out_frame_i = hashin_post(src, n_subdomains, n_iter, x_i["X_T"], x_i["X_C"], x_i["Y_T"], x_i["Y_C"], x_i["S_L"], x_i["S_T"], x_i["Y_T"])
            # Append other useful info from curves file
            if  append_curves:
                out_frame_i = pd.concat((out_frame_i, curves_i), axis=1)

            out_frame_i["Increment"] = out_frame_i.index
            out_frame_i["Sample"] = sample[0]
            out_frame_i["init_sam"] = int(x_i["init_sam"])
            # Write individual sample to .csv
            out_frame_i.to_csv("_".join(("outputs/hashin_UQ_out",subfolder,str(sample[0]))) + ".csv", index=False)
            # Append to overall dataframe
            if out_frame is None:
                out_frame = out_frame_i
            else:
                out_frame = pd.concat((out_frame, out_frame_i), axis = 0)
    
    out_frame.to_csv("outputs/hashin_UQ_out_all.csv", index=False)
    plt.show()
