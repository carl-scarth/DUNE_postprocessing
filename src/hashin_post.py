import vtk
import pandas as pd
import numpy as np
import os
import copy

def ascii_vtk_to_df_stress(vtkfile):
    # Read in VTK file as plain text to bypass use of the vtk library and reduce run time.
    # Assumes Jean kept the overall structure of the vtks fixed
    read_stress = False
    read_marker = False
    marker = []
    stress = []

    with open(vtkfile) as file:
        for line in file:
            # Read in marker block
            if read_marker and "LOOKUP_TABLE" not in line:
                if "LOCAL_STRESS" in line:
                    read_marker = False
                else:
                    marker.append([int(line.strip())])

            # Read in stress block
            if read_stress and "LOOKUP_TABLE" not in line:
                if "STRAIN" in line: 
                    read_stress = False # End of stress block
                    break # Break out of for loop as this is the last block we need to read
                else:
                    # Split string into stress components
                    stress.append([float(entry) for entry in line.strip().split()])

            # Detect start of MARKER block
            if "MARKER" in line:
                read_marker = True
            # Detect start of stress block
            if "LOCAL_STRESS" in line:
                read_stress = True

    df = np.concatenate((np.array(marker), np.array(stress)), axis = 1)
    
    return(df)

def hashin_ft(sigma_11, sigma_12, X_T, S_L, alpha = 1.0):
    # fibre tension failure criterion
    FT = np.zeros(sigma_11.shape[0])
    # Only relevant for positive fibre direction stress
    ft_ind = sigma_11 >= 0
    FT[ft_ind] = (sigma_11[ft_ind]/X_T)**2 + alpha*(sigma_12[ft_ind]/S_L)**2

    return FT

def hashin_fc(sigma_11, X_C):
    # fibre compression failure criterion
    FC = np.zeros(sigma_11.shape[0])
    # Only relevant for negative fibre direction stress
    fc_ind = sigma_11 < 0
    FC[fc_ind] = (sigma_11[fc_ind]/X_C)**2

    return FC

def hashin_mt(sigma_22, sigma_12, Y_T, S_L):
    # matrix tension failure criterion
    MT = np.zeros(sigma_22.shape[0])
    # Only relevant for positive matrix direction stress
    mt_ind = sigma_22 >= 0
    MT[mt_ind] = (sigma_22[mt_ind]/Y_T)**2 + (sigma_12[mt_ind]/S_L)**2

    return MT

def hashin_mc(sigma_22, sigma_12, Y_C, S_L, S_T):
    # matrix compression failure criterion
    MC = np.zeros(sigma_22.shape[0])
    # Only relevant for negative matrix direction stress
    mc_ind = sigma_22 < 0
    MC[mc_ind] = (sigma_22[mc_ind]/(2*S_T))**2 + ((Y_C/(2*S_T))**2 - 1)*(sigma_22[mc_ind]/Y_C) + (sigma_12[mc_ind]/S_L)**2

    return MC

def failure_f3(sigma_33, sigma_13, sigma_23, Y_3T, S_L, S_T):
    # Through-thickness failure criterion
    # Calculate regardless, but only use normal stress if in tension
    # Would be more well-behaved if treated tension and compression as separate failure indices as in Hashin
    F3 = np.zeros(sigma_33.shape[0])
    f3t_ind = sigma_33 >= 0
    # Contribution from elements in through-thickness tension
    F3[f3t_ind] = (sigma_33[f3t_ind]/Y_3T)**2 + (sigma_13[f3t_ind]/S_L)**2 + (sigma_23[f3t_ind]/S_T)**2
    # Otherwise
    F3[~f3t_ind] = (sigma_13[~f3t_ind]/S_L)**2 + (sigma_23[~f3t_ind]/S_T)**2

    F3 = np.sqrt(F3) # The function will be better behaved if I lose the sqrt, and still valid as failure surface is at F33 = F33^2 = 1

    return(F3)

def hashin_post(in_folder, n_subdomains, n_iter, X_T, X_C, Y_T, Y_C, S_L, S_T, Y_3T):
    # Load in all data from vtk files in in_folder, containing output across n_subdomains subdomains,
    # from n_iter iterations, and find the critical failure index in each Hashin failure mode
    # (plus extra through-thickness criterion)
    
    out_dict = {"FI_max" : [], "f_crit_sub" : []}
    # Loop over iterations
     
    for i in range(n_iter):
        if i == 0:
            FI_max = np.zeros(5)
            update_FI = True
        else:
            # Failure index must be monotonic - check against highest values from previous increment
            FI_max = copy.copy(out_dict["FI_max"][-1])
            update_FI = False # Has failure index been updated?

        f_crit_sub = np.zeros(5,dtype = int)
        for j in range(n_subdomains):
            filename = str(j)+'_NonLinear_iter-'+str(i)+'.vtk'
            src = os.path.join(os.getcwd(),in_folder,filename)
            # Load in stress and marker (indicating element type) from vtk
            df = ascii_vtk_to_df_stress(src)

            # Get index of elements which are composite (not cohesive zone or steel)
            comp_ind = df[:,0].astype(int) == 1
            df_comp = df[comp_ind,1:] # Discard marker

            # Calculate failure indices for composite elements
            FI = np.zeros((df.shape[0], 5))
            FI[comp_ind,0] = hashin_ft(df_comp[:,0], df_comp[:,3], X_T, S_L)
            FI[comp_ind,1] = hashin_fc(df_comp[:,0], X_C)
            FI[comp_ind,2] = hashin_mt(df_comp[:,1], df_comp[:,3], Y_T, S_L)
            FI[comp_ind,3] = hashin_mc(df_comp[:,1], df_comp[:,3], Y_C, S_L, S_T)
            FI[comp_ind,4] = failure_f3(df_comp[:,2], df_comp[:,5], df_comp[:,4], Y_3T, S_L, S_T)
            
            # Update critical faiure index value if the maximum value in the current subdomain is higher
            for k, column in enumerate(FI.T):
                if np.max(column) > FI_max[k]:
                    update_FI = True
                    FI_max[k] = np.max(column)
                    f_crit_sub[k] = j

        # Store critical values
        if not update_FI:
            # If the failure index hasn't been updated, preserve the previous critical subdomain
            out_dict["f_crit_sub"] = out_dict["f_crit_sub"][-1]
        out_dict["FI_max"].append(FI_max)
        out_dict["f_crit_sub"].append(f_crit_sub)

        out_frame = pd.concat((pd.DataFrame(out_dict["FI_max"], columns=["Max_F_FT","Max_F_FC","Max_F_MT","Max_F_MC","Max_F_33"]),
                              pd.DataFrame(out_dict["f_crit_sub"], columns=["ft_crit_sub","fc_crit_sub","mt_crit_sub","mc_crit_sub","f3_crit_sub"])),axis=1)
    
    return(out_frame)

if __name__ == "__main__":
    # Define properties
    n_iter = 24        # Number of iterations
    n_subdomains = 320 # Number of subdomains
    # Specify folder where output is stored
    in_folder = "C:/Users/cs2361/Documents/Jean_Benezech/DUNE-TEMP/CS04D_41-80/Data_CerTest_digiLab_2_hamilton074/"

    # Define material properties
    X_T = 2558
    X_C = 1690
    Y_T = 110
    Y_C = 285.7
    S_L = 112.7 # Fibre-direction shear stress
    Y_3T = 110 # Same as in-plane transverse direction
    S_T = S_L # Assume equal to longitudnal version (mechanics are different but this is a common assumption)

    out_frame = hashin_post(in_folder, n_subdomains, n_iter, X_T, X_C, Y_T, Y_C, S_L, S_T, Y_3T)
    out_frame.to_csv(in_folder + "_hashin.csv", index=False)