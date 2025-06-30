import vtk
import pandas as pd
import numpy as np
import os

def vtk_to_df_cell(vtk_file):
    # Read the VTK file and extract cell data

    """ # Initialise vtk reader object
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_file)
    reader.ReadAllScalarsOn() # Need this or only reads in the first output
    reader.Update()
    
    # Read data
    data = reader.GetOutput() """
    data = get_vtk_data(vtk_file)

    # Extract cells and data
    cells = data.GetCells()
    cell_data = data.GetCellData()
    n_cells = cells.GetNumberOfCells()
    
    # Extract all output data and convert to DataFrame
    df = vtk_data_to_df(cell_data, n_cells)
    
    # Rename stress and strain columns of dataframe
    index_mapping = ["11","22","33","12","23","13"] # Mapping of column index to tensor component index)
    df.rename(columns={"LOCAL_STRESS_"+str(i) : "STRESS_"+ind for i, ind in enumerate(index_mapping)},inplace=True)
    df.rename(columns={"STRAIN_"+str(i) : "STRAIN_"+ind for i, ind in enumerate(index_mapping)},inplace=True)

    return df

def vtk_to_df_stress(vtk_file, output_np = False):
    # Read the VTK file and extract cell data
    # Extracts only the stresses (and MARKER) for efficiency
    # If output_np outputs only numpy array, no column labels
    # Initialise vtk reader object
    data = get_vtk_data(vtk_file)

    # Extract cells and data
    cells = data.GetCells()
    cell_data = data.GetCellData()
    n_cells = cells.GetNumberOfCells()

    # Extract all output data and convert to DataFrame
    df = vtk_data_to_df(cell_data, n_cells, out_labels = ["MARKER","LOCAL_STRESS"], output_np=output_np)
    
    if not output_np:
        # Rename stress and strain columns of dataframe
        index_mapping = ["11","22","33","12","23","13"] # Mapping of column index to tensor component index)
        df.rename(columns={"LOCAL_STRESS_"+str(i) : "STRESS_"+ind for i, ind in enumerate(index_mapping)},inplace=True)
        df.rename(columns={"STRAIN_"+str(i) : "STRAIN_"+ind for i, ind in enumerate(index_mapping)},inplace=True)

    return df

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

def vtk_to_df_point(vtk_file):
    # Read the VTK file and extract point data
    # (not used - but keep in case I need to extract displacements later)
    
    """ # Initialise vtk reader object
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_file)
    reader.ReadAllScalarsOn() # Need this or only reads in the first output
    reader.Update()

    # Read data
    data = reader.GetOutput() """
    data = get_vtk_data(vtk_file)

    # Extract points and data
    points = data.GetPoints()
    point_data = data.GetPointData()
    n_points = points.GetNumberOfPoints()

    # Get point coordinates
    point_coords = []
    for i in range(n_points):
        point = points.GetPoint(i)  # Returns a tuple of (x, y, z)
        point_coords.append(point)

    point_coords = pd.DataFrame(point_coords, columns=['X', 'Y', 'Z'])
    
    # Extract all output data and convert to DataFrame
    output_df = vtk_data_to_df(point_data, n_points)

    # Combine with coordinates and return
    df = pd.concat((point_coords, output_df),axis=1)
    return df

def get_vtk_data(vtk_file):
    # Initialise vtk reader object
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_file)
    reader.ReadAllScalarsOn() # Need this or only reads in the first output
    reader.Update()
    # Read data
    data = reader.GetOutput()
    
    return(data)

def vtk_data_to_df(data, n_rows, out_labels = [], output_np = False):
    # Takes vtk data object (CellData or PointData) with unspecified number of outputs 
    # arrays with differing number of columns but fixed number of rows n_rows, and 
    # converts to dataframe
    # If out_labels provided only outputs data with matching label
    # If output_np only outputs numpy array

    n_arrays = data.GetNumberOfArrays()

    # Get data and output name from each array and structure as dictionary
    data_dict = {}
    for i in range(n_arrays):
        array_name = data.GetArrayName(i)
        # Has a list of outputs been requested
        if out_labels:
            if array_name in out_labels:
                array = data.GetArray(i)
                # Number of entries can vary depending upon the output, reshape accordingly
                array_data = np.array([array.GetTuple(j) for j in range(n_rows)]).reshape([n_rows,-1])
                data_dict[array_name] = array_data
        # Otherwise just output everything
        else:
            array = data.GetArray(i)
            # Number of entries can vary depending upon the output, reshape accordingly
            array_data = np.array([array.GetTuple(j) for j in range(n_rows)]).reshape([n_rows,-1])
            data_dict[array_name] = array_data
            
    if output_np:
        # Only using this when outputting only marker and stress
        df = np.concatenate((data_dict["MARKER"],data_dict["LOCAL_STRESS"]), axis=1)
    else:
        # Convert to pandas DataFrame
        df = pd.DataFrame()
        for key, value in data_dict.items():
            # Format column headings if there are multiple output components
            if value.shape[1] > 1:
                col_names = [key+"_"+str(i) for i in range(value.shape[1])]
            else:
                col_names = key
        
            df[col_names] = pd.DataFrame(value)

    return df

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

def hashin_post(in_folder, n_subdomains, n_iter, X_T, X_C, Y_T, Y_C, S_L, S_T, Y_3T, debug = False):
    # Load in all data from vtk files in in_folder, containing output across n_subdomains subdomains,
    # from n_iter iterations, and find the critical failure index in each Hashin failure mode
    # (plus extra through-thickness criterion)
    
    # Output everything (particularly failure indices) if debugging, and work in Pandas for readibility. 
    # Otherwise just ouput MARKER and STRESS and work in numpy for speed (save time not converting backward
    # and forward)
    if debug:
        # Initialise dictionary for storing output
        out_dict = {"ft_max" : [], "ft_crit_sub" : [], "ft_crit_el" : [],
                "fc_max" : [], "fc_crit_sub" : [], "fc_crit_el" : [],
                "mt_max" : [], "mt_crit_sub" : [], "mt_crit_el" : [],
                "mc_max" : [], "mc_crit_sub" : [], "mc_crit_el" : [],
                "f3_max" : [], "f3_crit_sub" : [], "f3_crit_el" : []}
        for i in range(n_iter):
            #print("Iteration " + str(i))
            # Initialise variables for tracking maximum failure index values
            ft_max, fc_max, mt_max, mc_max, f3_max = 0.0, 0.0, 0.0, 0.0, 0.0
            ft_crit_sub, fc_crit_sub, mt_crit_sub, mc_crit_sub, f3_crit_sub = None, None, None, None, None # Critcial subdomain
            ft_crit_el, fc_crit_el, mt_crit_el, mc_crit_el, f3_crit_el = None, None, None, None, None # Critical element
            for j in range(n_subdomains):
                filename = str(j)+'_NonLinear_iter-'+str(i)+'.vtk'
                src = os.path.join(os.getcwd(),in_folder,filename)
                df = vtk_to_df_cell(src)
                # Get index of elements which are composite (not cohesive zone or steel)
                df["MARKER"] = df["MARKER"].astype(int)
                    
                # Extract only composite elements
                comp_ind = df["MARKER"] == 1
                df_comp = df[comp_ind]
                    
                # Calculate failure indices for composite elements
                F_ft, F_fc, F_mt, F_mc, F_3 = (np.zeros(df.shape[0]) for k in range(5))
                F_ft[comp_ind] = hashin_ft(df_comp["STRESS_11"].values, df_comp["STRESS_12"].values, X_T, S_L)
                F_fc[comp_ind] = hashin_fc(df_comp["STRESS_11"].values, X_C)
                F_mt[comp_ind] = hashin_mt(df_comp["STRESS_22"].values,df_comp["STRESS_12"].values, Y_T, S_L)
                F_mc[comp_ind] = hashin_mc(df_comp["STRESS_22"].values,df_comp["STRESS_12"].values, Y_C, S_L, S_T)
                F_3[comp_ind] = failure_f3(df_comp["STRESS_33"].values, df_comp["STRESS_13"].values, df_comp["STRESS_23"].values, Y_3T, S_L, S_T)

                # Update critical faiure index value if the maximum value in the current subdomain is higher
                if np.max(F_ft) > ft_max:
                    ft_max = np.max(F_ft)
                    ft_crit_sub = j
                    ft_crit_el = np.argmax(F_ft)
                if np.max(F_fc) > fc_max:
                    fc_max = np.max(F_fc)
                    fc_crit_sub = j
                    fc_crit_el = np.argmax(F_fc)
                if np.max(F_mt) > mt_max:
                    mt_max = np.max(F_mt)
                    mt_crit_sub = j
                    mt_crit_el = np.argmax(F_mt)
                if np.max(F_mc) > mc_max:
                    mc_max = np.max(F_mc)
                    mc_crit_sub = j
                    mc_crit_el = np.argmax(F_mc)
                if np.max(F_3) > f3_max:
                    f3_max = np.max(F_3)
                    f3_crit_sub = j
                    f3_crit_el = np.argmax(F_3)

            # Store critical values
            out_dict["ft_max"].append(ft_max)
            out_dict["fc_max"].append(fc_max)
            out_dict["mt_max"].append(mt_max)
            out_dict["mc_max"].append(mc_max)
            out_dict["f3_max"].append(f3_max)
            out_dict["ft_crit_sub"].append(ft_crit_sub)
            out_dict["fc_crit_sub"].append(fc_crit_sub)
            out_dict["mt_crit_sub"].append(mt_crit_sub)
            out_dict["mc_crit_sub"].append(mc_crit_sub)
            out_dict["f3_crit_sub"].append(f3_crit_sub)
            out_dict["ft_crit_el"].append(ft_crit_el)
            out_dict["fc_crit_el"].append(fc_crit_el)
            out_dict["mt_crit_el"].append(mt_crit_el)
            out_dict["mc_crit_el"].append(mc_crit_el)
            out_dict["f3_crit_el"].append(f3_crit_el)
    
        out_frame = pd.DataFrame(out_dict)
    else: 
        out_dict = {"FI_max" : [], "f_crit_sub" : []}#, "f_crit_el" : []}
        for i in range(n_iter):
            #print("Iteration " + str(i))
            # Initialise variables for tracking maximum failure index values
            #tic = time.time()
            
            FI_max = np.zeros(5)
            f_crit_sub = np.zeros(5,dtype = int)
            #tic = time.time()
            for j in range(n_subdomains):
                filename = str(j)+'_NonLinear_iter-'+str(i)+'.vtk'
                src = os.path.join(os.getcwd(),in_folder,filename)
                # Numpy version of code
                #df = vtk_to_df_stress(src, output_np=True)
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
                        FI_max[k] = np.max(column)
                        f_crit_sub[k] = j
                        #f_crit_el[k] = np.argmax(column)

            #toc = time.time() - tic            
            #print(toc)
            #print("Numpy " + str(toc) + " seconds")
            # Could speed up further by not outputting critical subdomain etc

            # Store critical values
            out_dict["FI_max"].append(FI_max)
            out_dict["f_crit_sub"].append(f_crit_sub)
            #out_dict["f_crit_el"].append(f_crit_el)

        out_frame = pd.concat((pd.DataFrame(out_dict["FI_max"], columns=["Max_F_FT","Max_F_FC","Max_F_MT","Max_F_MC","Max_F_33"]),
                              pd.DataFrame(out_dict["f_crit_sub"], columns=["ft_crit_sub","fc_crit_sub","mt_crit_sub","mc_crit_sub","f3_crit_sub"])),axis=1)    
    
    return(out_frame)


if __name__ == "__main__":

    # Make sure I treat S_T as separate input

    # Define properties

    n_iter = 19
    n_subdomains = 320 # There might be an issue if this changes across the samples. Perhaps detect automatically...
    # Specify folder where output is stored
    in_folder = "C:/Users/cs2361/Documents/Jean_Benezech/DUNE-TEMP/CS04D_41-80/Data_CerTest_digiLab_2_hamilton041/"
    #in_folder = "D:/DUNE/CS04D_41-80/Data_CerTest_digiLab_2_hamilton041/"

    # Define material properties
    X_T = 2558
    X_C = 1690
    Y_T = 110
    Y_C = 285.7
    S_L = 112.7 # Fibre-direction shear stress
    Y_3T = 110 # Same as in-plane transverse direction
    # Transverse shear strain - I think this is impossible to measure.
    S_T = S_L # Assume equal to longitudnal version (mechanics are different but this is a common assumption - Meng Yi uses this) Also Jean in his through-thickness criterion    
    # From Jean's code (in-situ transverse shear strength from Camanho et al e.g. Prediction of size effects in notched laminates using continuum damage mechanics)
    #alpha0 = 0.925 # Fracture angle of a ply under transverse compression (53 degrees from the above paper)
    #S_T = Y_C*np.cos(alpha0)*(np.sin(alpha0) + np.cos(alpha0)/np.tan(2*alpha0))

    debug = False

    out_frame = hashin_post(in_folder, n_subdomains, n_iter, X_T, X_C, Y_T, Y_C, S_L, S_T, Y_3T, debug = debug)
    out_frame.to_csv(in_folder + "_hashin.csv", index=False)
    