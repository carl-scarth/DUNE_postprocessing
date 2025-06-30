import pandas as pd
import numpy as np

def load_curves(src):
    # Load in Jean's curves data
    print(src)
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