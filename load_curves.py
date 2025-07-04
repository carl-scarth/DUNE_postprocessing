import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

src_path = "../../Bayes_Cantilever_Calibration/python/src/"
sys.path.insert(0, src_path)

from utils import set_plot_params

class CurvesPlot:
    # For different curves plots
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax2 = []
        self.new_ax = True
        self.new_twinax = True
        self.lines = []

    def twinax(self): 
        # Duplicate x axis if comparing two types of plot
        self.ax2 = self.ax.twinx()

    def plot_fd(self,curves_data, force_col, disp_col, color_list = [], **kwargs):
        # Create force-displacement plot
        # Displacement column can be passed as a single string or as a list
        if isinstance(disp_col, list):
            if color_list:
                [self.lines.extend(self.ax.plot(curves_data[col].values, curves_data[force_col].values,"-", label = col, color = color, **kwargs)) for col, color in zip(disp_col, color_list)]
            else:
                [self.lines.extend(self.ax.plot(curves_data[col].values, curves_data[force_col].values,"-", label = col, **kwargs)) for col in disp_col]
        else:
            self.lines.extend(self.ax.plot(curves_data[disp_col].values, curves_data[force_col].values,"-", label = disp_col, **kwargs))
        
        if self.new_ax:
            self.ax.set_xlabel("Displacement (mm)")
            self.ax.set_ylabel("Force (kN)")
        
    def plot_failure_ind(self, curves_data, x_col, failure_col, failure_load = None, twin_ax = False, color_list = [], **kwargs):
        # Plot failure index
        # Failure index column can be passed as single string or as a list
        # x_col indicates columns containing x axis
        if twin_ax:
            if self.new_twinax:
                self.twinax()
                self.new_twinax = False
                self.ax2.set_ylabel("Failure Index")
        elif self.new_ax:
            self.ax.set_ylabel("Failure Index")
        if self.new_ax: # Needs to be separate to above loop as true regardless of which y axis is used
            self.ax.set_xlabel(x_col)
            self.new_ax = False

        if isinstance(failure_col, list):
            if color_list:
                if twin_ax:
                    [self.lines.extend(self.ax2.plot(curves_data[x_col].values, curves_data[col].values,"-", color = color, label = failure_col, **kwargs)) for col, color in zip(failure_col, color_list)]
                else:
                    [self.lines.extend(self.ax.plot(curves_data[x_col].values, curves_data[col].values,"-", color = color, label = failure_col, **kwargs)) for col, color in zip(failure_col, color_list)]
            else:
                for i, col in enumerate(failure_col):
                    if twin_ax:
                        [self.lines.extend(self.ax2.plot(curves_data[x_col].values, curves_data[col].values,"-", label = failure_col, **kwargs)) for col in failure_col]
                    else:
                        [self.lines.extend(self.ax.plot(curves_data[x_col].values, curves_data[col].values,"-", label = failure_col, **kwargs)) for col in failure_col]
                        if failure_load:
                            self.ax.plot(failure_load[col],1.0,'kx')
        else:
            if twin_ax:
                self.lines.extend(self.ax2.plot(curves_data[x_col].values, curves_data[failure_col].values,"-", label = failure_col, **kwargs))
            else: 
                self.lines.extend(self.ax.plot(curves_data[x_col].values, curves_data[failure_col].values,"-", label = failure_col, **kwargs))
                if failure_load is not None:
                    self.ax.plot(failure_load[failure_col],1.0,'x', **kwargs)


    def plot_all_curves(self, curves_data, force_col, disp_col, failure_col, **kwargs):
        self.plot_fd(curves_data, force_col, disp_col, **kwargs)
        if isinstance(disp_col, list):
            disp_col = disp_col[0]
        self.plot_failure_ind(curves_data, disp_col, failure_col, twin_ax = True, **kwargs)
        
    def plot_limit_state(self):
        # Plot line across y-axis at failure_ind = 1.0
        if self.new_twinax:
            self.ax.plot(self.ax.get_xlim(),[1,1],"--k")
        else:
            self.ax2.plot(self.ax2.get_xlim(),[1,1],"--k")

    def set_xlim(self, xlim):
            self.ax.set_xlim(xlim)

    def set_ylim(self, ylim, twin_ax = False):
        if twin_ax:
            self.ax2.set_ylim(ylim)
        else:
            self.ax.set_ylim(ylim)

    def set_xlabel(self, label):
        self.ax.set_xlabel(label)

    def set_ylabel(self, label, twin_ax = False):
        if twin_ax:
            self.ax2.set_ylabel(label)
        else:
            self.ax.set_ylabel(label)

    def set_title(self, title, twin_ax = False, **kwargs):
        if twin_ax:
            self.ax2.set_title(title, **kwargs)
        else:
            self.ax.set_title(title, **kwargs)
        
    def plot_legend(self, simplify_labels = True, **kwargs):
        labels = [line.get_label() for line in self.lines]
        unique_labels = list(set(labels))
        if simplify_labels:
            labels_leg = unique_labels
            lines_leg = [[line for line in self.lines if line.get_label() == label][0] for label in labels_leg]
        else:
            labels_leg = []
            lines_leg = []
            for label in unique_labels:
                lines_i = [(lab, line) for lab, line in zip(labels, self.lines) if lab == label]
                labels_leg.extend(["_".join((line[0], str(i))) for i, line in enumerate(lines_i)])
                lines_leg.extend([line[1] for line in lines_i])

        if self.new_twinax:
            self.ax.legend(lines_leg, labels_leg, **kwargs)
        else:
            self.ax2.legend(lines_leg, labels_leg, **kwargs)
        

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
    # Default names of columns containing failure indices
    
    #failure_load = np.empty(len(failure_cols))
    failure_load = []
    # Loop over each failure mode
    for i, col in enumerate(failure_cols):
        # Determine if failure occurs in this mode
        print(col)
        failure_init = curves_data[col]>=1.0
        if failure_init.any():
            # If failure occurs - find iteration at which failure occurs
            # Linearly interpolate failure load from previous iteration
            # (It may be more appropriate to linearly interpolate over
            # the square root as the failure indices are (mostly) quadratic)
            failure_ind = curves_data[failure_init].iloc[0].name
            postfailure_load = curves_data.iloc[failure_ind][load_col]
            prefailure_load = curves_data.iloc[failure_ind-1][load_col]
            postfailure_fi = curves_data.iloc[failure_ind][col]
            prefailure_fi = curves_data.iloc[failure_ind-1][col]
            # Linear interpolate assuming failure happens at failure index of 1
            alpha = (1 - prefailure_fi)/(postfailure_fi - prefailure_fi)
            # failure_load[i] = prefailure_load + alpha*(postfailure_load - prefailure_load)
            failure_load.append(prefailure_load + alpha*(postfailure_load - prefailure_load))
        else:
            # Otherwise return a nan
            # failure_load[i] = (np.nan)
            failure_load.append(np.nan)
    #failure_load = pd.Series(failure_load, index = failure_cols)
    return(failure_load)


if __name__ == "__main__":

    set_plot_params()
    #plt.rcParams.update({'axes.prop_cycle' : plt.cycler(color=[plt.cm.tab20(i) for i in range(20)])})
    color_cycle = [plt.cm.tab20(i) for i in range(20)]#plt.cycler(color=[plt.cm.tab20(i) for i in range(20)])
    color_cycle = color_cycle*3
    #plt.rcsetup.cycler(color_cycle)
    #plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[plt.cm.tab20(i) for i in range(20)]) 


    from_csv = False # If reading from csv with some data processing already done
    # Location of curves files
    #in_folder = "../DUNE_curves/CS04D_41-80/"
    #in_folder = "../DUNE_curves/temp/"
    #in_folder = "delam_bounding/"#New_test_comp_UB/"
    in_folder = "../../digiLab/Archive3_post_holiday/Curves/"
    #in_csv = "hashin_example_UQ_1.csv"
    #in_csv2 = "hashin_example_UQ_2.csv"
    
    #in_folder = "New_test_condition/Curves/"

    # Which columns are of interest
    """ cols_qoi = ["applied_displacement",
                "applied_load",
                "Displacement_DIC[2]",
                "hashin_tension_fibre",
                "Max_F_FT",
                "hashin_compression_fibre",
                "Max_F_FC",
                "hashin_tension_matrix",
                "Max_F_MT",
                "hashin_compression_matrix",
                "Max_F_MC",
                "Max_F_33",
                "max_damage_cohesive",
                "Max_F_CZ"
                ] """
    cols_qoi = ["applied_displacement",
                "applied_load",
                "Displacement_DIC[2]",
                "Max_F_FT",
                "Max_F_FC",
                "Max_F_MT",
                "Max_F_MC",
                "Max_F_33",
                "max_damage_cohesive",
                "max_predamage_cohesive",
                "Max_F_CZ"
                ]
    
    # Columns containing failure indices
    """     cols_failure = ["hashin_tension_fibre",
                        "hashin_compression_fibre",
                        "hashin_tension_matrix",
                        "hashin_compression_matrix",
                        "Max_F_33",
                        "max_damage_cohesive",
                        "Max_F_CZ"]
    
    cols_failure = ["ft_max","fc_max","mt_max","mc_max"]#,"f3_max"] """
    
    cols_failure = ["Max_F_FT",
                "Max_F_FC",
                "Max_F_MT",
                "Max_F_MC",
                "Max_F_33",
                "max_damage_cohesive",
                "max_predamage_cohesive",
                "Max_F_CZ"
                ]
    
    # We won't need both the max_f and hashin options - 
    # Need to understand which is correct by calculating
    # separately

    failure_load = []
    if from_csv:
        # Bodge for now - add applied load colummn to uq code in future
        applied_load = [0, 22117.2, 44042.7, 65724.8, 87093.5, 108054, 128480, 148204, 167021, 200945, 228439, 248862, 256665, 260045, 263112, 265891, 268408, 270682, 271735]  
        curves_frame = pd.read_csv(in_csv)
        curves_frame2 = pd.read_csv(in_csv2)
        curves_frame2["Sample"] = curves_frame2["Sample"]+100
        curves_frame = pd.concat((curves_frame,curves_frame2),axis=0)
        
        for sample in curves_frame["Sample"].unique():
            curves_data = curves_frame[curves_frame["Sample"] == sample].sort_values("Increment")
            curves_data["applied_load"] = np.array(applied_load)
            curves_data.reset_index(inplace=True)
            failure_load.append(extract_failure_load(curves_data, cols_failure))
    else:
        i = 0
        for file in os.listdir(in_folder):
            if file.split(".")[-1] == "txt":
                curves_data = load_curves(os.path.join(in_folder,file))
                curves_data = curves_data[cols_qoi]
                curves_data["Increment"] = curves_data.index
                curves_data["Sample"] = i
                failure_load.append(extract_failure_load(curves_data, cols_failure))
                if i == 0:            
                    curves_frame = curves_data
                else:
                    curves_frame = pd.concat((curves_frame, curves_data), axis = 0)
                i += 1
    
    failure_load = pd.DataFrame(failure_load, columns = cols_failure)
    failure_load = failure_load/1000.0    

    crit_load = pd.concat((failure_load.min(axis=1), failure_load.idxmin(axis=1)), axis = 1)
    crit_load.columns = ["crit_failure_load","failure_mode"]
    
    print(failure_load)
    print(crit_load)

    #failure_load.hist()
    #crit_load.hist()
    #plt.show()
    #print(failure_load.describe())

    curves_frame.set_index(["Sample", "Increment"], inplace=True)
    pd.set_option('display.max_rows', 10000)
    curves_frame["applied_displacement"] = -curves_frame["applied_displacement"]
    curves_frame["applied_load"] = curves_frame["applied_load"]/1000.0
    
    force_disp = CurvesPlot()
    #failure_plot = CurvesPlot()
    failure_plot = CurvesPlot()
    failure_plot_ft = CurvesPlot()
    failure_plot_fc = CurvesPlot()
    failure_plot_mt = CurvesPlot()
    failure_plot_mc = CurvesPlot()
    failure_plot_f33 = CurvesPlot()
    failure_plot_dam = CurvesPlot()
    failure_plot_predam = CurvesPlot()


    max_disp = curves_frame["applied_displacement"].max()   
    max_disp_all = max(curves_frame["applied_displacement"].max(), curves_frame["Displacement_DIC[2]"].max())
    max_force = curves_frame["applied_load"].max()
    for sample in curves_frame.groupby(level=0):
        # Sample is a tuple, with entry 0 = sample #, entry 1 = pd.df

        force_disp.plot_fd(sample[1], "applied_load", "applied_displacement", color = color_cycle[sample[0]])
        failure_load_i = failure_load.iloc[sample[0]]
        # Only highlight failure load for critical value
        failure_load_i[failure_load_i.index != crit_load.iloc[sample[0]].failure_mode] = np.nan

        failure_plot_ft.plot_failure_ind(sample[1], "applied_load", "Max_F_FT", failure_load = failure_load_i, markersize = 12, color = color_cycle[sample[0]])
        failure_plot_fc.plot_failure_ind(sample[1], "applied_load", "Max_F_FC", failure_load = failure_load_i, markersize = 12, color = color_cycle[sample[0]])
        failure_plot_mt.plot_failure_ind(sample[1], "applied_load", "Max_F_MT", failure_load = failure_load_i, markersize = 12, color = color_cycle[sample[0]])
        failure_plot_mc.plot_failure_ind(sample[1], "applied_load", "Max_F_MC", failure_load = failure_load_i, markersize = 12, color = color_cycle[sample[0]])
        failure_plot_f33.plot_failure_ind(sample[1], "applied_load", "Max_F_33", failure_load = failure_load_i, markersize = 12, color = color_cycle[sample[0]])
        failure_plot_dam.plot_failure_ind(sample[1], "applied_load", "max_damage_cohesive", failure_load = failure_load_i, markersize = 12, color = color_cycle[sample[0]])
        failure_plot_predam.plot_failure_ind(sample[1], "applied_load", "max_predamage_cohesive", failure_load = failure_load_i, markersize = 12, color = color_cycle[sample[0]])


        #        force_disp.plot_fd(sample[1], "applied_load", ["applied_displacement", "Displacement_DIC[2]"], color_list = ["r","b"])
        #failure_plot.plot_failure_ind(sample[1], "applied_load", "hashin_tension_fibre", failure_load = failure_load_i, color = "r", markersize = 12)
        #failure_plot.plot_failure_ind(sample[1], "applied_load", "hashin_compression_fibre", failure_load = failure_load_i, color = "g", markersize = 12)
        #failure_plot.plot_failure_ind(sample[1], "applied_load", "hashin_compression_matrix", failure_load = failure_load_i, color = "b", markersize = 12)
        failure_plot.plot_failure_ind(sample[1], "applied_load", "Max_F_FT", failure_load = failure_load_i, color = "r", markersize = 12)
        failure_plot.plot_failure_ind(sample[1], "applied_load", "Max_F_FC", failure_load = failure_load_i, color = "g", markersize = 12)
        failure_plot.plot_failure_ind(sample[1], "applied_load", "Max_F_MT", failure_load = failure_load_i, color = "b", markersize = 12)
        failure_plot.plot_failure_ind(sample[1], "applied_load", "Max_F_MC", failure_load = failure_load_i, color = "c", markersize = 12)
        failure_plot.plot_failure_ind(sample[1], "applied_load", "Max_F_33", failure_load = failure_load_i, color = "y", markersize = 12)
        failure_plot.plot_failure_ind(sample[1], "applied_load", "max_damage_cohesive", failure_load = failure_load_i, color = "m", markersize = 12)
        failure_plot.plot_failure_ind(sample[1], "applied_load", "max_predamage_cohesive", failure_load = failure_load_i, color = "k", markersize = 12)


        # SOMETIMES ACTIVE - LOOKS LIKE THERE IS INCONSISTENT BEHAVIOUR - SOME HAVE VALUES IN ORDER 0, 1, OTHER IN THE ORDER OF 10S. COULD THIS BE A PERCENTAGE?
        # Problematic - could remove but this is the main failure mode here...
        # Spuriously high values of Max_F_CZ coming from boundary regions - need to ignore and only take values within certain region - won't be able to do
        # from curves files - this complicates things...
        # I think that there might be an issue with this one - high values aren't reflected in vtk files. Check code. Possible to post-process?


        # Is there are correlation between inputs - e.g. rig stiffness - does a low value lead to a stress concentration?
        #failure_plot.plot_failure_ind(sample[1], "applied_load", "Max_F_CZ", failure_load = failure_load_i, markersize = 12, color = "c")
        # NOT ACTIVE, BUT SUSPICIOUS - SEEMS TO MAX OUT BELOW 1z
        # failure_plot.plot_failure_ind(sample[1], "applied_load", "max_damage_cohesive", failure_load = failure_load_i, color = "m", markersize = 10)
        # NEVER ACTIVE
        # failure_plot.plot_failure_ind(sample[1], "applied_load", "hashin_tension_matrix", failure_load = failure_load_i, color = "y", markersize = 10)
        # failure_plot.plot_failure_ind(sample[1], "applied_load", "Max_F_33", failure_load = failure_load_i, color = "k", markersize = 10)

#        failure_plot.plot_failure_ind(sample[1], "applied_load", "hashin_tension_fibre")#, color = "c")
        #  if sample[0] >15 and sample[0] <= 19:
        #    force_disp.plot_fd(sample[1], "applied_load", "applied_displacement")
        #    failure_plot.plot_failure_ind(sample[1], "applied_load", "Max_F_CZ")#, color = "c")

        
    # force_disp.set_xlim([0, max_disp_all])
    force_disp.set_xlim([0, 4.5])
    force_disp.set_ylim([0, 335])
    #force_disp.plot_legend(simplify_labels=True)
    #force_disp.plot_legend(simplify_labels=False, loc = "center right")
    # failure_plot.set_xlim([0, max_force])
    #failure_plot.set_ylim([0, 1.75])
    failure_plot.set_xlim([0, 335])
    failure_plot.set_xlabel("Force (kN)")
    failure_plot.plot_limit_state()
    #failure_plot.plot_legend(simplify_labels=True)  

    plot_dict = {"Fibre tension" : failure_plot_ft,
                     "Fibre compression" : failure_plot_fc,
                     "Matrix tension" : failure_plot_mt,
                     "Matrix compression" : failure_plot_mc,
                     "Through-thickness" : failure_plot_f33,
                     "Cohesive non-delam" : failure_plot_dam,
                     "Cohesive delam" : failure_plot_predam}
    
    for key, plot in plot_dict.items():
        plot.set_xlim([0, 375])
        plot.set_xlabel("Force (kN)")
        plot.plot_limit_state()
        plot.plot_legend(simplify_labels=False, loc = "center left")
        plot.set_title(key)
    plt.show()
    EFAFDA
    # some samples stop prematurely amd no failure occurs
    # Max_F_CZ seems strange

#    plt.show()

    # Try to figure out weird behaviour in Max_F_CZ
    # plot 
    # Update digiLab
    # Work on python version of full-field code in background
    # Dig out 2020 BC Campllight album (and others) from desktop PC
    # Try to figure out stopping criteria - are some runs stopping prematurely - noise?s
    
    # Alternative frame

    alt_csv = "hashin_example_all.csv"
    alt_frame = pd.read_csv(alt_csv)
    alt_frame = alt_frame[["Sample","Increment","ft_max", "fc_max", "mt_max", "mc_max", "f3_max"]]
    alt_frame.set_index(["Sample", "Increment"], inplace=True)
    alt_frame.columns = ["hashin_tension_fibre","hashin_compression_fibre",
                        "hashin_tension_matrix",
                        "hashin_compression_matrix",
                        "Max_F_33"]
    
    
    print(curves_frame)
    print((curves_frame[["hashin_tension_fibre","hashin_compression_fibre",
                        "hashin_tension_matrix",
                        "hashin_compression_matrix",
                        "Max_F_33"]] - alt_frame).abs().max())
    # Some error on fibre tension for at least one of the samples - chase that up....
    asdsa
    #for column in alt_frame.columns.values:
        #alter
       
        # Will have to download the data for one case somewhere...

        # Here at least extract the simplest level
        # Critical load for each sample
        # Then also force displacement for each sample

        # Pack backs and tidy before doing any more work        

    

    fig3, ax3 = plt.subplots()
    lines2 = []
    lines2.extend(ax3.plot(curves_data["D_xmax[0]"].values, curves_data["applied_load"].values/1000.0,"-b", label="Midpoint displacement"))
    ax4 = ax3.twinx()
    for name,col_ind in failure_cols.items():
        lines2.extend(ax4.plot(curves_data["D_xmax[0]"].values, curves_data[col_ind].values, label=name))

    ax4.plot([0, curves_data["D_xmax[0]"].max()],[1,1],"--k")
    labels2 = [line.get_label() for line in lines2]
    ax3.set_xlabel("Out-of-plane displacement at midpoint")
    ax3.set_ylabel("Force (kN)")
    ax3.set_xlim([0, curves_data["D_xmax[0]"].max()])
    ax3.set_ylim([0, 280])
    ax4.set_ylim([0,1.3])
    ax4.legend(lines2, labels2)


    # Not sure which point this is - take some points on paraview to check
    # Trends make sense for longitudinal displacement, but not for the other components
    fig4, ax4 = plt.subplots()
    ax4.plot(curves_data["D_zmax[2]"].values, curves_data["applied_load"].values/1000.0, label="z_min")
    ax4.plot(curves_data["D_zmin[2]"].values, curves_data["applied_load"].values/1000.0, label="z_max")
    ax4.set_xlabel("Longitudinal displacement at endpoint")
    ax4.set_ylabel("Force (kN)")
    ax4.legend()

    fig5, ax5 = plt.subplots()
    for name,col_ind in failure_cols.items():
        ax5.plot(curves_data["applied_load"].values/1000.0, curves_data[col_ind].values, label=name)

    ax5.set_xlim([0, curves_data["applied_load"].max()/1000.0])
    ax5.set_ylim([0, 1.3])
    ax5.plot([0, curves_data["applied_load"].max()/1000.0],[1.0, 1.0],'--k')
    ax5.plot([prefailure_load/1000.0,prefailure_load/1000.0],[0, 1.3],'--g')
    ax5.plot([failure_load/1000.0,failure_load/1000.0],[0, 1.3],'--r')
    ax5.legend(loc = "upper left")
    ax5.set_xlabel("Force (kN)")
    ax5.set_ylabel("Failure index")

    print(prefailure_load)
    print(failure_load)
    print((failure_load-prefailure_load)/failure_load)

    plt.show()