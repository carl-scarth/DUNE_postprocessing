import matplotlib.pyplot as plt
# Object for storing plots of curves data generating across samples

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