# DUNE Post-processing scripts

Scripts for plottting and postprocessing output from DUNE. Contents include:
- **load_curves.py**: 
Loads in data from curves files, calculates and outputs the critical failure load and mode, and produces Force-displacement/Failure Index plots across a set of samples. To run curves files must be stored in "inputs/curves". Alternatively modify the location on Line 22. Calculated failure loads are outputted to "outputs/curves_outputs.csv".
- **hashin_UQ.py**: 
Runs the Hashin postprocessing script (src/hashin_post.py) using .vtk output from an intial set of sample runs of the DUNE model, for a new Latin Hypercube taken across the in-plane strength parameters to generate new ouput failure load values for the in-plane and through-thickness (intralaminar) modes. Assumes vtks are stored in "inputs/VTKs", outputs written under "outputs/".

Example input data has been provided to demonstrate usage.