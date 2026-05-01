# MPhys_Project

The code developed to investigate the circumgalactic medium of galactic haloes within the COLIBRE simulation suite.
As linked in the appendix of the report.

Notes: 

-HPC directory contains all the code that was used to run parallel tasks with SLURM and is where the bulk of the data was computed.
Data was saved and later analysed in the notebooks

-Modules contains functions that were used throughout the project.

-Notebooks contains all the important notebooks used for analysis and debugging/testing

-The root directory also contains the batch scripts used to run the slurm jobs. track_halos was changed to run 
L025m5_evo.py (it originally ran aniso_track_L025m5.py, the old parallel code)

-In L025m5_analysis.ipynb, the user is reminded to read the Note located at the top of the notebook

-The file paths that were used during the development of this code have been removed for privacy. However placeholders are commented
in their place making it clear which file was being accessed.
