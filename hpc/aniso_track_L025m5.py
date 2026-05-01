# hpc/compute_anisotropies.py
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import swiftsimio as sw
from swiftgalaxy import SWIFTGalaxies, SOAP



# New modules folder
sys.path.append(str(Path(__file__).parent.parent))
from modules.physics_utils import analysis_anisotropy


#---------------
#The original script used to track halo properties and compute anisotropes for the L025m5 sample

#However the script was found to be subject to a halo alignment issue where mergers affected alignent using stellar AM

#The script was modified to use gas AM for alignment and can be found in L025m5_evo.py

#This script is left here for reference
#---------------

if __name__ == "__main__":

    # Snapshot and Catalogue File Paths
    snap_files = 'snapshots_list.txt'
    cat_files = 'catalogues_list.txt'

    ## Read in file paths from text files
    with open(snap_files, 'r') as sf:
        snap_files = [line.strip() for line in sf.readlines()]
    with open(cat_files, 'r') as cf:
        cat_files = [line.strip() for line in cf.readlines()]


    #Task ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID',0))
    
    # load catalogue for current task
    cd = sw.load(cat_files[task_id])
 
    #load tracking ids from z = 0
    df = pd.read_csv('results/l25m5_track_ids.csv')
    ids = df['track_id'].values.astype(np.int64)

    #Retrieve associated soap indices for tracking ids
    mask = np.isin(cd.input_halos_hbtplus.track_id, ids)
    soap_indices = np.flatnonzero(mask)


    if len(soap_indices) > 0:

        #Initialise all haloes in the snapshot with the associated soap indices
        sgs = SWIFTGalaxies(snap_files[task_id], SOAP(cat_files[task_id], soap_index = soap_indices.flatten(),extra_mask=None))


        #Perform analysis and save results
        data = sgs.map(analysis_anisotropy) 
        
        #Remove skipped haloes
        valid_data = [d for d in data if d is not None]
        if valid_data:

            df = pd.DataFrame(valid_data, columns=['track_id','halo_index', 'mass_aniso', 'gas_frac', 'star_count', 'mmbh_mass', 'stellar_corot','mass','gas_count','cgm_gas_count']) 
            
            cd = sw.load(cat_files[task_id])

            z = cd.metadata.redshift
            a = cd.metadata.a
            age = cd.metadata.cosmology.age(z).value #Age Gyr

            df['age_Gyr'] = age
            df['z'] = z
            df['a'] = a

            df.to_csv(f'results/L025m5_updated/gas_density_anisotropies_{task_id}.csv', index=False) #Save results to CSV for analysisS
        else:
            print(f'Task {task_id}: No valid data to save.')
    
    else:
        print(f'Task {task_id}: No requested haloes exist within this snapshot.')