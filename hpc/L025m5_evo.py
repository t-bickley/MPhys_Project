
import os
import sys
from pathlib import Path
import pandas as pd

from swiftgalaxy import SWIFTGalaxy, SOAP, MaskCollection, SWIFTGalaxies
from scipy.spatial.transform import Rotation
from scipy.ndimage import rotate 
from swiftsimio.visualisation import project_gas, project_pixel_grid
import matplotlib.patches as patches
import pandas as pd

import numpy as np, scipy, matplotlib.pyplot as plt, unyt as u, swiftsimio as sw

# -----------
#This script is used to track the evolution in the L025m5 sample
#It makes use of the modified rotational alignment that uses the gas disk angular momentum rather than
#stellar angular momentum

#Very similar to aniso_track_L025m5.py in terms of retrieving other properties of the same halo
#Anisotropies are different!
#------------

# Define Imaging Function for Gas Density Projecton (the same used in the module folder however with different applications)
def sg_img(sg):


    # The area of projection (1.5x multiplier to ensure there is space around the edges to allow for further rotations)
    L_kpc = 1.5*sg.halo_catalogue.spherical_overdensity_200_crit.soradius.to('kpc').value #1.5 to ensure we capture all gas within r200c



    return  project_gas(
    sg,
    resolution=512,
    region=sw.cosmo_array(
        [-L_kpc, L_kpc , -L_kpc, L_kpc],
        u.kpc,
        comoving=True,
        scale_factor=sg.metadata.a,
        scale_exponent=1
    ),
    project='masses',
    parallel=False,
    periodic=True,

)

# Define Anisotropy Calculation Function (the same used in the module folder however with different applications)
def pixel_anisotropy(img, res=512): 

    #Rotation to align quadrants, allows for simpler summations
    mat = rotate(img.T, angle=45, reshape=False, order=3) 
    
    # Mark the boundaries of the quadrants 
    mid = res // 2
    
    # NumPy slicing to sum quadrants instantly
    # [row_start:row_end, col_start:col_end]
    q0 = mat[:mid, :mid].sum()   # Top-left
    q1 = mat[mid:, :mid].sum()   # Bottom-left
    q2 = mat[:mid, mid:].sum()   # Top-right
    q3 = mat[mid:, mid:].sum()   # Bottom-right
    
    
    mass_major = q1 + q2
    mass_minor = q0 + q3
    
    if mass_major == 0:
        return np.nan # prevent zero division
    

    return mass_minor / mass_major


# Define Masking Function, uses the Angular Momentum vector of the gas disk rather than stellar angular momentum
# This is to avoid misalignment issues caused mainly due to mergers within the temporal evolution of a halo



def mask_edge_on_gas(sg): 

   
    r = sg.gas.spherical_coordinates.r
    r200 = sg.halo_catalogue.spherical_overdensity_200_crit.soradius
    ang_mom_vec = sg.halo_catalogue.exclusive_sphere_10kpc.angular_momentum_gas #gas disk angular momentum

   

    group_nr = sg.gas.group_nr_bound
    mask = MaskCollection(
        
        #Radial Cut to isolate CGM gas
        #Retrieve only gas particles that are only bound to the central halo, or unbound (removes satellite gas)
        gas= (r > 0.25*r200) & (r < r200) & ((group_nr == -1) | (group_nr == sg.halo_catalogue.input_halos.halo_catalogue_index))
        
    )

    sg.mask_particles(mask)

    # Define a new coordinate system from the angular momentum vector
    ang_mom_vec = ang_mom_vec.squeeze()

    zhat = (ang_mom_vec / np.sqrt(np.sum(ang_mom_vec**2))).to_value(u.dimensionless)  # we'll align L with the z-axis

    arb = np.ones(3) / np.sqrt(3)  # Take an arbitrary vector
    xvec = arb - arb.dot(zhat) * zhat # Make it orthogonal to zhat
    xhat = xvec / np.sqrt(np.sum(xvec**2)) # normalise it

    yhat = np.cross(zhat, xhat)  # Get the unit vector orthogonal to both xhat and zhat

   
    rotmat = np.vstack((xhat, -zhat, yhat)).T ##Edge-on sidewards
    
    sg.rotate(Rotation.from_matrix(rotmat))


# New analysis function that incorporates the rotational alignment based on the gas disk
def analysis_anisotropy_gas(sg):

    #Physical Properties of the Halo
    g_mass = sg.halo_catalogue.spherical_overdensity_200_crit.gas_mass.to('Msun').value[0]
    t_mass = sg.halo_catalogue.spherical_overdensity_200_crit.total_mass.to('Msun').value[0]

    # Skip Halo since it has no mass

    if t_mass == 0:
        return None

    # Skip Halo since it has no gas
    if g_mass == 0: # or alternatively check len(sg.gas) 
        return None  # Skip, nothing to image
    


    s_count = sg.halo_catalogue.exclusive_sphere_10kpc.number_of_star_particles.value[0]
    if s_count == 0:
        return None  # Skip, cannot calculate stellar angular momentum
    

    #gas particle count (back up to gas mass)
    g_count = sg.halo_catalogue.spherical_overdensity_200_crit.number_of_gas_particles.value[0]

    if g_count == 0:
        return None #skip, nothing to image


    gas_frac = g_mass / t_mass
    mmbh_mass = sg.halo_catalogue.spherical_overdensity_200_crit.most_massive_black_hole_mass.to('Msun').value[0]
    stellar_corot = sg.halo_catalogue.exclusive_sphere_30kpc.kappa_corot_stars.value[0]

    

    #Anisotropy
    mask_edge_on_gas(sg)

    #Number of gas particles in CGM (after masking)
    cgm_count = len(sg.gas.coordinates) 
    if cgm_count == 0:
        return None #skip, nothing to image
    

    img = sg_img(sg)

    aniso_val = pixel_anisotropy(img)

    return sg.halo_catalogue.input_halos_hbtplus.track_id.value[0], sg.halo_catalogue.soap_index, aniso_val, gas_frac, s_count, mmbh_mass, stellar_corot, t_mass, g_count, cgm_count

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


    # The two track IDs for the example haloes within the paper
    # To track other haloes of interest, simply add thir track IDs to the list below (validation will ensure theyre present in the snapshot)
    # If not present then the data is skipped without crashing the code


    halo_index_1 = np.where(cd.input_halos_hbtplus.track_id == 195)[0]
     
    halo_index_2 = np.where(cd.input_halos_hbtplus.track_id == 53330)[0]

    soap_indices = np.concatenate([halo_index_1, halo_index_2]).tolist()

    if len(soap_indices) > 0:

        #Initialise all haloes in the snapshot with the associated soap indices
        sgs = SWIFTGalaxies(snap_files[task_id], SOAP(cat_files[task_id], soap_index = soap_indices,extra_mask=None))


        #Perform analysis and save results
        data = sgs.map(analysis_anisotropy_gas) 
        
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

            df.to_csv(f'results/L025m5_updated/example_haloes_gas_density_anisotropies_{task_id}.csv', index=False) 
        else:
            print(f'Task {task_id}: No valid data to save.') #Data validation 
    
    else:
        print(f'Task {task_id}: No requested haloes exist within this snapshot.') #Data Validation