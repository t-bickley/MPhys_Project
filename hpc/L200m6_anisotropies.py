import os, sys, argparse 

from swiftgalaxy import SWIFTGalaxy, SOAP, MaskCollection, SWIFTGalaxies
from scipy.spatial.transform import Rotation
from swiftsimio.visualisation import project_gas, project_pixel_grid
from scipy.ndimage import rotate 

import matplotlib.patches as patches
import pandas as pd

import numpy as np, scipy, matplotlib.pyplot as plt, unyt as u, swiftsimio as sw

def mask_edge_on(sg): 

   
    r = sg.gas.spherical_coordinates.r
    r200 = sg.halo_catalogue.spherical_overdensity_200_crit.soradius
    ang_mom_vec = sg.halo_catalogue.exclusive_sphere_10kpc.angular_momentum_stars

   

    group_nr = sg.gas.group_nr_bound
    mask = MaskCollection(
        #gas=sg.gas.spherical_coordinates.r < 50. * u.kpc, ##Change back to 50
        
        gas= (r > 0.25*r200) & (r < r200) & ((group_nr == -1) | (group_nr == sg.halo_catalogue.input_halos.halo_catalogue_index))
        #gas=  (r < r200) & ((group_nr == -1) | (group_nr == sg.halo_catalogue.input_halos.halo_catalogue_index)) #(include disk)
        
        #gas= ((group_nr == -1) | (group_nr == sg.halo_catalogue.soap_index))
        #Will also utilise group_nr bound
    )

    sg.mask_particles(mask)

    # Define a new coordinate system from the angular momentum vector
    ang_mom_vec = ang_mom_vec.squeeze()

    zhat = (ang_mom_vec / np.sqrt(np.sum(ang_mom_vec**2))).to_value(u.dimensionless)  # we'll align L with the z-axis

    arb = np.ones(3) / np.sqrt(3)  # Take an arbitrary vector
    xvec = arb - arb.dot(zhat) * zhat # Make it orthogonal to zhat
    xhat = xvec / np.sqrt(np.sum(xvec**2)) # normalise it

    yhat = np.cross(zhat, xhat) 

  

    rotmat = np.vstack((xhat, -zhat, yhat)).T 
    
    sg.rotate(Rotation.from_matrix(rotmat))

def sg_img(sg):

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

def rotate_and_project(sg):
    
    mask_edge_on(sg)
    return sg_img(sg)


def pixel_anisotropy(img, res=512): 
    mat = rotate(img.T, angle=45, reshape=False, order=3) # <--- this should be probably applied in the mask function
    
    mid = res // 2
    
    # NumPy slicing to sum quadrants instantly
    # [row_start:row_end, col_start:col_end]
    q0 = mat[:mid, :mid].sum()   # Top-left
    q1 = mat[mid:, :mid].sum()   # Bottom-left
    q2 = mat[:mid, mid:].sum()   # Top-right
    q3 = mat[mid:, mid:].sum()   # Bottom-right
    
    
    mass_major = q1 + q2
    mass_minor = q0 + q3
    
    return mass_minor / mass_major

def analysis_anisotropy(sg):

    mask_edge_on(sg)

    img = sg_img(sg)

    aniso_val = pixel_anisotropy(img)

    return sg.halo_catalogue.soap_index, aniso_val


### Script was written before the module folder was created, so has differences in structure an style

if __name__ == "__main__":

    # Snapshot and Catalogue File Paths
    cat_path =  #L200m6 catalogue path here, z = 0
    snap_path =  #L200m6 snapshot path here, z = 0


    #Task ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID',0))

    #Retrieve all halo indexes from properties file 
    all_halo_indices = pd.read_csv('halo_properties_L200m6.csv')['halo_index'].values

    #Assort haloes for each slurm array task
    
    chunks = np.array_split(all_halo_indices, 128)

    candidates = chunks[task_id]

    print(f'Task {task_id} processing {len(candidates)} haloes...')

    if len(candidates) > 0:

        soap_candidates = SOAP(cat_path, soap_index=candidates,extra_mask=None) 
        sgs = SWIFTGalaxies(snap_path, soap_candidates) 
    
        
        data = sgs.map(analysis_anisotropy) 
    
        
        df = pd.DataFrame(data, columns=['halo_index', 'mass_aniso']) 
        df.to_csv(f'results/L200m6/gas_density_anisotropies_{task_id}.csv', index=False) 