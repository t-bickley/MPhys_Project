from swiftgalaxy import SWIFTGalaxy, SOAP, MaskCollection, SWIFTGalaxies
from scipy.spatial.transform import Rotation
from scipy.ndimage import rotate 
from swiftsimio.visualisation import project_gas, project_pixel_grid
import matplotlib.patches as patches
import pandas as pd

import numpy as np, scipy, matplotlib.pyplot as plt, unyt as u, swiftsimio as sw

def mask_edge_on(sg): 
    '''
    Masks the gas particles of a SWIFTGalaxy object to create an edge-on view.
    
    sg: SWIFTGalaxy object
    '''
   
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

    yhat = np.cross(zhat, xhat)  # Get the unit vector orthogonal to both xhat and zhat

   # rot_angle = np.pi/4
   # x = xhat * np.cos(rot_angle) - zhat * np.sin(rot_angle)
   # y = xhat * np.sin(rot_angle) + zhat * np.cos(rot_angle)
    #rotmat = np.vstack((x, y, yhat)).T
    #rotmat = np.vstack((-zhat, yhat, xhat)).T ##Edge-on upwards

    rotmat = np.vstack((xhat, -zhat, yhat)).T ##Edge-on sidewards
    
    sg.rotate(Rotation.from_matrix(rotmat))


def sg_img(sg):
    '''
    Projects the gas distribution of given galaxy

    sg: SWIFTGalaxy object

    '''

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
    

def analysis_anisotropy(sg):

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
    mask_edge_on(sg)

    #Number of gas particles in CGM (after masking)
    cgm_count = len(sg.gas.coordinates) 
    if cgm_count == 0:
        return None #skip, nothing to image
    

    img = sg_img(sg)

    aniso_val = pixel_anisotropy(img)

    return sg.halo_catalogue.input_halos_hbtplus.track_id.value[0], sg.halo_catalogue.soap_index, aniso_val, gas_frac, s_count, mmbh_mass, stellar_corot, t_mass, g_count, cgm_count