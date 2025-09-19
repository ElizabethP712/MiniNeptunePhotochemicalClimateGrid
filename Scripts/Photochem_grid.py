import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import astropy.units as u
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline

from astropy import constants
from photochem.utils import zahnle_rx_and_thermo_files
from photochem.extensions import gasgiants # Import the gasgiant extensions

import json
from astroquery.mast import Observations
from photochem.utils import stars

import star_spectrum
import pickle
import requests

from mpi4py import MPI

#from gridutils import make_grid
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import tarfile
import gridutils
import star_spectrum
import h5py
import PICASO_Climate_grid

# Finds the associated PT profile and calculates Photochemical Composition of a Planet

def find_PT_grid(filename='results/PICASO_climate_fv.h5', total_flux=None, log10_planet_metallicity=None, tint=None, gridvals=PICASO_Climate_grid.get_gridvals_PICASO_TP()):
    """
    This finds the matching PT profile in the PICASO grid to be used for Photochem grid calculation.
    
    Parameters:
    filename: string
        This is the directory path to the PICASO grid being referenced (this is the output of makegrid for climate model using PICASO)
    total_flux: float
        This is the total flux of the star on the K2-18b like planet you are searching for in units of xSolar flux in Photochem calculation.
    log10_planet_metallicity: string (but still a number)
        This is the planet metallicity of K2-18b like planet in units of xSolar metallicity in Photochem calculation.
    tint: float
        This is the planet interior temperature in units of Kelvin in Photochem calculation.
    gridvals: function
        This calls the parameter space used to create the grid whose path was given in filename (i.e. PICASO by default).
        
    Results:
    PT_list: 2D array
        This provides the matching pressures in a list (small to large, bars), then temperatures (small to large, Kelvin) in a list of the matching input values from the PICASO grid.
    convergence_values: 1D array
        This provides whether or not the PICASO model converged (0 = False, 1 = True).
    
    """
    gridvals_metal = [float(s) for s in gridvals[1]]
    gridvals_dict = {'total_flux':gridvals[0], 'planet_metallicity':np.array(gridvals_metal), 'tint':gridvals[2]}

    with h5py.File(filename, 'r') as f:
        input_list = np.array([total_flux, log10_planet_metallicity, tint])
        matches = (list(f['inputs'] == input_list))
        row_matches = np.all(matches, axis=1)
        matching_indicies = np.where(row_matches)

        matching_indicies_flux = np.where(list(gridvals_dict['total_flux'] == input_list[0]))
        matching_indicies_metal = np.where(list(gridvals_dict['planet_metallicity'] == input_list[1]))
        matching_indicies_tint = np.where(list(gridvals_dict['tint'] == input_list[2]))

        flux_index, metal_index, tint_index = matching_indicies_flux[0], matching_indicies_metal[0], matching_indicies_tint[0]

        if matching_indicies[0].size == 0:
            print(f'A match given total flux, planet metallicity, and tint does not exist')
            PT_list = None
            convergence_values = None
            return PT_list, convergence_values
            
        else:
            pressure_values = np.array(f['results']['pressure'][flux_index[0]][metal_index[0]][tint_index[0]])
            temperature_values = np.array(f['results']['temperature'][flux_index[0]][metal_index[0]][tint_index[0]])
            convergence_values = np.array(f['results']['converged'][flux_index[0]][metal_index[0]][tint_index[0]])
            PT_list = pressure_values, temperature_values
            print(f'Was able to successfully find your input parameters in the PICASO TP profile grid!')
            return PT_list, convergence_values

def linear_extrapolate_TP(P, T):
    
    """
    This extends the pressure and temperature profile from PT deeper into the atmosphere.
    
    Parameters:
    P: 1D np.array
        This is the pressure (currently set to be in PICASO default order & units (ascending, bars).
        
    T: 1D np.array
        This is the temperature (currently set to be in PICASO default order & units (ascending, K)
        
    Results:
    P: 1D np.array
        This returns a pressure array that has been extended based on a linear model mapped to a logarithmic pressure and linear Temperature of the last 5 points that adds 13 additional points up to a pressure of 10^6 bars. Units are still ascending bars.
    T: 1D np.array 
        This returns a temperature array associated with the additional pressure points added. Units are still ascending Kelvin.
    
    """
    
    new_temperature = T[-5:]
    new_pressure = np.log10(P[-5:])
    m, b = np.polyfit(new_temperature, new_pressure, 1)
    x_final = new_pressure[-5:][-1]
    
    add_pres_val = np.linspace(x_final, 6, 13)
    add_pres_val_rm_final = add_pres_val[1:]
    add_temp_val = (add_pres_val_rm_final - b)/m
    
    P = np.concatenate((P, np.array(10**add_pres_val_rm_final)))
    T = np.concatenate((T, np.array(add_temp_val)))

    return P, T

# Make it so the sol and soleq are the same length (needed for saving to h5)
def interpolate_photochem_result_to_nlayers(out, nlayers):

    """
    This makes sure the output arrays are the same length in resolution by interpolating results specific to pressure, temperature, Kzz, and mixing ratios.
    
    Parameters:
    out: dictionary
        This is the output you get from applying Photochem_Gas_Giant, or something similar with keys of np.arrays.
    nlayers: float
        This is how many values you wish to maintain in your grid.
      
    Results:
    sol: dictionary
        Each key's valued array is now the length of nlayers.
    
    """
    
    sol = {}

    # Make a new array of pressures
    sol['pressure'] = np.logspace(np.log10(np.max(out['pressure'])),np.log10(np.min(out['pressure'])),nlayers)
    log10P_new = np.log10(sol['pressure'][::-1]).copy() # log space and flipped of new pressures
    log10P = np.log10(out['pressure'][::-1]).copy() # log space and flipped old pressures

    # Do a log-linear interpolation of temperature
    T = np.interp(log10P_new, log10P, out['temperature'][::-1].copy())
    sol['temperature'] = T[::-1].copy()

    # Do a log-log interpolation of Kzz
    Kzz = np.interp(log10P_new, log10P, np.log10(out['Kzz'][::-1].copy()))
    sol['Kzz'] = 10.0**Kzz[::-1].copy()

    # Do a log-log interpolation of mixing ratios
    for key in out:
        if key not in ['pressure','temperature','Kzz']:
            tmp = np.log10(np.clip(out[key][::-1].copy(),a_min=1e-100,a_max=np.inf))
            mix = np.interp(log10P_new, log10P, tmp)
            sol[key] = 10.0**mix[::-1].copy()

    return sol

# Calculates the Chemical Composition of the Planet using Photochem
def Photochem_Gas_Giant(total_flux=None, log10_planet_metallicity=None, tint=None, log_Kzz=None, PT_filename=None):

    """
    This calculates the 1D photochemical composition of a K218b-like planet around a Sun-like star.
    
    Parameters:
    total_flux = float
        This is the total flux of radiation from the star directed onto the planet in units of x Solar. Should be the same value as those used in your PICASO grid.
    mh = string
        This is the metallicity of the planet in units of log10x Solar (so 0.5 = 10^0.5 ~ 3x Solar Metallicity). Should be the same value as those used in your PICASO grid.
    tint = float
        This is the internal temperature of the planet in units of Kelvin. Should be the same value as those used in your PICASO grid.
    log_Kzz: float
        This is the exponent of the eddy diffusion coefficient you want to use in units of cm^2/s.
    PT_filename: string
        This is the path to the PICASO PT grid you would like to use.
      
    Results:
    sol: dictionary of same length 1D arrays
        These are the molecular abundances %/total # of elements (all would be a value of 1), the pressures with each layer (this time in descending order in dynes/cm^2), the temperatures with each layer (this time in descending order in Kelvin) when a steady state was reached.
    soleq: dictionary of same length 1D arrays
        Same as sol, but when the molecules were in chemical equilibrium.
    pc: IDK
        This kept track of the inputs into the .EvoAtmosphereGasGiants function from Photochem.
    convergence_values: 1D array
        These were the values where PICASO either converged (i.e. 1 = True) or did not (i.e. 0 = False). 
    converged: 1D array
        These were the values where Photochem either converged (i.e. 1 = True) or did not (i.e. 0 = False). 
    
    """

    # Constant variables mimicing K2-18b:

    # For K2-18b (Planet Parameters)
    atoms_names = ['H', 'He', 'N', 'O', 'C'] # We select a subset of the atoms in zahnle_earth.yaml (leave out Cl), remove Sulpher for faster convergence
    mass_planet = 8.63 * (5.972e+24) * 1e3 # of K2-18b, but in grams
    radius_planet = 2.61 * (6.371e+6) * 1e2 # of K2-18b but in cm
    solar_zenith_angle = 60 # Used in Tsai et. al. (2023), in degrees
    CtoO = 1.0 # 1x solar C/O ratio
    planet_Teq = stars.equilibrium_temperature(total_flux*1361, 0)

    # Dependent constant variables
    if os.path.exists(f'sun_flux_file_{planet_Teq}'):
        stellar_flux_file = f'sun_flux_file_{planet_Teq}'
        print(f"The stellar flux file already exists")
    else:
        wv, F = star_spectrum.solar_spectrum(Teq=planet_Teq, outputfile= f'sun_flux_file_{planet_Teq}')
        stellar_flux_file = f'sun_flux_file_{planet_Teq}'

    PT_list, convergence_values = find_PT_grid(filename=PT_filename, total_flux=total_flux, log10_planet_metallicity=log10_planet_metallicity, tint=tint) # This is for when I have the grid, but for now I am testing the rest out to make sure it works.

    # Test Data - This works fine.
    #with open('out_Sun_5778_initp3bar.pkl', 'rb') as file:
    #    out_reopened = pickle.load(file)
    #    pressure = out_reopened['pressure']
    #    temperature = out_reopened['temperature']
    #PT_list = np.array(pressure), np.array(temperature)
    #convergence_values = np.array([1])

    # Define P-T Profile (convert from PICASO to Photochem)
    P_extended, T_extended = linear_extrapolate_TP(PT_list[0], PT_list[1]) # Extend the end to bypass BOA Error of mismatching boundary conditions.
    #P = np.flip(np.array(PT_list[0]) * (10**6)).copy()
    #T = np.flip(np.array(PT_list[1])).copy()
    P = np.flip(np.array(P_extended) * (10**6)).copy() # Convert from bars to dynes/cm^2
    T = np.flip(np.array(T_extended)).copy()
    
    # Check if numpy array is sorted (investigating error)
    sorted_P = np.flip(np.sort(P)).copy()
    unsorted_indices = np.where(P != sorted_P)[0]
    
    # Generate reaction & thermodynamic files for gas giants
    zahnle_rx_and_thermo_files(
    atoms_names=atoms_names,
    rxns_filename='photochem_rxns.yaml',
    thermo_filename='photochem_thermo.yaml',
    remove_reaction_particles=True # For gas giants, we should always leave out reaction particles.
    )

    # Initialize ExoAtmosphereGasGiant
    # Assigns 
    pc = gasgiants.EvoAtmosphereGasGiant(
        mechanism_file='photochem_rxns.yaml',
        stellar_flux_file=stellar_flux_file,
        planet_mass=mass_planet,
        planet_radius=radius_planet,
        solar_zenith_angle=solar_zenith_angle,
        thermo_file='photochem_thermo.yaml'
    )
    # Adjust convergence parameters:
    pc.var.conv_longdy = 0.03 # converges at 3% (change of mixing ratios over long time)
    pc.gdat.max_total_step = 10000 # assumes convergence after 10,000 steps
    
    pc.gdat.verbose = True # printing
    
    # Define the host star composition
    molfracs_atoms_sun = np.ones(len(pc.gdat.gas.atoms_names))*1e-10 # This is for the Sun
    comp = {
        'H' : 9.21e-01,
        'N' : 6.23e-05,
        'O' : 4.51e-04,
        'C' : 2.48e-04,
        'S' : 1.21e-05,
        'He' : 7.84e-02
    }

    tot = sum(comp.values())
    for key in comp:
        comp[key] /= tot
    for i,atom in enumerate(pc.gdat.gas.atoms_names):
        molfracs_atoms_sun[i] = comp[atom]
    
    pc.gdat.gas.molfracs_atoms_sun = molfracs_atoms_sun

    # Assume a default radius for particles 1e-5cm was default, so we increased the size but think of these in microns
    particle_radius = pc.var.particle_radius
    particle_radius[:,:] = 1e-3 #cm or 10 microns
    pc.var.particle_radius = particle_radius

    # Assumed Kzz (cm^2/s) in Tsai et al. (2023)
    Kzz_zero_grid = np.ones(P.shape[0])
    Kzz = Kzz_zero_grid*(10**log_Kzz) #Note Kzz_fac was meant to be the power of 10 since we are in log10 space

    # Initialize the PT based on chemical equilibrium 
    pc.gdat.BOA_pressure_factor = 3
    pc.initialize_to_climate_equilibrium_PT(P, T, Kzz, float(log10_planet_metallicity), CtoO)
    
    # Integrate to steady state
    converged = pc.find_steady_state()

    # Check if the model converged after 10,000 steps
    if not converged:
        assert pc.gdat.total_step_counter > pc.gdat.max_total_step - 10
        
    sol_raw = pc.return_atmosphere()
    soleq_raw = pc.return_atmosphere(equilibrium=True)

    # Call the interpolation of the grid 
    sol = interpolate_photochem_result_to_nlayers(out=sol_raw, nlayers=100)
    soleq = interpolate_photochem_result_to_nlayers(out=soleq_raw, nlayers=100)
    convergence_values = np.array([convergence_values[0] for _ in range(len(sol['pressure']))])
    converged = np.array([converged for _ in range(len(sol['pressure']))])

    # Print out the lengths of arrays: Save the size of the grid for future reference.
    
    print(f"This is for the input value of flux:{total_flux}, metal:{float(log10_planet_metallicity)}, tint:{tint}, log_Kzz: {log_Kzz}")
    for key, value in sol.items():
        if isinstance(value, np.ndarray):  # Check if the value is a list (or array)
            print(f"The array for sol's '{key}' has a length of: {len(value)}")
        else:
            print(f"The value for sol's '{key}' is not an array.")

    for key, value in soleq.items():
        if isinstance(value, np.ndarray):  # Check if the value is a list (or array)
            print(f"The array for soleq's '{key}' has a length of: {len(value)}, Length of pressure: {len(P)}")
        else:
            print(f"The value for soleq's '{key}' is not an array.")

    # Add nan's to fit the grid if underestimated, and make sure list goes from largest to smallest.
    

    return sol, soleq, pc, convergence_values, converged

def get_gridvals_Photochem():

    """
    This provides the input parameters to run the photochemical model over multiple computers (i.e. paralell computing).

    Parameter(s):
    log10_totalflux = np.array of floats
        This is the total flux of the starlight on the planet in units of x Solar
    log10_planet_metallicity = np.array of strings
        This is the planet metallicity in units of log10 x Solar (i.e. 0.5 inputed means 10^0.5 ~ 3x Solar Metallicity)
    tint = np.array of floats
        This is the internal temperature of the planet in units of Kelvin
    log_kzz = np.array of floats
        This is the eddy diffusion coefficient (the power of 10) in cm^2/s
    
    Returns:
        A tuple array of each array of input parameters to run via parallelization and return a 1D photochemical model.
    
    """

    # True Values to replace after test case:
    log10_totalflux = np.array([0.1, 0.5, 1, 1.5, 2])
    log10_planet_metallicity = np.array(['0.5', '1.0', '1.5', '2.0']) # in solar, the opacity files are min 0 and max 2, so I cannot do 2.5 and 3.0!
    tint = np.array([20, 40, 60, 100, 120, 140, 160, 200]) # in Kelvin
    log_Kzz = np.array([5, 7, 9]) # in cm^2/s 
    
    """
    # Test Case
    log10_totalflux = np.array([2, 1])
    log10_planet_metallicity = (['2', '1']) # in solar
    tint = np.array([40, 60]) # in Kelvin
    log_Kzz = np.array([9, 7]) # in cm^2/s (I think) 
    
    """
    gridvals = (log10_totalflux, log10_planet_metallicity, tint, log_Kzz)

    return gridvals
    
def Photochem_1D_model(x):

    """
    This runs Photochem_Gas_Giant on Tijuca for parallel computing.

    Parameters:
        x needs to be in the order of total flux, planet metallicity, tint, and kzz!
        total flux = units of solar (float)
        planet metallicity = units of solar but needs to be a float/integer NOT STRING
        tint = units of Kelvin (float)
        kzz = units of cm^2/s (float)

    Results:
    combined_dict: dictionary
        This gives you all the results of Photochem_Gas_Giant, except matches the length of convergence arrays with molecular abundances at steady state and renames where or not PICASO converged as "converged_TP" and whether or not Photochem converged as "converged_PC". Both are in the binary equivalent of the boolean True/False (i.e. 1/0). 
        
    """

    # For Tijuca
    log10_totalflux, log10_planet_metallicity, tint, log_Kzz = x
    sol, soleq, pc, convergence_values, converged = Photochem_Gas_Giant(total_flux=log10_totalflux, log10_planet_metallicity=log10_planet_metallicity, tint=tint, log_Kzz=log_Kzz, PT_filename='results/PICASO_climate_fv.h5')

    # Merge the sol & soleq & convergence arrays into a single dictionary
    modified_sol_dict = {key + "_sol": value for key, value in sol.items()}
    modified_soleq_dict = {key + "_soleq": value for key, value in soleq.items()}
    combined_dict = {**modified_sol_dict, **modified_soleq_dict}
    combined_dict['converged_TP'] = convergence_values
    combined_dict['converged_PC'] = converged

    return combined_dict 

if __name__ == "__main__":
    """
    To execute running 1D Photochemical model for the range of values in get_gridvals_Photochem, type the folling command into your terminal:
   
    # mpiexec -n X python Photochem_grid.py
    
    """
    gridutils.make_grid(
        model_func=Photochem_1D_model,
        gridvals=get_gridvals_Photochem(),
        filename='results/Photochem_1D_test2.h5',
        progress_filename='results/Photochem_1D_test2.log'
    )
