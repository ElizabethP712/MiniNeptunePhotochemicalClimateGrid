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

import h5py

# Finds the corresponding k-coefficient opacity files from Roxana et. al, 2021 on zenodo

def download_opacity(mh='1.0', ctoO='1.0', save_directory="opacities"):
    """
    Downloads a file from a given URL and saves it to a specified directory.
    
    Parameters:
    
    mh: string
        This is the metallicity of the planet in units of xSolar
    ctoO: string
        This is the c/o ratio of the planet in units of xSolar
    save_directory: string
        This is the directory all opacity folders downloaded will be saved in

    Results:
    Returns the path to the saved folder w/ opacities
    
    """

    # Sort through possible downloads
    
    record_url = "https://zenodo.org/records/7542068" # Replace with the actual record URL
    response = requests.get(record_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find download links (this might require inspecting the HTML structure of Zenodo pages)
    # Example: looking for <a> tags with specific classes or attributes
    
    for link in soup.find_all('a', href=True):
        mh_converted = f"{int(np.float64(mh)*100):03d}"
        co_converted = f"{int(np.float64(ctoO)*100):03d}"
        if "download=1" in link['href'] and f"feh+{mh_converted}_co_{co_converted}" in link.text:
            download_url = link['href']
            print(f"Found download URL: {download_url}")
            break
    else:
        print("File not found.")
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    if os.path.exists(os.path.join(save_directory, f"sonora_2020_feh+{mh_converted}_co_{co_converted}.data.196")):
        print(f'File already downloaded')
        return os.path.join(save_directory, f"sonora_2020_feh+{mh_converted}_co_{co_converted}.data.196")

    else:

        try:
            # Ensure the URL is absolute if it's relative
            if not download_url.startswith(('http://', 'https://')):
                webpage_url = "https://zenodo.org/"
                download_url = requests.compat.urljoin(webpage_url, download_url)
                print(download_url)
            
            # 4. Download the file
            file_response = requests.get(download_url, stream=True) # Use stream=True for large files
            file_response.raise_for_status() # Raise an exception for bad status codes
        
            # Extract filename from URL
            filename = os.path.basename(urlparse(download_url).path)
            if not filename:  # Handle cases where URL might not have a clear filename
                filename = f"sonora_2020_feh+{mh_converted}_co_{co_converted}.data.196.tar.gz"
                
            file_path = os.path.join(save_directory, filename)
        
            with open(file_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Successfully downloaded '{filename}' to '{save_directory}'")
    
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file from {url}: {e}")
    
        # Assuming 'downloaded_file.gz' is the path to your downloaded .gz file
    
        try:
            # Open the .tar.gz file in read mode ('r:gz' for gzip compressed tar files)
            with tarfile.open(file_path, "r:gz") as tar:
                # Extract all contents to the specified directory
                tar.extractall(save_directory)
            print(f"Successfully extracted '{file_path}' to '{save_directory}'")
            return os.path.join(save_directory, filename[:-7])
        except tarfile.ReadError as e:
            print(f"Error reading tar file: {e}")
            return None
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None


# Calculates the PT Profile Using PICASO; w/ K2-18b & G-star Assumptions for non-changing parameters; change mh, tint, and total_flux.

def calc_semi_major_SUN(Teq):
    """
    Calculates the semi-major distance from the Sun of a planet whose equilibrium temperature can vary.
    
    Parameters:
    
    Teq: float
        This is the equilibrium temperature (in Kelvin) calculated based on total flux (or otherwise) of the planet.

    Results:
    
    distance_AU: float
        Returns the distance from the planet to the Sun to maintain equilibrium temperature in AU.
    
    """
    luminosity_star = 3.846*(10**26) # in Watts for the Sun
    boltzmann_const = 5.670374419*(10**-8) # in W/m^2 * K^4
    distance_m = np.sqrt(luminosity_star / (16 * np.pi * boltzmann_const * (Teq**4)))
    distance_AU = distance_m / 1.496e+11
    return distance_AU

def PICASO_PT_Planet(mh='2.0', tint=60, total_flux=1, nlevel=91, nofczns=1, nstr_upper=85, rfacv=0.5, outputfile=None, pt_guillot=True, prior_out=None):

    """
    Calculates the semi-major distance from the Sun of a planet whose equilibrium temperature can vary.
    
    Parameters:
    
    mh = string
        This is the metallicity of the planet in units of log10 x Solar
    tint = float
        This is the internal temperature of the planet in units of Kelvin
    total_flux = float
        This is the total flux of radiation from the star directed onto the planet in units of x Solar
    nlevel = float
        Number of plane-parallel levels in your code
    nofczns = float
        Number of convective zones
    nstr_upper = float
        Top most level of guessed convective zone
    rfacv = float
        Based on Mukherjee et al. Eqn. 20, this tells you how much of the hemisphere(s) is being irradiated; if stellar irradiation is 50% (one hemisphere), rfacv is 0.5 and if just night side then rfacv is 0. If tidally locked planet, rfacv is 1.
        
    Results: CHECK THIS WHEN RUNNING CASES THAT DIDN'T CONVERGE
    
    out: dictionary
        Creates an output file that contains pressure (bars), temperature (Kelvin), and whether the model converged or not (0 = False, 1 = True), along with all input data.
    basecase: dictionary
        Creates an output file that contains the original guesses for pressure and temperature.
    
    """

    # Values of K2-18b
    ctoO='1.0'
    mass_planet = 8.63*5.972e+24*u.kg # of K2-18b
    radius_planet = 2.61*6.371e+6*u.m # of K2-18b
    grav = (const.G * (mass_planet)) / ((radius_planet)**2) # of K2-18b
    
    # Depends on mh and ctoO
    ck_db = download_opacity(mh=mh, ctoO=ctoO, save_directory="opacities")
    opacity_ck = jdi.opannection(ck_db=ck_db, method='preweighted') # grab your opacities

    # Values of the Host Star (assuming G-Star)
    T_star = 5778 # K, star effective temperature, the min value is 3500K 
    logg = 4.4 #logg , cgs
    metal = 0.0 # metallicity of star
    r_star = 1 # solar radius

    # Calculate Teq & Semi-Major Axis
    # What is the semi-major axis that is self-consistent?
    Teq = stars.equilibrium_temperature(total_flux*1361, 0) # Note converts total_flux in Earth units to Watts/m^2
    semi_major = calc_semi_major_SUN(Teq=Teq) # in AU
        
    # Starting Up the Run
    cl_run = jdi.inputs(calculation="planet", climate = True) # start a calculation 
    cl_run.gravity(gravity=grav.value, gravity_unit=u.Unit('m/(s**2)')) # input gravity
    cl_run.effective_temp(tint) # input effective temperature
    cl_run.star(opacity_ck, temp =T_star,metal =metal, logg =logg, radius = r_star, 
            radius_unit=u.R_sun,semi_major= semi_major , semi_major_unit = u.AU)#opacity db, pysynphot database, temp, metallicity, logg

    # Initial T(P) Guess
    nstr_deep = nlevel -2
    nstr = np.array([0,nstr_upper,nstr_deep,0,0,0]) # initial guess of convective zones

    # Try to fix the convergence issue by using other results as best guesses
    #with h5py.File('results/PICASO_climate_fv.h5', 'r') as f:
    #    pressure = np.array(list(f['results']['pressure'][1][0][0]))
    #    temp_guess = np.array(list(f['results']['temperature'][1][0][0]))

    if pt_guillot == True:
        pt = cl_run.guillot_pt(Teq, nlevel=nlevel, T_int = tint, p_bottom=3, p_top=-6)
        temp_guess = pt['temperature'].values
        pressure = pt['pressure'].values
    elif pt_guillot == False:
        temp_guess = prior_out['temperature']
        pressure = prior_out['pressure']
    
    # Try using the T(P) profile from the test case instead of Guillot et al 2010.
    # with open('out_Sun_5778_initP3bar.pkl', 'rb') as file:
    #     out_Gstar = pickle.load(file)
    
    #temp_guess = pt['temperature'].values
    #pressure = pt['pressure'].values

    # Initial Convective Zone Guess
    cl_run.inputs_climate(temp_guess= temp_guess, pressure= pressure, 
                      nstr = nstr, nofczns = nofczns , rfacv = rfacv)
    
    out = cl_run.climate(opacity_ck, save_all_profiles=True,with_spec=True)
    # print(f'This is the type of output I get with the keys: {out.keys()}, for the input {[mh, tint, total_flux]}')
    base_case = jdi.pd.read_csv(jdi.HJ_pt(), delim_whitespace=True)

    # Saves out and base_case to python file to be re-loaded.
    if outputfile == None:
        return out, base_case
        
    else:
        with open(f'out_{outputfile}.pkl', 'wb') as f:
            pickle.dump(out, f)
        with open(f'basecase_{outputfile}.pkl', 'wb') as f:
            pickle.dump(base_case, f)
        return out, base_case


def PICASO_fake_climate_model_testing_errors(mh, tint, total_flux, outputfile=None):
    
    fake_dictionary = {'mh': np.full(10, mh) , "tint": np.full(10, tint), 'total_flux': np.full(10, total_flux)}
    return fake_dictionary

def PICASO_climate_model(x):
    
    """
    This takes the values from get_gridvals_PICASO_TP and plugs them into PICASO_PT_Planet for parallel computing,
    then saves the results to new, simplified dictionary.

    Parameter(s):
    x: 1D array of input parameters in the order of total_flux, mh, then tint.
        mh = string like '0.0' in terms of solar metalicity
        tint = float like 70 in terms of Kelvin
        total_flux = float in terms of solar flux

    Results:
    new_out: dictionary
        This simplifies the output of PICASO into a dictionary with three keys,
        pressure at each iterated point in the profile in units of bars,
        temperature at each iterated point in the profile in units of Kelvin,
        Noting that both go from smaller value to larger value,
        and converged representing whether or not results converged (0 = False, 1 = True)
        
    
    """
    # For Tijuca
    log10_totalflux, log10_planet_metallicity, tint = x
    # print(f'This is the value of {x} used in the climate model')
    out, base_case = PICASO_PT_Planet(mh=log10_planet_metallicity, tint=tint, total_flux=log10_totalflux, outputfile=None)

    count = 0
    while out['converged'] == 0:  # An infinite loop that will be broken out of explicitly
        count += 1
        
        print(f"Loop iteration, Recalculating PT Profile: {count}")
        
        out, base_case = PICASO_PT_Planet(mh=log10_planet_metallicity, tint=tint, total_flux=log10_totalflux, outputfile=None, pt_guillot=False, prior_out = out)

        if count == 3:
            print(f"Hit the maximum amount of loops without converging.")
            break  # Exit the loop when count reaches 3

    desired_keys = ['pressure', 'temperature', 'converged']
    new_out = {key: out[key] for key in desired_keys if key in out} # Only picks out some array results from Photochem b/c not all were arrays
    new_out['converged'] = np.array([new_out['converged']])
    # Try specifying the dictionary w/ inputs and outputs

    # Testing (with a simple dictionary, the code works)
    # out = PICASO_fake_climate_model_testing_errors(mh=log10_planet_metallicity, tint=tint, total_flux=log10_totalflux, outputfile=None)

    return new_out

def get_gridvals_PICASO_TP():

    
    """
    This provides the input parameters to run the climate model over multiple computers (i.e. paralell computing).

    Parameter(s):
    log10_totalflux = np.array of floats
        This is the total flux of the starlight on the planet in units of x Solar
    log10_planet_metallicity = np.array of strings
        This is the planet metallicity in units of log10 x Solar
    tint = np.array of floats
        This is the internal temperature of the planet in units of Kelvin
    
    Returns:
        A tuple array of each array of input parameters to run via parallelization and return a 1D climate PT profile.
    
    """
    
    # True Values to replace after test case:
    log10_totalflux = np.array([0.1, 0.5, 1, 1.5, 2])
    log10_planet_metallicity = np.array(['0.5', '1.0', '1.5', '2.0']) # in solar, the opacity files are min 0 and max 2, so I cannot do 2.5 and 3.0!
    tint = np.array([20, 40, 60, 100, 120, 140, 160, 200]) # in Kelvin

    """
    # Test Case:
    log10_totalflux = np.array([0.5])
    log10_planet_metallicity = np.array(['1.0'])
    tint = np.array([20]) # in Kelvin
    """

    gridvals = (log10_totalflux, log10_planet_metallicity, tint)
    
    return gridvals

if __name__ == "__main__":

    """
    To execute running 1D PICASO climate model for the range of values in get_gridvals_PICASO_TP, type the folling command into your terminal:
    # mpiexec -n X python PICASO_Climate_grid.py

    """
    
    gridutils.make_grid(
        model_func=PICASO_climate_model, 
        gridvals=get_gridvals_PICASO_TP(), 
        filename='results/PICASO_climate_fv_no_fix_converg_take2.h5', 
        progress_filename='results/PICASO_climate_fv_no_fix_converg_take2.log'
    )



