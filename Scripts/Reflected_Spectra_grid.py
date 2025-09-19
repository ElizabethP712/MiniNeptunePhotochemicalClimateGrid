import warnings
warnings.filterwarnings('ignore')
import picaso.justdoit as jdi
import picaso.justplotit as jpi
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline

from astropy import constants
from photochem.utils import zahnle_rx_and_thermo_files
from photochem.extensions import gasgiants # Import the gasgiant extensions

import json
from astroquery.mast import Observations
from photochem.utils import stars

import pickle
import requests

from mpi4py import MPI

from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import tarfile
import gridutils
import star_spectrum
import Photochem_grid
import h5py
import numpy as np
import pandas as pd
import copy

def download_opacity(mh='1.0', ctoO='1.0', save_directory="opacities"):
    
    """
    Downloads a file from a given URL and saves it to a specified directory. More specifically, finds the corresponding k-coefficient opacity files from Roxana et. al, 2021 on zenodo

    Parameters:
    mh: string
        This is the metallicity of the planet in units of x Solar metallicity.
    ctoO: string
        This is the c/o ratio of the planet in units of x Solar c/o ratio.
    save_directory: string
        This is the filename and path of where you want to save the downloaded opacity.

    Results:
    Will either mention the opacity file already exists and return the path, or will create one and return the path.
    
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

    
def earth_spectrum(opacity_path, df_mol_earth, phase, atmosphere_kwargs={}):

    """
    Calculates the Modern Earth Reflected Spectrum at full phase around the same star (Sun). 

    Parameters:
    opacity_path: string
        This provides the path to the opacity file you wish to use (we recommend v3 from Batalha et. al. 2025 on zenodo titled "Resampled Opacity Database for PICASO".
    atmosphere_kwargs: 'key': value
        If you wish to exclude any molecules, you can create a key titled 'exclude_mol' and add a list of molecules you do not wish to computer the reflected spectra of.
    df_mol_earth: dictionary with allowable abundances of molecules from the period of Earth you want
        
        EXAMPLE:

        df_mol_earth = {"N2": 0.79,
            "O2": 0.21,
            "O3": 7e-7,
            "H2O": 3e-3,
            "CO2": 300e-6,
            "CH4": 1.7e-6
        }

    Results:
    wno: grid of 150 values
        This is something, idk.
    fpfs: grid of 150 values
        This is the relative flux of the planet and star (fp/fs). 
    albedo: grid of 150 values
        This is something, idk.
    
    """

    earth = jdi.inputs()
    
    # Phase angle 
    earth.phase_angle(phase, num_tangle=8, num_gangle=8) #radians
    
    # Define planet gravity
    earth.gravity(radius=1, radius_unit=jdi.u.Unit('R_earth'),
                 mass =1, mass_unit=jdi.u.Unit('M_earth')) #any astropy units available
    earth.approx(raman="none")
    
    # Define star (same as used in K218b grid calculations)
    stellar_radius = 1 # Solar radii
    stellar_Teff = 5778 # K
    stellar_metal = 0.0 # log10(metallicity)
    stellar_logg = 4.4 # log10(gravity), in cgs units
    opacity = jdi.opannection(filename_db=opacity_path, wave_range=[0.3,2.5])
    
    earth.star(opannection=opacity,temp=stellar_Teff,logg=stellar_logg,semi_major=1, metal=stellar_metal,
               semi_major_unit=u.Unit('au')) 

    # P-T-Composition
    nlevel = 90 
    P = np.logspace(-6, 0, nlevel)
    df_atmo = earth.TP_line_earth(P , nlevel = nlevel)
    df_pt_earth =  pd.DataFrame({
        'pressure':df_atmo['pressure'].values,
        'temperature':df_atmo['temperature'].values})

    if df_mol_earth == None:
        df_mol_earth_modern_default = pd.DataFrame({
                "N2":P*0+0.79,
                "O2":P*0+0.21,
                "O3":P*0+7e-7,
                "H2O":P*0+3e-3,
                "CO2":P*0+300e-6,
                "CH4":P*0+1.7e-6
            })
        
        df_atmo_earth = df_pt_earth.join(df_mol_earth_modern_default, how='inner')
        print(df_atmo_earth)

    else:
        df_mol_earth_grid_dict = {}
        df_mol_earth_grid = pd.DataFrame({})
        for key in df_mol_earth:
            df_mol_earth_grid_dict[key] = df_mol_earth[key] + (P*0)
            for key in df_mol_earth_grid_dict:
                df_mol_earth_grid[key] = pd.Series(df_mol_earth_grid_dict[key])

        df_atmo_earth = df_pt_earth.join(df_mol_earth_grid, how='inner')
        print(df_atmo_earth)
            
    if 'exclude_mol' in atmosphere_kwargs:
        sp = atmosphere_kwargs['exclude_mol'][0]
        if sp in df_atmo_earth:
            df_atmo_earth[sp] *= 0
            
    # earth.atmosphere(df=df_atmo_earth, **atmosphere_kwargs)
    
    earth.atmosphere(df=df_atmo_earth)
    earth.surface_reflect(0.1,opacity.wno)

    # Cloud free spectrum
    df_cldfree = earth.spectrum(opacity,calculation='reflected',full_output=True)

    # Clouds
    ptop = 0.6
    pbot = 0.7
    logdp = np.log10(pbot) - np.log10(ptop)  
    log_pbot = np.log10(pbot)
    earth.clouds(w0=[0.99], g0=[0.85], 
                 p = [log_pbot], dp = [logdp], opd=[10])

    # Cloud spectrum
    df_cld = earth.spectrum(opacity,full_output=True)

    # Average the two spectra
    wno, alb, fpfs, albedo = df_cldfree['wavenumber'],df_cldfree['albedo'],df_cldfree['fpfs_reflected'], df_cldfree['albedo']
    wno_c, alb_c, fpfs_c, albedo_c = df_cld['wavenumber'],df_cld['albedo'],df_cld['fpfs_reflected'], df_cld['albedo']
    _, albedo = jdi.mean_regrid(wno, 0.5*albedo+0.5*albedo_c,R=150)
    wno, fpfs = jdi.mean_regrid(wno, 0.5*fpfs+0.5*fpfs_c,R=150)
    

    return wno, fpfs, albedo

def make_case_earth(opacity_path=f'/Users/epawelka/Documents/NASA_Ames_ProjS25/AmesProjS25Work/picaso_v4/reference/opacities/opacities_0.3_15_R15000.db', df_mol_earth=None, phase=0, species=None):

    """
    This calculates a dictionary of wno, albedo, and fpfs results from earth_spectrum.

    Provide a list if you wish to limit the species calculated by the reflected light spectra.
    species = ['O2','H2O','CO2','O3','CH4']
    
    """
    res = {}
    res['all'] = earth_spectrum(opacity_path, df_mol_earth, phase) # in order of wno, fpfs, alb
    
    if species is not None:
        for sp in species:
            tmp = earth_spectrum(opacity_path, atmosphere_kwargs={'exclude_mol': [sp]})
            res[sp] = tmp[:2]
        return res

    else:
        return res
    
def calc_semi_major_SUN(Teq):

    """
    This calculates the semi major axis (AU) between the Sun and a planet given the planet's equilibrium temperature (K).
    """
    luminosity_star = 3.846*(10**26) # in Watts for the Sun
    boltzmann_const = 5.670374419*(10**-8) # in W/m^2 * K^4
    distance_m = np.sqrt(luminosity_star / (16 * np.pi * boltzmann_const * (Teq**4)))
    distance_AU = distance_m / 1.496e+11
    return distance_AU
    
def find_Photochem_match(filename='results/Photochem_1D_fv.h5', total_flux=None, log10_planet_metallicity=None, tint=None, Kzz=None, gridvals= Photochem_grid.get_gridvals_Photochem()):
    
    """
    This finds the Photochem match on the grid based on inputs into the Reflected Spectra grid.

    Parameters:
    filename: string
        this is the file path to the output of makegrid for Photochemical model
    total_flux: float
        This is the total flux of stellar radiation on your planet in units of x Solar flux.
    log10_planet_metallicity: float
        This is the planet's metallicity in units of log10 x Solar metallicity.
    tint: float
        This is the planet's internal temperature in Kelvin.
    Kzz: float
        This is the eddy diffusion coefficient in logspace (i.e. the power of 10) in cm/s^2.
    gridvals: tuple of 1D arrays
        Input values for total_flux, planet metallcity, tint, and kzz used to make the Photochemical grid.

    Results:
    sol_dict_new: dictionary of np.arrays
        This provides the matching solutions dictionary from photochem matching total_flux, metallicity, tint, and kzz inputs.
    soled_dict_new: dictionary of np.arrays
        This provides the matching solutions dictionary (from chemical equilibrium) matching total_flux, metallicity, tint, and kzz inputs.
    PT_list: 2D array
        This provides the matching pressure (in dynes/cm^2), temperature (Kelvin) from the Photochemical grid solution (not PICASO, since this involved some extrapolation and interpolation). 
    convergence_PC: 1D array
        This provides information on whether or not the Photochem model converged, using the binary equivalent of booleans (1=True, 0=False)
    convergence_TP: 1D array
        This provides information on whether or not the PICASO model used in Photochem was converged, using binary equivalent of booleans (1=True, 0=False)
        
    """
    gridvals_metal = [float(s) for s in gridvals[1]]
    planet_metallicity = float(log10_planet_metallicity)
    gridvals_dict = {'total_flux':gridvals[0], 'planet_metallicity':gridvals_metal, 'tint':gridvals[2], 'Kzz':gridvals[3]}

    with h5py.File(filename, 'r') as f:
        input_list = np.array([total_flux, planet_metallicity, tint, Kzz])
        matches = (list(f['inputs'] == input_list))
        row_matches = np.all(matches, axis=1)
        matching_indicies = np.where(row_matches)

        matching_indicies_flux = np.where(list(gridvals_dict['total_flux'] == input_list[0]))
        matching_indicies_metal = np.where(list(gridvals_dict['planet_metallicity'] == input_list[1]))
        matching_indicies_tint = np.where(list(gridvals_dict['tint'] == input_list[2]))
        matching_indicies_kzz = np.where(list(gridvals_dict['Kzz'] == input_list[3]))

        flux_index, metal_index, tint_index, kzz_index = matching_indicies_flux[0], matching_indicies_metal[0], matching_indicies_tint[0], matching_indicies_kzz[0]

        if matching_indicies[0].size == 0:
            print(f'A match given total flux, planet metallicity, and tint does not exist')
            sol_dict_new = None
            soleq_dict_new = None
            PT_list = None
            convergence_PC = None
            convergence_TP = None
            return sol_dict_new, soleq_dict_new, PT_list, convergence_PC, convergence_TP
            
        else:
            sol_dict = {}
            soleq_dict = {}
            for key in list(f['results']):
                if key.endswith("sol"):
                    sol_dict[key] = np.array(f['results'][key][flux_index[0]][metal_index[0]][tint_index[0]][kzz_index[0]])
                elif key.endswith("soleq"):
                    soleq_dict[key] = np.array(f['results'][key][flux_index[0]][metal_index[0]][tint_index[0]][kzz_index[0]])

            sol_dict_new = {key.removesuffix('_sol') if key.endswith('_sol') else key: value 
    for key, value in sol_dict.items()}

            soleq_dict_new = {key.removesuffix('_soleq') if key.endswith('_soleq') else key: value 
    for key, value in soleq_dict.items()}
                        
            pressure_values = np.array(f['results']['pressure_sol'][flux_index[0]][metal_index[0]][tint_index[0]][kzz_index[0]])
            temperature_values = np.array(f['results']['temperature_sol'][flux_index[0]][metal_index[0]][tint_index[0]][kzz_index[0]])
            convergence_PC = np.array(f['results']['converged_PC'][flux_index[0]][metal_index[0]][tint_index[0]][kzz_index[0]])
            convergence_TP = np.array(f['results']['converged_TP'][flux_index[0]][metal_index[0]][tint_index[0]][kzz_index[0]])
            PT_list = pressure_values, temperature_values
            print(f'Was able to successfully find your input parameters in the PICASO TP profile grid!')
            
            return sol_dict_new, soleq_dict_new, PT_list, convergence_PC, convergence_TP

def find_pbot_old(sol):

    """
    This finds the bottom of an Earth-like block of a gray cloud.

    Parameters:
    sol: dictionary
        This is the solution in a steady state produced by the Photochem grid.
    
    Results:
    H2O_cld: np.array
        This tells us what pressure values are closest to 10^-4 in magnitude (within 10^-5 to 10^2) under H2Oaer (liquid water mixing ratio).
    pbot: float
        This gives us the pressure at the bottom of the cloud (closest to 10^-4).
        
    """
    
    H2O_cld = []
    H2Oaer_mixing = list(sol['H2Oaer'])
    print(f"This is the list for H2Oaer_mixing: {H2Oaer_mixing}")
    for value in H2Oaer_mixing:
        if (value / (10**-4)) > (9e-1) and (value / (10**-4)) < (9e+2):
            H2O_cld.append(value)
            print(f'first iteration:{H2O_cld}')
    if len(H2O_cld) == 1:
        index = H2Oaer_mixing.index(H2O_cld[0])
        pbot = sol['pressure'][index]
    elif len(H2O_cld) > 1:
        H2O_cld.clear()
        for value in H2Oaer_mixing:
            if (value / (10**-4)) > (9e-1) and (value / (10**-4)) < (9e+1):
                H2O_cld.append(value)
                print(f'second iteration: {H2O_cld}')
    if len(H2O_cld) == 1:
        print(H2O_cld)
        index = H2Oaer_mixing.index(H2O_cld[0])
        pbot = sol['pressure'][index]
    elif len(H2O_cld) > 1:
        H2O_cld.clear()
        for value in H2Oaer_mixing:
            if (value / 10**-4) > 0.99999 and (value / 10**-4) < 9:
                H2O_cld.append(value)
                print(f'third iteration: {H2O_cld}')
    if len(H2O_cld) == 1:
        index = H2Oaer_mixing.index(H2O_cld[0])
        pbot = sol['pressure'][index]
    elif len(H2O_cld) > 1:
        H2O_cld = H2O_cld[0]
        index = H2Oaer_mixing.index(H2O_cld)
        pbot = sol['pressure'][index]
        print(f'Chose first element in nearly converged array')      
    else:
        H2O_cld = None
        pbot = None
        print(f'H2Oaer has no values close to 10**-4 in mixing ratio')

    return H2O_cld, pbot

def find_pbot(sol=None, solaer=None, tol=0.9):

    """
    Parameters:
    pressures: ndarray
        Pressure at each atmospheric layer in dynes/cm^2
    H2Oaer: ndarray
        Mixing ratio of H2O aerosols.
    tol: float, optional
        The threshold value for which we define the beginning of the cloud, 
        by default 1e-4. 

    Returns:
    P_bottom: float
        The cloud bottom pressure in dynes/cm^2
        
    """

    pressure = sol['pressure']
    H2Oaer = solaer['H2Oaer']

    # There is no water cloud in the model, so we return None
    # For the cloud bottom of pressure

    if np.max(H2Oaer) < 1e-20:
        return None

    # Normalize so that max value is 1
    H2Oaer_normalized = H2Oaer/np.max(H2Oaer)

    # loop from bottom to top of atmosphere, cloud bottom pressure
    # defined as the index level where the normalized cloud mixing ratio
    # exeeds tol .

    ind = None
    
    for i, val in enumerate(H2Oaer_normalized):
        if val > tol:
            ind = i
            break

    if ind is None:
        raise Exception('A problem happened when trying to find the bottom of the cloud.')

    # Bottom of the cloud
    pbot = pressure[ind]

    return pbot


# Make a Global Variable
opacity_path=f'/Users/epawelka/Documents/NASA_Ames_ProjS25/AmesProjS25Work/picaso_v4/reference/opacities/opacities_0.3_15_R15000.db'
OPACITY = jdi.opannection(filename_db=opacity_path, wave_range=[0.3,2.5])

# Flip the data between PICASO and Photochem

def make_picaso_atm(sol):
    """
    Takes in a dictionary from Photochem output, converts pressure from dynes/cm^2 to bars, and flips all data for PICASO, and gets rid of any aer molecule abundances. Returns a dictionary.
    
    """
    sol_dict_noaer = {}
    sol_dict_aer = {}
    for key in sol.keys():
        if not key.endswith('aer'):
            sol_dict_noaer[key] = sol[key]
        elif key.endswith('aer'):
            sol_dict_aer[key] = sol[key]
        else:
            continue
        
    atm = copy.deepcopy(sol_dict_noaer)
    atm['pressure'] /= 1e6 # in bars
    for key in atm:
        atm[key] = atm[key][::-1].copy()

    sol_dict_aer = copy.deepcopy(sol_dict_aer)
    for key in sol_dict_aer:
        sol_dict_aer[key] = sol_dict_aer[key][::-1].copy()
    
    return atm, sol_dict_aer

# This calculates the spectrum (for now, without clouds)
def reflected_spectrum_K218b_Sun(total_flux=None, planet_metal=None, tint=None, Kzz=None, phase_angle=None, Photochem_file='results/Photochem_1D_fv.h5', atmosphere_kwargs={}):

    """
    This finds the reflected spectra of a planet similar to K218b around a Sun.

    Parameters:
    total_flux: float
        This is the total flux of stellar radiation on your planet in units of x Solar flux.
    log10_planet_metallicity: float
        This is the planet's metallicity in units of log10 x Solar metallicity.
    tint: float
        This is the planet's internal temperature in Kelvin.
    Kzz: float
        This is the eddy diffusion coefficient in logspace (i.e. the power of 10) in cm/s^2.
    phase_angle: float
        This is the phase of orbit the planet is in relative to its star and the observer (i.e. how illuminated it is), units of radians.
    Photochem_file: string
        This is the path to the Photochem grid you would like to pull composition information from.
    atmosphere_kwargs: dict 'exclude_mol': value where value is a string
        If left empty, all molecules are included, but can limit how many molecules are calculated. 

    Results: IDK for sure though
    wno: grid of 150 points
        ???
    fpfs: grid of 150 points
        This is the relative flux of the planet and star (fp/fs). 
    alb: grid of 150 points
        ???
    np.array(clouds): grid of 150 points
        This is a grid of whether or not a cloud was used to make the reflective spectra using the binary equivalent to booleans (True=1, False=0).
        
    """

    opacity = OPACITY

    planet_metal = float(planet_metal)
    
    start_case = jdi.inputs()

    # Then calculate the composition from the TP profile
    class K218b:
        
        planet_radius = (2.61*6.371e+6*u.m) # in meters
        planet_mass = (8.63*5.972e+24*u.kg) # in kg
        planet_Teq = stars.equilibrium_temperature(total_flux*1361, 0) # Equilibrium temp (K)
        planet_grav = (const.G * (planet_mass)) / ((planet_radius)**2) # of K2-18b in m/s^2
        planet_ctoO = 1.0 # 1x solar

    class Sun:
        
        stellar_radius = 1 # Solar radii
        stellar_Teff = 5778 # K
        stellar_metal = 0.0 # log10(metallicity)
        stellar_logg = 4.4 # log10(gravity), in cgs units

    semi_major = calc_semi_major_SUN(Teq=K218b.planet_Teq) # in AU
    solar_zenith_angle = 60 # Used in Tsai et al. (2023)
        
    # Star and Planet Parameters (Stay the Same & Should Match Photochem & PICASO)
    start_case.phase_angle(phase_angle, num_tangle=8, num_gangle=8) #radians, using default here

    jupiter_mass = const.M_jup.value # in kg
    jupiter_radius = 69911e+3 # in m
    start_case.gravity(gravity=K218b.planet_grav, gravity_unit=jdi.u.Unit('m/(s**2)'), radius=(K218b.planet_radius.value)/jupiter_radius, radius_unit=jdi.u.Unit('R_jup'), mass=(K218b.planet_mass.value)/jupiter_mass, mass_unit=jdi.u.Unit('M_jup'))
    
    # star temperature, metallicity, gravity, and opacity (default opacity is opacity.db in the reference folder)
    start_case.star(opannection=opacity, temp=Sun.stellar_Teff, logg=Sun.stellar_logg, semi_major=semi_major, metal=Sun.stellar_metal, radius=Sun.stellar_radius, radius_unit=jdi.u.R_sun, semi_major_unit=jdi.u.au)

    # Match Photochemical Files
    sol_dict, soleq_dict, PT_list, convergence_PC, convergence_TP = find_Photochem_match(filename=Photochem_file, total_flux=total_flux, log10_planet_metallicity=planet_metal, tint=tint, Kzz=Kzz)

    # Determine Planet Atmosphere & Composition

    atm, sol_dict_aer = make_picaso_atm(sol_dict) # Converted Pressure of Photochem, in dynes/cm^2, back to bars and flip all arrays before placing into PICASO
    df_atmo = jdi.pd.DataFrame(atm)

    if 'exclude_mol' in atmosphere_kwargs:
        sp = atmosphere_kwargs['exclude_mol'][0]
        if sp in df_atmo:
            df_atmo[sp] *= 0
            
    start_case.atmosphere(df = df_atmo) 
    df_cldfree = start_case.spectrum(opacity, calculation='reflected', full_output=True)
    wno_cldfree, alb_cldfree, fpfs_cldfree = df_cldfree['wavenumber'], df_cldfree['albedo'], df_cldfree['fpfs_reflected']
    _, alb_cldfree_grid = jdi.mean_regrid(wno_cldfree, alb_cldfree, R=150)
    wno_cldfree_grid, fpfs_cldfree_grid = jdi.mean_regrid(wno_cldfree, fpfs_cldfree, R=150)

    print(f'This is the length of the grids created: {len(wno_cldfree_grid)}, {len(fpfs_cldfree_grid)}')

    # Determine Whether to Add Clouds or Not?

    if "H2Oaer" in sol_dict_aer:
        # What if we added Grey Earth-like Clouds?
        
        # Calculate pbot:
        pbot = find_pbot(sol = atm, solaer=sol_dict_aer)

        if pbot is not None:
            print(f'pbot was calculated, there is H2Oaer and a cloud was implemented')
            logpbot = np.log10(pbot)
        
            # Calculate logdp:
            ptop_earth = 0.6
            pbot_earth = 0.7
            logdp = np.log10(pbot_earth) - np.log10(ptop_earth)  
    
            # Default opd (optical depth), w0 (single scattering albedo), g0 (asymmetry parameter)
            start_case.clouds(w0=[0.99], g0=[0.85], 
                              p = [logpbot], dp = [logdp], opd=[10])
            # Cloud spectrum
            df_cld = start_case.spectrum(opacity,full_output=True)
            
            # Average the two spectra - This differs between Calculating Earth Reflected Spectra 
            wno_c, alb_c, fpfs_c, albedo_c = df_cld['wavenumber'],df_cld['albedo'],df_cld['fpfs_reflected'], df_cld['albedo']
            _, alb = jdi.mean_regrid(wno_cldfree, 0.5*alb_cldfree+0.5*albedo_c,R=150)
            wno, fpfs = jdi.mean_regrid(wno_cldfree, 0.5*fpfs_cldfree+0.5*fpfs_c,R=150)

            # Match the length of the clouds array with the length of wno or alb (fpfs is different length)
            clouds = [1] * len(wno)

            return wno, fpfs, alb, np.array(clouds)

        else:
            print(f'pbot is empty, so no cloud is implemented')
            wno = wno_cldfree_grid.copy()
            alb = alb_cldfree_grid.copy()
            fpfs = fpfs_cldfree_grid.copy()

            # Match the length of the clouds array with the length of wno or alb (fpfs is different length)
            clouds = [0] * len(wno)

            print(f'This is the length of the values I want to save: wno {len(wno)}, alb {len(alb)}, fpfs {len(fpfs)}, clouds {len(clouds)}')

            return wno, fpfs, alb, np.array(clouds)

    else:
        print(f'H2Oaer is not in solutions')
        wno = wno_cldfree_grid.copy()
        alb = alb_cldfree_grid.copy()
        fpfs = fpfs_cldfree_grid.copy()
        print(f'For the inputs: {total_flux}, {planet_metal}, {tint}, {Kzz}, {phase_angle}, The length should match: wno - {len(wno)}, alb - {len(alb)}, fpfs - {len(fpfs)}')
        
        # Match the length of the clouds array with the length of wno or alb (fpfs is different length)
        clouds = [0] * len(wno) # This means that there are no clouds

        return wno, fpfs, alb, np.array(clouds)

def make_case_RSM(total_flux=None, planet_metal=None, tint=None, Kzz=None, phase_angle=None, limit_sp = False, species=['O2','H2O','CO2','O3','CH4']):

    """
    This calculates a dictionary of wno, albedo, and fpfs results from reflected_spectrum_K218b_Sun. When limit_sp is True, it will exclude species O2, H2O, CO2, O3, and CH4, but by default just puts outputs into a dictionary with the keys wno, fpfs, albedo, and clouds.
    
    """
    
    res = {}
    
    wno, fpfs, albedo, clouds = reflected_spectrum_K218b_Sun(total_flux=total_flux, planet_metal=planet_metal, tint=tint, Kzz=Kzz, phase_angle=phase_angle)
    
    res['wno'] = wno
    res['fpfs'] = fpfs
    res['albedo'] = albedo
    res['clouds'] = clouds

    if limit_sp == True:
        for sp in species:
            tmp = reflected_spectrum_K218b_Sun(total_flux=total_flux, planet_metal=planet_metal, tint=tint, Kzz=Kzz, phase_angle=phase_angle, atmosphere_kwargs={'exclude_mol': [sp]})
            res[sp] = tmp[:2]
            
        return res
    else:
        return res

def get_gridvals_RSM():

    """
    This provides the input parameters to run the reflected spectra model over multiple computers (i.e. paralell computing).

    Parameter(s):
    log10_totalflux = np.array of floats
        This is the total flux of the starlight on the planet in units of x Solar
    log10_planet_metallicity = np.array of strings
        This is the planet metallicity in units of log10 x Solar
    tint = np.array of floats
        This is the internal temperature of the planet in units of Kelvin
    log_kzz = np.array of floats
        This is the eddy diffusion coefficient (the power of 10) in cm^2/s
    phase_angle = np.array of floats
        This is the phase of orbit the planet is relative to its star & the observer in radians
    
    Returns:
        A tuple array of each array of input parameters to run via parallelization and return a reflected spectra grid.

    # Test Case: This was with the old results from Tijuca
    log10_totalflux = np.array([0.1, 0.5, 1, 1.5, 2])
    log10_planet_metallicity = np.array(['0.5', '1.0', '1.5', '2.0']) # in solar, the opacity files are min 0 and max 2, so I cannot do 2.5 and 3.0!
    tint = np.array([20, 40, 60, 100]) # in Kelvin
    log_Kzz = np.array([5, 7, 9]) # in cm^2/s 
    phase_angle_list = np.linspace(0, np.pi, 19)
    phase_angle = phase_angle_list[:-1] # in radians, this goes in 10 degree intervals from 0 to 170 degrees
    
    # Test Case:
    log10_totalflux = np.array([0.5]) # in terms of x solar
    log10_planet_metallicity = np.array([2.0]) # in terms of x solar
    tint = np.array([40, 60]) # in Kelvin
    log_Kzz = np.array([7]) # the power associated with 10
    phase_angle = np.array([180*(np.pi/180)]) 
    
    """
    
    # True Values to replace after test case (what is being run on Tijuca)
    log10_totalflux = np.array([0.1, 0.5, 1, 1.5, 2])
    log10_planet_metallicity = np.array(['0.5', '1.0', '1.5', '2.0']) # in solar, the opacity files are min 0 and max 2, so I cannot do 2.5 and 3.0!
    tint = np.array([20, 40, 60, 100, 120, 140, 160, 200]) # in Kelvin
    log_Kzz = np.array([5, 7, 9]) # in cm^2/s 
    phase_angle_list = np.linspace(0, np.pi, 19)
    phase_angle = phase_angle_list[:-1] # in radians, this goes in 10 degree intervals from 0 to 170 degrees
    gridvals = (log10_totalflux, log10_planet_metallicity, tint, log_Kzz, phase_angle)
    
    return gridvals

def Reflected_Spectra_model(x):

    """
    This runs Photochem_Gas_Giant on Tijuca for parallel computing.

    Parameters:
        x needs to be in the order of total flux, planet metallicity, tint, and kzz!
        total flux = units of solar (float)
        planet metallicity = units of log10 solar but needs to be a float/integer NOT STRING
        tint = units of Kelvin (float)
        kzz = units of cm^2/s (float)
        phase_angle = units of radians (float)

    Results:
    res: dictionary
        This gives you all the results of reflected_spectrum_K218b_Sun.

    """
    # For Tijuca
    log10_totalflux, log10_planet_metallicity, tint, log_Kzz, phase_angle = x
    
    res = make_case_RSM(total_flux=log10_totalflux, planet_metal=log10_planet_metallicity, tint=tint, Kzz=log_Kzz, phase_angle=phase_angle)

    return res

if __name__ == "__main__":
    """
    To execute running Reflected Spectra model for the range of values in get_gridvals_RSM, type the folling command into your terminal:
   
    # mpiexec -n X python Reflected_Spectra_grid.py
    
    """
    gridutils.make_grid(
        model_func=Reflected_Spectra_model, 
        gridvals=get_gridvals_RSM(), 
        filename='results/ReflectedSpectra_fv.h5', 
        progress_filename='results/ReflectedSpectra_fv.log'
    )
