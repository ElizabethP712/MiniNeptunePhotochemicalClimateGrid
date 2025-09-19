import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import Photochem_grid
import Reflected_Spectra_grid as Reflected_Spectra
import PICASO_Climate_grid
import FilterGrids
import GraphsKey
from photochem.utils import stars
import pickle

from matplotlib import pyplot as plt
from itertools import cycle

from photochem.clima import rebin

def validate_inputs(flux, metal, tint, kzz, phase):

"""
This makes sure all inputs are within the range used in the default grid. Please change the ranges in the script if you have changed the ranges of inputs
for the PICASO, Photochem, and Reflected Spectra grids describing mini Neptunes.

Parameters:
    flux: np.array
        This is the total flux of the planet in units of x Solar.
    metal: np.array
        This is the metallicity of the planet in units of log10 x Solar (so 0.5 is 10^0.5 = 3x Solar)
    tint: np.array
        This is the internal temperature in Kelvin of the planet.
    kzz: np.array
        This is the eddy diffusion coeficient used in logspace and cm^2/s.
    phase: np.array
        This is how much of the face of the planet we can observe in its orbit with 0 being the brightest and increasing degrees becomming
        more dim. Units should be in radians. 
Returns:
    Boolean, if within range will return True, and if not will return False.
"""
    # Define ranges for each parameter
    ranges = {
        'flux': {'min': 0.1, 'max': 2},
        'metal': {'min': 0.5, 'max': 2},
        'tint': {'min': 20, 'max': 200},
        'kzz': {'min': 5, 'max': 9},
        'phase': {'min': 0, 'max': 2.97}
    }

    if not (ranges['flux']['min'] <= flux <= ranges['flux']['max']):
        print(f"Error: Flux ({flux}) is out of range [{ranges['flux']['min']}, {ranges['flux']['max']}]")
        return False
    if not (ranges['metal']['min'] <= metal <= ranges['metal']['max']):
        print(f"Error: Metallicity ({metal}) is out of range [{ranges['metal']['min']}, {ranges['metal']['max']}]")
        return False
    if not (ranges['tint']['min'] <= tint <= ranges['tint']['max']):
        print(f"Error: Tint ({tint}) is out of range [{ranges['tint']['min']}, {ranges['tint']['max']}]")
        return False
    if not (ranges['kzz']['min'] <= kzz <= ranges['kzz']['max']):
        print(f"Error: Tint ({kzz}) is out of range [{ranges['kzz']['min']}, {ranges['kzz']['max']}]")
        return False
    if not (ranges['phase']['min'] <= phase <= ranges['phase']['max']):
        print(f"Error: Phase ({phase}) is out of range [{ranges['phase']['min']}, {ranges['phase']['max']}]")
        return False
        
    # ... continue for all parameters

    return True

# This is the function describing g(x)
def mini_nep_model(wv, flux, metal, tint, kzz, phase):
"""
This solve for the flux of planet/star ratio (fpfs) (will interpolate if inputs are not exactly on grids), and also returns the wavelengths associated with those fpfs values.

Parameters:
    flux: np.array
        This is the total flux of the planet in units of x Solar.
    metal: np.array
        This is the metallicity of the planet in units of log10 x Solar (so 0.5 is 10^0.5 = 3x Solar)
    tint: np.array
        This is the internal temperature in Kelvin of the planet.
    kzz: np.array
        This is the eddy diffusion coeficient used in logspace and cm^2/s.
    phase: np.array
        This is how much of the face of the planet we can observe in its orbit with 0 being the brightest and increasing degrees becomming
        more dim. Units should be in radians. 
Returns:
    fpfs_interpolated: list
        These are the fpfs values interpolated
    fpfs: list
        These are the fpfs values not interpolated
    wv: list
        These are the associated wavenumbers with those fpfs values
"""

    range_check = validate_inputs(flux, metal, tint, kzz, phase)

    if range_check is False:
        fpfs_interpolated = 0
        fpfs = 0
        wno = 0
        
    else:
        test_results = FilterGrids.find_ReflectedSpectra_sol(total_flux=flux, log10_planet_metallicity=metal, tint=tint, Kzz=kzz, phase=phase)
        alb = test_results['albedo']
        wno = test_results['wno']
        fpfs= test_results['fpfs']
    
        # Need to find a match for wno to wv and interpolate results?
        fpfs_interpolated = np.interp(wv, wno, fpfs)

        print(f"Length of interpolated fpfs vs wno: {len(fpfs_interpolated)}, {len(wno)}, {len(wv)}")
    
    return fpfs_interpolated, fpfs, wv
        

def objective(z, wv, fpfs_earth):

"""
This solves for a single list of inputs, a given range of wavenumbers, and the flux planet/star ratio of the earth you'd like to optimize your mini-Neptune for
that is affiliated with those wavenumbers. 

Parameters:
    z: multidimensional array of inputs for flux, metal, tint, kzz, and phase
        These are the same inputs you'd want to plug into the mini_nep_model function.
    wv: list of floats
        These are the wavenumbers you wish to find associated vector norms of for fpfs values.
        Typically in our notebooks we used earth wv and were searching for associated mini-Neptune wavenumbers to
        Earths.
    fpfs_earth: list of floats
        These are the flux planet/star ratios associated with the wv provided, in our case of Earth or whatever
        you want to compare with your mini-Neptunes. 
Returns:
    vector norm: float or None
        If it is able to calculate fpfs_interp of your minineptune, it then calcuates how far off that is from Earths at 
        a given wavelength. If it is not able to calculate fpfs_interp, it returns None.

"""
    res_dict = {}
    flux, metal, tint, kzz, phase = z
    fpfs_interp, fpfs_before, wno_before = mini_nep_model(wv, flux, metal, tint, kzz, phase)
    
    if fpfs_interp is not None:
        return np.sqrt(np.sum((fpfs_interp - fpfs_earth)**2))
    else:
        return None

def calc_objective_dict(wv_earth, fpfs_earth, resolution=5, total_flux_list=None, planet_metal_list=None, tint_list=None, kzz_list=None, phase_list=None):

"""
This solves for a range of inputs, a range of wavenumbers, and the flux planet/star ratio of the earth you'd like to optimize your mini-Neptune for affiliated
with those wavenumbers. 

Parameters:
    wv_earth: list of floats
        These are the wavenumbers you wish to find associated vector norms of for fpfs values.
        Typically in our notebooks we used earth wv and were searching for associated mini-Neptune wavenumbers to
        Earths.
    fpfs_earth: list of floats
        These are the flux planet/star ratios associated with the wv provided, in our case of Earth or whatever
        you want to compare with your mini-Neptunes. 
    resolution: integer
        This tells you how many points you want to range over for each input in the script;
        Default values are associated with the default grid provided in the github repo. 
    total_flux_list: list of floats
        Total fluxes you wish to range over. Should be in units of x Solar total flux.
    planet_metal_list: list of floats or np.linspace
        Planet metallicities you wish to range over. Should be in units of log10 x Solar Metallicity.
    tint_list: list of floats or np.linspace
        Internal temperature of the planet you wish to range over. Should be in units of Kelvin.
    kzz_list: list of floats or np.linspace
        Eddy diffusion coefficient in units of cm^2/s, also in log10 space. 
    phase_list: list of floats or np.linspace
        Phase of light visible from planet in orbit, in units of radians. 
Returns:
    fpfs_dict: dictionary
        This saves all the fpfs differences between Earth and your mini-Neptune or planet from grids with their associated inputs as the keys.

"""

    if total_flux_list is not None:
        total_flux = total_flux_list
    else:
        total_flux = np.linspace(0.1, 2, resolution)
    
    if planet_metal_list is not None:
        planet_metal = planet_metal_list
    else:
        planet_metal = np.linspace(0.5, 2, resolution)
        
    if tint_list is not None:
        tint = tint_list
    else:
        tint = np.linspace(20, 200, resolution)
        
    if kzz_list is not None:
        kzz = kzz_list
    else:
        kzz = np.linspace(5, 9, resolution)
        
    if phase_list is not None:
        phase = phase_list
    else:
        phase = np.linspace(0, 2.9, resolution)
    
    fpfs_dict = {}
    
    for flux in total_flux:
        for metal in planet_metal:
            for tint_val in tint:
                for kzz_val in kzz:
                    for phase_val in phase:
                        z = flux, metal, tint_val, kzz_val, phase_val
                        fpfs_difference_grid = objective(z=z, wv=wv_earth, fpfs_earth=fpfs_earth)
                        fpfs_dict[f'{z}'] = fpfs_difference_grid

    return fpfs_dict

def graph_vec_norm_fpfs(fpfs_dict=None, tol=None, full_plot=True):

"""
This plots a scatterplot comparing the fpfs difference value to the index it had in the list of inputs (which were the keys),
providing a visual method of seeing what case had the smallest fpfs difference between a mini-Neptune and Earth.

Parameters:
    fpfs_dict: dictionary
        This saves all the fpfs differences between Earth and your mini-Neptune or planet from grids with their associated inputs as the keys.
    tol: integer
        This limits the y-axis, or fpfs_difference values to zoom into the results in the graph, and visually narrow down which case was minimized.
        All values that are below the tolerance will be plotted in orange. 
    full_plot:
        This still plots values outside of the tolerance in blue.

Returns:
    minNep_inputs_float: multidimensional list (2D mostly)
        This provides all inputs whose fpfs differences were below the tolerance
    fpfs_values_min: 2D list
        This provides all fpfs differences that were below the tolerance
    index_min: 2D list
        This provides the index of all inputs whose fpfs differences were below the tolerance
"""

    index = np.linspace(0, len(fpfs_dict.keys()) - 1, len(fpfs_dict.keys()))
    
    fpfs_values = []
    fpfs_values_min = []
    index_min = []
    minNep_inputs = []
    
    index_value = -1
    for key in fpfs_dict.keys():
        index_value += 1
        value = fpfs_dict[key]
        fpfs_values.append(value)

        if tol is not None:
            if value < tol:
                fpfs_values_min.append(value)
                index_min.append(index_value)
                minNep_inputs.append(key)
        else:
            fpfs_values_min.append(value)
            index_min.append(index_value)
            minNep_inputs.append(key)

            
    plt.figure()

    minNep_inputs_float = []

    for key in minNep_inputs:
        
        # Remove parentheses
        cleaned_s = key.strip('()')
        
        # Split the string into a list of number strings
        number_strings = cleaned_s.split(', ')
        
        # Convert each string to a float using a list comprehension
        float_array = [float(num_str) for num_str in number_strings]

        # This converts the inputs from a string back to a list with floats. 
        minNep_inputs_float.append(float_array)


    if full_plot==True:
        plt.scatter(index, fpfs_values, c='blue')
        plt.scatter(index_min, fpfs_values_min, c='orange')
    else:
        plt.scatter(index_min, fpfs_values_min, c='orange')

    return minNep_inputs_float, fpfs_values_min, index_min


def graph_1v1_planet_comp(minNep_inputs_float, wv_earth, fpfs_earth, earth_type=None):

"""
This plots the miniNeptune reflected spectra that was optimized via the graph_vec_norms function,
and compares it to a reflected spectra of Earth.

Parameters:
    minNep_inputs_float: 2D list
        Pull this from running graph_vec_norm_fpfs.
    wv_earth: list of floats
        These are the wavenumbers you wish to find associated vector norms of for fpfs values.
        Typically in our notebooks we used earth wv and were searching for associated mini-Neptune wavenumbers to
        Earths.
    fpfs_earth: list of floats
        These are the flux planet/star ratios associated with the wv provided, in our case of Earth or whatever
        you want to compare with your mini-Neptunes. 
    earth_type: 'Archean' or 'Earth' strings
        There are default settings for either Archean or Earth, and changes the labels in the graph
"""
    
    for index in minNep_inputs_float:
        flux = index[0]
        metal = index[1]
        tint = index[2]
        kzz = index[3]
        phase = index[4]
    
        fpfs_interpolated, fpfs, wno = mini_nep_model(wv=wv_modern, flux=flux, metal=metal, tint=tint, kzz=kzz, phase=phase)
    
        plt.figure()
        plt.title(f'{flux, metal, tint, kzz, phase}')
        plt.plot(1e4/wv_earth,fpfs_earth,ls='-',marker='', color='cyan', label=f'{earth_type} Earth')
        plt.plot(1e4/wno, fpfs_interpolated,ls='-',marker='', color='blue', label='Mini Nep')
        
        #plt.plot(1e4/wv_modern, fpfs_interpolated, ls='--', color='red')
        plt.legend()

def calc_RSM_earth_phases(df_mol_earth=None, phase_earth=None, earth_RSM_dict={}, earth_type='Archean'):

"""
This will calculate the reflected light spectra of multiple phases of Earth, for different types of Earth.

Parameters:
    df_mol_earth: dict
        This should be a dictionary of molecules and their abundances as values (between 0 and 1).
    phase_earth: 2D list
        This should be a list of phases you wish to iterate over in radians.
    earth_RSM_dict: by default, an empty dictionary
        You could, in theory, add to an already existing earth_RSM_dictionary here if this function has already been run
        for one earth case like Archean.
    earth_type: string
        This is the type of earth whose molecular abundance you are simulating. Note the rest of the code in this script
        focuses on Modern and Archean earths. 
Returns:
    earth_RSM_dict: dictionary
        The keys each describe the type of earth, wv/fpfs/alb value, and phase associated with that are, 
        while the values are the wv, fpfs, and alb values recorded from the reflected light spectra calculation.
"""

    if phase_earth == None:
        phase_earth = np.linspace(0, np.pi, 19)
        phase_angle = phase_earth[:-1]
    else:
        phase_angle = phase_earth
        earth_RSM_dict = {}

    if df_mol_earth == None:
        
        df_mol_archean_earth = {
                "N2":0.945,
                "CO2":0.05,
                "CO":0.0005,
                "CH4":0.005, 
                "H2O":0.003
            }

        df_mol_earth = df_mol_archean_earth
        
    else:

        df_mol_earth = df_mol_earth
    
    for phase in phase_angle:
        
        res_earth = Reflected_Spectra.make_case_earth(df_mol_earth=df_mol_earth, phase=phase)
        res_earth = Reflected_Spectra.make_case_earth(df_mol_earth=df_mol_earth, phase=phase)
        
        wv = res_earth['all'][0]
        fpfs = res_earth['all'][1]
        alb = res_earth['all'][2]
    
        earth_RSM_dict[f"{earth_type}_wv_{phase}"] = wv
        earth_RSM_dict[f"{earth_type}_fpfs_{phase}"] = fpfs
        earth_RSM_dict[f"{earth_type}_alb_{phase}"] = alb
    
    with open("earth_diff_phases.pkl", "wb") as file:
        pickle.dump(earth_RSM_dict, file)
        print(f'File has been recorded, closing.')
        file.close()

    return earth_RSM_dict

def restructure_objective_res(minNep_inputs_float, wv_earth):

    """
    This restructures the objective results by resolving for the flux of planet/star ratio and wavenumber
    values associated with the reflected spectra calculations. 

    Parameters:
        minNep_inputs_float: graph_vec_norm_fpfs output
        wv_earth: list
            This is the wavenumber of earth that you want to solve for fpfs differences with
    
    """
    
    fpfs_interpolated_minNep = []
    wno_interpolated_minNep = []
    input_assos_minNep = []
    
    for index in minNep_inputs_float:
        flux = index[0]
        metal = index[1]
        tint = index[2]
        kzz = index[3]
        phase = index[4]
    
        fpfs_interpolated, fpfs, wno = mini_nep_model(wv=wv_earth, flux=flux, metal=metal, tint=tint, kzz=kzz, phase=phase)
        fpfs_interpolated_minNep.append(fpfs_interpolated)
        wno_interpolated_minNep.append(wno)
        input_assos_minNep.append(index)

    return fpfs_interpolated_minNep, wno_interpolated_minNep, input_assos_minNep

def plot_RSM_earthphases(phase_earth, fpfs_earth, wv_earth, fpfs_minNep, wno_minNep, input_assos_minNep, type_earth='None', lim_earth_rang=None, lim_minNep_rang=None):

    """
    This plots multiple mini Neptune structures compared with a range of reflected spectras of Earth at different phases. 

    Parameters:
        phase_earth: np.linspace array
            This provides all the phases used to calculate reflected spectra for Earth, in radians
        fpfs_earth: 2D list
            This is a list of all flux of planet/star ratios associated with wv_earth and phases of Earth.
        wv_earth: 2D list
            This is the list of wavenumbers explored for various phases of Earth.
        fpfs_minNep: 2D list
            This is the list of flux of planet/star ratios associated with the miniNeptunes you wish to graph.
        wno_minNep: 2D list
            This is the list of wavenumbers associated with mini Neptune planet/star flux ratios.
        input_assos_minNep: 2D list
            These are the integer indices associated with what key inputs and values the mini Neptunes had.
        type_earth: string
            These represent what type of earth was being described by the fpfs_earth, wv_earth, etc., values. 
            We mostly used Archean and Modern Earth.
        lim_earth_rang: integer
            This limits how many phases of Earth you actually plot
        lim_minNep_rang: integer
            This limits how many miniNeptunes you actually plot
    """

    print(f" This is the length right before plotting mini Neptunes: {len(wno_minNep), len(fpfs_minNep)}")
    
    # This creates a new plot for every input value used from the Mini Neptune Grid
    index = -1
 
    if lim_minNep_rang == None:
        for inputs in input_assos_minNep:
            index += 1
            plt.figure(figsize=(10,10))
            plt.title(f'Earth with different phases')
            plt.xlabel(f'Wavelength in microns')
            plt.ylabel(f'Flux Ratio of planet/star')
            plt.plot(1e4/wno_minNep[index], fpfs_minNep[index], ls='--', marker='', color='red', label=f'{inputs}')
            
            index_earth = np.linspace(0, len(phase_earth) - 1, len(phase_earth), dtype='int')
        
            # Define a list of colors or use a color cycle
            colors = ['green', 'blue', 'purple', 'orange']
            color_cycler = cycle(colors) # For cycling through colors if more data sets than colors
        
        # This then plots a range of phases of Earth on top of the mini-Neptune.
            if lim_earth_rang == None:
                for index_earth in index_earth:
                    plt.plot(1e4/wv_earth[index_earth], fpfs_earth[index_earth], ls='-', marker='', color = next(color_cycler), alpha=0.25, label=f'{type_earth} Earth (phase {phase_earth[index_earth]})')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
            else:
                for index_earth in index_earth[:lim_earth_rang]:
                    plt.plot(1e4/wv_earth[index_earth], fpfs_earth[index_earth], ls='-', marker='', color = next(color_cycler), alpha=0.25, label=f'{type_earth} Earth (phase {phase_earth[index_earth]})')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    elif lim_minNep_rang is not None:
        for inputs in input_assos_minNep[:lim_minNep_rang]:
            index += 1
            plt.figure(figsize=(10,10))
            plt.title(f'Earth with different phases')
            plt.xlabel(f'Wavelength in microns')
            plt.ylabel(f'Flux Ratio of planet/star')
            plt.plot(1e4/wno_minNep[index], fpfs_minNep[index], ls='--', marker='', color='red', label=f'{inputs}')
            
            index_earth = np.linspace(0, len(phase_earth) - 1, len(phase_earth), dtype='int')
        
            # Define a list of colors or use a color cycle
            colors = ['green', 'blue', 'purple', 'orange']
            color_cycler = cycle(colors) # For cycling through colors if more data sets than colors
        
        # This then plots a range of phases of Earth on top of the mini-Neptune.
            if lim_earth_rang == None:
                for index_earth in index_earth:
                    plt.plot(1e4/wv_earth[index_earth], fpfs_earth[index_earth], ls='-', marker='', color = next(color_cycler), alpha=0.25, label=f'{type_earth} Earth (phase {phase_earth[index_earth]})')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
            else:
                for index_earth in index_earth[:lim_earth_rang]:
                    plt.plot(1e4/wv_earth[index_earth], fpfs_earth[index_earth], ls='-', marker='', color = next(color_cycler), alpha=0.25, label=f'{type_earth} Earth (phase {phase_earth[index_earth]})')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def graph_RSM_arc_mod_earthphase_minNep(filename='earth_diff_phases.pkl', phase_earth=None, wv_earth_minNep=None, minNep_inputs_float=None, type_earth='Modern'):

    """
    Typically, this function is used to call the plot_RSM_earthphases function more easily. It opens saved data for earth at different phases and then plots it against
    the list of miniNeptunes you chose. 

    Parameters:
        filename: string
            This describes the file location of a pickle file containing a dictionary of the reflected spectrum (wv, alb, fpfs) of Earth at different phases values created using the function calc_RSM_earth_phases.
        phase_earth: list/array from np.linspace
            These were the phases used to create the earth_diff_phases file.
        wv_earth_minNep: 2D list
            These are the optimized mini Neptunes found by using the function, graph_vec_norm_fpfs, but specific to the wavenumbers used.
        minNep_inputs_float: 2D list
            These are the optimized mini Neptunes found by using the function, graph_vec_norm_fpfs, but specific to the inputs (keys) of the resulting dictionary.
        type_earth: string
            This describes what type of earth your data was describing from filename, as it helps label the plot axis and legends.  Note the only
            available data options for this function are 'Archean' and 'Modern' earth. 
    """

    # This function is specific to archean and modern Earth; this will change with alterations to the keys of the earth_diff_phases file.
    wv_archean_list = []
    fpfs_archean_list = []
    wv_modern_list = []
    fpfs_modern_list = []
    
    with open('earth_diff_phases.pkl', 'rb') as file:
        earth_dict = pickle.load(file)
        for phase in phase_earth:
            for key in list(earth_dict.keys()):
                if key.endswith(f'_{phase}'):
                    if key.startswith(f'Archean_wv_'):
                        wv_archean = earth_dict[key]
                        wv_archean_list.append(wv_archean)
                    if key.startswith(f'Archean_fpfs_'):
                        fpfs_archean = earth_dict[key]
                        fpfs_archean_list.append(fpfs_archean)
                    if key.startswith(f'Modern_wv_'):
                        wv_modern = earth_dict[key]
                        wv_modern_list.append(wv_modern)
                    if key.startswith(f'Modern_fpfs_'):
                        fpfs_modern = earth_dict[key]
                        fpfs_modern_list.append(fpfs_modern)
                    
    # This plots the reflected light spectra of the mini Neptune cases listed in minNep_inputs_float

    if type_earth=='Modern':
        fpfs_interpolated_minNep, wno_interpolated_minNep, input_assos_minNep = restructure_objective_res(minNep_inputs_float=minNep_inputs_float, wv_earth=wv_earth_minNep)
        plot_RSM_earthphases(phase_earth=phase_earth, fpfs_earth=fpfs_modern_list, wv_earth=wv_modern_list, fpfs_minNep=fpfs_interpolated_minNep, wno_minNep=wno_interpolated_minNep, input_assos_minNep=input_assos_minNep, type_earth='Modern')
        
    elif type_earth=='Archean':
        fpfs_interpolated_minNep, wno_interpolated_minNep, input_assos_minNep = restructure_objective_res(minNep_inputs_float=minNep_inputs_float, wv_earth=wv_earth_minNep)
        plot_RSM_earthphases(phase_earth=phase_earth, fpfs_earth=fpfs_archean_list, wv_earth=wv_archean_list, fpfs_minNep=fpfs_interpolated_minNep, wno_minNep=wno_interpolated_minNep, input_assos_minNep=input_assos_minNep, type_earth='Archean')
        
    else:
        print(f"Only available types of Earth for this function are Archean and Modern.")

def graph_hist_input_comp(archean_earth_minNep_inputs_float, modern_earth_minNep_inputs_float):

    """
    This plots a histogram comparison of archean and modern earth inputs that were optimized by the graph_vec_norm_fpfs function,
    specific to Modern and Archean Earth. Works best for optimization problems where multiple cases were considered "optimized". 
    The histogram itself compares inputs of the mini Neptunes that best fit the Archean vs Modern earth reflected light spectras. 

    Parameters:
        archean_earth_minNep_inputs_float = output of graph_vec_norm_fpfs w/ archean earth data
        modern_earth_minNep_inputs_float = output of graph_vec_norm_fpfs w/ modern earth data
    """
    
    # This shows how many of the same inputs resulted in an output close to the Earth like spectrum.
    keys_list_float = modern_earth_minNep_inputs_float # Modern Earth Matches
    keys_list_float_archean = archean_earth_minNep_inputs_float # Archean Earth Matches
    
    flux_values_modern_earth = [sublist[0] for sublist in keys_list_float if sublist] 
    metal_values_modern_earth = [sublist[1] for sublist in keys_list_float if sublist] 
    tint_values_modern_earth = [sublist[2] for sublist in keys_list_float if sublist] 
    kzz_values_modern_earth = [sublist[3] for sublist in keys_list_float if sublist]
    phase_values_modern_earth = [sublist[4] for sublist in keys_list_float if sublist]
    
    flux_values_archean_earth = [sublist[0] for sublist in keys_list_float_archean if sublist] 
    metal_values_archean_earth = [sublist[1] for sublist in keys_list_float_archean if sublist] 
    tint_values_archean_earth = [sublist[2] for sublist in keys_list_float_archean if sublist] 
    kzz_values_archean_earth = [sublist[3] for sublist in keys_list_float_archean if sublist] 
    phase_values_archean_earth = [sublist[4] for sublist in keys_list_float_archean if sublist] 
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1, figsize=(20,10))
    ax1.set_title(f"Total Flux:")
    ax1.hist(flux_values_modern_earth, color='blue', align='mid', label='Matched Modern Earth', bins=10)
    ax1.hist(flux_values_archean_earth, color='skyblue',  align='mid', label='Matched Archean Earth', bins=10)
    ax1.legend()
    
    ax2.set_title(f"Planet Metallicity:")
    ax2.hist(metal_values_modern_earth, color='red',  align='mid', label='Matched Modern Earth', bins=10)
    ax2.hist(metal_values_archean_earth, color='magenta', label='Matched Archean Earth', bins=10)
    ax2.legend()
    
    ax3.set_title(f"Planet Tint:")
    ax3.hist(tint_values_modern_earth, color='green', align='mid', label='Matched Modern Earth', bins=10)
    ax3.hist(tint_values_archean_earth, color='cyan', label='Matched Archean Earth', bins=10)
    ax3.legend()
    
    ax4.set_title(f"Planet Kzz:")
    ax4.hist(kzz_values_modern_earth, color='black', align='mid', label='Matched Modern Earth', bins=10)
    ax4.hist(kzz_values_archean_earth, color='grey', label='Matched Archean Earth', bins=10)
    ax4.legend()
    
    ax5.set_title(f"Planet Phase:")
    ax5.hist(phase_values_modern_earth, color='orange',  align='mid', label='Matched Modern Earth', bins=10)
    ax5.hist(phase_values_archean_earth, color='pink',  align='mid', label='Matched Archean Earth', bins=10)
    ax5.legend()
    
    plt.subplots_adjust(wspace=0.5, hspace=2) # Increase horizontal and vertical space

def vec_norm_fpfs_minimized(fpfs_dict=None):

    """
    This takes the output from the function, calc_objective_dict, and solves for the minimized miniNeptune fpfs differences case, and
    makes it so you can skip the graph_vec_norm step of optimization returning the same outputs as graph_vec_norm did.

    Parameters:
        fpfs_dict: dictionary
            This is the output of the calc_objective_dict function.
    Returns:
        minNep_inputs_float: multidimensional list (2D mostly)
        This provides all inputs whose fpfs differences were associated with the smallest fpfs difference between a mini Neptune and Earth.
    fpfs_values_min: 2D list
        This provides the fpfs values associated with the smallest fpfs difference between a mini Neptune and Earth.
    index_min: 2D list
        This provides the index of all inputs whose fpfs differences were minimized. 

    """
    
    fpfs_values = []
    fpfs_values_min = []
    index_min = []
    minNep_inputs = []
    minNep_inputs_float = []
    
    index_value = -1
    for key in fpfs_dict.keys():
        index_value += 1
        value = fpfs_dict[key]
        fpfs_values.append(value)
        minNep_inputs.append(key)

    for key in minNep_inputs:
        
        # Remove parentheses
        cleaned_s = key.strip('()')
        
        # Split the string into a list of number strings
        number_strings = cleaned_s.split(', ')
        
        # Convert each string to a float using a list comprehension
        float_array = [float(num_str) for num_str in number_strings]

        # This converts the inputs from a string back to a list with floats. 
        minNep_inputs_float.append(float_array)


    minimized_fpfs_value = min(fpfs_values)
    minimized_index = fpfs_values.index(minimized_fpfs_value)

    minNep_inputs_float_min = minNep_inputs_float[minimized_index]

    return minNep_inputs_float_min, minimized_fpfs_value, minimized_index

def find_closest(sorted_list, x):

    """
    This finds the closest values in the sorted list, on the left and right, that x comes close to.

    Parameters:
        sorted_list: list
        x: float
    
    Returns:
        Closest value, either on the left or right, that x is to in the sorted list. 

    """

    # Find the index where x would be inserted in sorted_list
    insertion_point = np.searchsorted(np.array(sorted_list), x, side='left')
    
    # If x is smaller than all elements, the first element is the closest
    if insertion_point == 0:
        return sorted_list[0]
    
    # If x is larger than all elements, the last element is the closest
    if insertion_point == len(sorted_list):
        return sorted_list[-1]
    
    # Otherwise, compare the element to the left and right of the insertion point
    left_val = sorted_list[insertion_point - 1]
    right_val = sorted_list[insertion_point]
    
    if x - left_val <= right_val - x:
        return left_val
    else:
        return right_val


def check_for_cloud(minNep_inputs):

    """
    This finds the closest values in the Reflected Spectra Grid, on the left and right, that the miniNeptune inputs found in optimization come close to.

    Parameters:
        minNep_inputs: list
        Should provide a list of input parameters [flux, metallicity, tint, kzz, phase], that you would like to compare to those that were used to create the Reflected Spectra Grid.
    
    Returns:
        boolean
        This describes whether the closest value used in the grid had a cloud or not (had = True, not = False), in order to determine whether an interpolated case should have imposed
        a cloud or not when looking at results. This is part of the module to check whether or not clouds have been properly applied. 
        
    """

    # Set full grid of Reflected Spectra results
    filename='results/ReflectedSpectra_fv.h5'
    gridvals=Reflected_Spectra.get_gridvals_RSM()
    
    gridvals_metal = [float(s) for s in gridvals[1]]
    gridvals_dict = {'total_flux':gridvals[0], 'planet_metallicity':np.array(gridvals_metal), 'tint':gridvals[2], 'kzz':gridvals[3], 'phase':gridvals[4]}
    index_flux = np.linspace(0, len(gridvals_dict['total_flux']) - 1, len(gridvals_dict['total_flux']), dtype=int)
    index_metal = np.linspace(0, len(gridvals_dict['planet_metallicity']) - 1, len(gridvals_dict['planet_metallicity']), dtype=int)
    index_tint = np.linspace(0, len(gridvals_dict['tint']) - 1, len(gridvals_dict['tint']), dtype=int)
    index_kzz = np.linspace(0, len(gridvals_dict['kzz']) - 1, len(gridvals_dict['kzz']), dtype=int)
    index_phase = np.linspace(0, len(gridvals_dict['phase']) - 1, len(gridvals_dict['phase']), dtype=int)
    true_converg_list = np.array([1])

    # Find the closest inputs to the values inputed by the user
    for input_value in minNep_inputs:
        print(input_value)
    
        user_flux = input_value[0]
        user_metal = input_value[1]
        user_tint = input_value[2]
        user_kzz = input_value[3]
        user_phase = input_value[4]

        # Check for closest index values
        closest_flux = find_closest(gridvals_dict['total_flux'], user_flux)
        closest_metal = find_closest(gridvals_dict['planet_metallicity'], user_metal)
        closest_tint = find_closest(gridvals_dict['tint'], user_tint)
        closest_kzz = find_closest(gridvals_dict['kzz'], user_kzz)
        closest_phase = find_closest(gridvals_dict['phase'], user_phase)

        input_list = [closest_flux, closest_metal, closest_tint, closest_kzz, closest_phase]

        print(closest_flux, closest_metal, closest_tint, closest_kzz, closest_phase)
    
        matching_indicies_clouds = []
        matching_values_clouds = []
        
        with h5py.File(filename, 'r') as f:
            
            for flux in index_flux:
                for metal in index_metal:
                    for tint in index_tint:
                        for kzz in index_kzz:
                            for phase in index_phase:
                                if np.array(list(f['results']['clouds'][flux][metal][tint][kzz][phase]))[0] == true_converg_list:
                                    index = np.array([flux, metal, tint, kzz, phase])
                                    matching_indicies_clouds.append(index)

            for array in matching_indicies_clouds:
                
                flux_index = array[0]
                metal_index = array[1]
                tint_index = array[2]
                kzz_index = array[3]
                phase_index = array[4]
                
                flux_value = gridvals_dict['total_flux'][flux_index]
                metal_value = gridvals_dict['planet_metallicity'][metal_index]
                tint_value = gridvals_dict['tint'][tint_index]
                kzz_value = gridvals_dict['kzz'][kzz_index]
                phase_value = gridvals_dict['phase'][phase_index]
                list_value = [flux_value, metal_value, tint_value, kzz_value, phase_value]
                matching_values_clouds.append(list_value)
                
        
            print(len(matching_indicies_clouds), len(matching_values_clouds))
        
            print(f'Cases with clouds totaled: {len(matching_values_clouds)}')

        # Then check to see if my closest inputs were included in this list of cases with clouds.
        if input_list in matching_values_clouds:
            print(f'The nearest input in the grid found, {input_list}, did have clouds.')
            return True
        else: 
            print(f'The nearest input in the grid found, {input_list}, did not have clouds.')
            return False

def calc_sol_dict(minNep_inputs):

    """
    This resolves for the photochemistry and PT profiles of the optimized miniNeptune inputs. 
    This was mostly made for easier plotting for changing the imposed cloud.

    Parameters:
        minNep_inputs: [flux, metallicity, tint, kzz, phase]
        Note that the units of each are always the same, in xSolar, log10 xSolar, Kelvin, cm^2/s, and radians, annd are all related to the planet.
    
    Returns:
        sol_dict: dictionary
            Each key contains abundances of molecules in a steady state.
        solaeq_dict: dictionary
            Each key contains abundances of molecules in chemical equilibrium.
        PT_list_Photochem: 2D list
            First list is pressure (dynes/cm^2), second is temperature (Kelvin)
    """
    
    for input_value in minNep_inputs:
        
        flux_index = input_value[0]
        metal_index = input_value[1]
        tint_index = input_value[2]
        kzz_index = input_value[3]
        phase_index = input_value[4]
    
        PT_list, sol_dict, soleq_dict, wno, albedo, fpfs, PT_list_Photochem = GraphsKey.find_all_plotting_values(total_flux=flux_index, planet_metal=metal_index, tint=tint_index, kzz=kzz_index, phase=phase_index, calc_PT=True, calc_PhotCh=True, calc_RSM=True)

        new_sol_dict = {}
        new_solaeq_dict = {}
        
        for key in sol_dict.keys():
            
            if key.endswith('_sol'):
                sol_dict[key] = list(sol_dict[key])
            if key == 'pressure_sol':
                sol_dict[key] = list(sol_dict[key])
            if key == 'temperature_sol':
                sol_dict[key] = list(sol_dict[key])

        sol_dict = {key.replace('_sol', '') if key.endswith('_sol') else key: value
        for key, value in sol_dict.items()}

        for key in sol_dict.keys():
            if key.endswith('aer'):
                new_solaeq_dict[key] = list(sol_dict[key])
            if key == 'pressure':
                new_solaeq_dict[key] = list(sol_dict[key])
            if key == 'temperature':
                new_solaeq_dict[key] = list(sol_dict[key])
            else:
                continue

        for key in sol_dict.keys():
            if key.endswith('aer'):
                continue
            else:
                new_sol_dict[key] = list(sol_dict[key])

    return new_sol_dict, new_solaeq_dict, PT_list_Photochem


def plot_photochem_model_with_cloud(minNep_inputs):

    """
    This plots a single mini Neptune case with its PT profile, Photochemical abundances, and a top and bottom layer representing the cloud that was implemented (if it had a cloud).

    Parameters:
        minNep_inputs: [flux, metallicity, tint, kzz, phase]
            Note that the units of each are always the same, in xSolar, log10 xSolar, Kelvin, cm^2/s, and radians, annd are all related to the planet.
    """

    cloud_check = check_for_cloud(minNep_inputs = minNep_inputs)
    print(cloud_check)
    sol_dict, solaer_dict, PT_list_Photochem = calc_sol_dict(minNep_inputs)

    if cloud_check is True:

        pbot = Reflected_Spectra.find_pbot(sol=sol_dict, solaer=solaer_dict)

        sol = sol_dict.copy()
        for key in solaer_dict.keys():
            if key.endswith('aer'):
                sol[key] = list(solaer_dict[key])

        print(sol.keys())
        
        # Plot the Composition from Photochem
        fig, ax1 = plt.subplots(1,1,figsize=[8,6])
        species = ['CO2','H2O','CH4','CO','NH3','H2','HCN', 'H2Oaer']
         
        for i,sp in enumerate(species):
            ax1.plot(sol[sp],np.array(sol['pressure'])/1e6, c='C'+str(i), label=sp)
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim(1e-8,1)
        ax1.set_ylim(1000,1e-7)
        ax1.grid(alpha=0.4)
        ax1.set_xlabel('Mixing Ratio', fontsize=16)
        ax1.set_ylabel('Pressure (bar)', fontsize=16)
        ax1.set_yticks(10.0**np.arange(-6,2))
        ax1.tick_params(axis='x', labelsize=16) 
        ax1.tick_params(axis='y', labelsize=16)
        
        
        # Thickness of the box cloud
        ptop_earth = 0.6
        pbot_earth = 0.7
        logdp = np.log10(pbot_earth) - np.log10(ptop_earth)
        
        # Outline of the box cloud
        x_values = np.logspace(-8, 0, 10)
        y_values_bot = [pbot/10**6] * len(x_values)
        y_values_top = [(pbot/10**6) - ((10**(np.log10(pbot/10**6) - logdp)))] * len(x_values)
        
        ax1.plot(x_values, y_values_bot, c='black', ls='--', label='Cloud Bottom')
        ax1.plot(x_values, y_values_top, c='black', ls='--', label='Cloud Top')
        ax1.legend(ncol=1,bbox_to_anchor=(1,1.0),loc='upper left')
         
        ax2 = ax1.twiny()
        ax2.set_xlabel('Temperature (K)', fontsize=16)
        ax2.tick_params(axis='x', labelsize=16)
        ax2.plot(PT_list_Photochem[1], (PT_list_Photochem[0]/(1e6)), c='blue', ls='--',label='Photochem PT Profile')
            
        plt.title('K2-18b Around Sun (G-Star)', fontsize=20)
        plt.tight_layout()
        plt.savefig('K218b_Sun_ReflectedSpectra_Clds.png')
            
        plt.legend()
        
        plt.show()

    else:
        print(f"There were no clouds for this case!")

        sol = sol_dict.copy()
        
        # Plot the Composition from Photochem
        fig, ax1 = plt.subplots(1,1,figsize=[5,4])
        species = ['CO2','H2O','CH4','CO','NH3','H2','HCN']
         
        for i,sp in enumerate(species):
            ax1.plot(sol[sp],np.array(sol['pressure'])/1e6, c='C'+str(i), label=sp)
          
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim(1e-8,1)
        ax1.set_ylim(1000,1e-7)
        ax1.grid(alpha=0.4)
        ax1.set_xlabel('Mixing Ratio')
        ax1.set_ylabel('Pressure (bar)')
        ax1.set_yticks(10.0**np.arange(-6,2))
        ax1.legend(ncol=1,bbox_to_anchor=(1,1.0),loc='upper left')
        
        ax2 = ax1.twiny()
        ax2.set_xlabel('Temperature (K)')
        ax2.plot(PT_list_Photochem[1], (PT_list_Photochem[0]/(1e6)), c='blue', ls='--',label='Photochem PT Profile')
            
        plt.title('K2-18b Around Sun (G-Star)')
            
        plt.legend()
        
        plt.show()
   

def solve_optimal_graph_bundle(minNep_modernEarth, minNep_archeanEarth, lim_minNep):

    """
    Not completed, ran out of time, but good idea. 
    
    Steps Towards Optimization (Do BEFORE calling this function):
    1. Calculate fpfs and wv of earth using Reflected_Spectra.make_case_earth function
    2. fpfs_dict = calc_objective_dict(wv_earth, fpfs_earth, resolution=5)
    3. minNep_inputs_float, fpfs_values_min, index_min = graph_vec_norm_fpfs(fpfs_dict=None, tol=None, full_plot=True) to limit what input values you wish to investigate further based on their objective calculation via the vector norm method. 
    4. Make sure that you have the earth_diff_phases.pkl file in your directory, if not run the calc_RSM_earth_phases(df_mol_earth=None, phase_earth=None, earth_RSM_dict={}, earth_type='Archean') function for both Archean and Modern specs. Run it for one first, then provide the output as the earth_type input to add another type to the dictionary.

    What this function calls:
    5. graph_RSM_arc_mod_earthphase_minNep(filename='earth_diff_phases.pkl', phase_earth=None, minNep_inputs_float=None, limit_input=30, type_earth='Modern') is what will plot your list of filtered inputs of the mini_Neptune against the different phases of the Earth Spectrum.
    6. graph_1v1_planet_comp(minNep_inputs_float, wv_earth, fpfs_earth, earth_type=None) will give you a one on one comparison to one of the phases of Earth with all inputted values for the mini-Neptune.
    
    """
    
    graph_RSM_arc_mod_earthphase_minNep(filename='earth_diff_phases.pkl', phase_earth=None, minNep_inputs_float=minNep_modernEarth, limit_input=limit_input, type_earth='Modern')
    graph_RSM_arc_mod_earthphase_minNep(filename='earth_diff_phases.pkl', phase_earth=None, minNep_inputs_float=minNep_archeanEarth, limit_input=lim_minNep, type_earth='Archean')       

# The following functions help solve an optimization problem incorporating a single point & part of the spectrum
def flux_in_wv_range(wv, F, wv_range):

    """Compute the flux passing through `wv_range`

    Parameters:
        wv : ndarray
            Center of wavelengths bins
        F : ndarray
            Flux going through each bin.
        wv_range : ndarray
            An array of length 2 that gives a wavelength range.

    Returns:
        F_in_wv_range : float
            The flux passing through wv_range
    """
    assert np.all(wv[1:] > wv[:-1])
    print(len(wv_range), wv_range)
    assert len(wv_range) == 2
    wavl = stars.make_bins(wv)
    print(wv_range[0], wavl[0])
    assert wv_range[0] >= wavl[0]
    assert wv_range[-1] <= wavl[-1]
    F_in_wv_range = rebin(wavl.copy(), F.copy(), wv_range.copy())[0]
    return F_in_wv_range


def calc_objective_dict_point_wv_range(wv_earth, fpfs_earth, resolution=5, total_flux_list=None, planet_metal_list=None, tint_list=None, kzz_list=None, phase_list=None, bin_lim=[[.5, .6]], wv_lim=[[0.87, 1.2]]):


    """
    This function calculated the optimal, through vector norms, mini Neptune match to an Earth reflected spectrum, with the added option of limiting the wavelengths compared or binning 
    the fpfs (flux of planet/star ratios) values over a wavelength range into a single point.

    Parameters:
        wv_earth: 2D list
            This is the list of wavenumbers explored for various phases of Earth.
        fpfs_earth: 2D list
            This is the list of flux of planet/star ratios associated with the Earths you used for wv_earth.
        resolution: integer
            This is how many values/input to range over when exploring the mini Neptune grids. If you don't wish to use the default
            values that was using the mini-Neptune space I used for my project, you can manually request flux, metallicity, tint, kzz, and phase
            via the list parameters and set this one to None. 
        total_flux_list: 1D list of floats
            Total fluxes you wish to range over. Should be in units of x Solar total flux.
        planet_metal_list: 1D list of floats or np.linspace
            Planet metallicities you wish to range over. Should be in units of log10 x Solar Metallicity.
        tint_list: list of floats or np.linspace
            Internal temperature of the planet you wish to range over. Should be in units of Kelvin.
        kzz_list: 1D list of floats or np.linspace
            Eddy diffusion coefficient in units of cm^2/s, also in log10 space. 
        phase_list: 1D list of floats or np.linspace
            Phase of light visible from planet in orbit, in units of radians. 
        bin_lim: 2D list
            This provides the wavelength range, in microns, you wish to bin fpfs values into a single point
        wv_lim: 2D list
            This provides the wavelength range you wish to limit the reflected spectra being compared between for both Earth and the mini-Neptune.
        
    Returns:
        fpfs_dict: dictionary
            Saves all fpfs differences values for mini Neptunes compared to Earth for the full spectrum, limited wavelength range, and binned points. 

            fpfs_dict[f'{z}'] = fpfs_difference_grid, provides the fpfs difference between the full spectrum of the mini Neptune and of Earth
            fpfs_dict[f'fpfsrebinned_diff_{z}_{bin_lim_range}'] = fpfs_difference_grid, provides the fpfs difference between the binned spectrum of the mini Neptune and Earth
            fpfs_dict[f'fpfs_wvlim_{z}_{wv_lim_range}'] = fpfs_difference_grid, provides the fpfs difference between the wavelength limited spectrum of the mini Neptune and Earth

        fpfs_wv_dict: dictionary
            Saves all wavelength related values and fpfs values, including those that have been binned into the single point, those between the wavelength range, and those without any limitations just
            based on the initial inputs you entered describing the mini Neptune.

            'wv_earth_wv': wv_earth_in_wv_lim, provides the Earth wavelength in microns in the wavelength limit
            'wv_minNep_wv': wv_minNep_in_wv_lim, provides the mini Neptune wavelength in microns in the wavelength limit
            'fpfs_earth_wv': fpfs_earth_in_wv_lim, provides the Earth fpfs in the wavelength limit
            'fpfs_minNep_wv': fpfs_minNep_in_wv_lim, provides the mini Neptune fpfs in the wavelength limit
            'wv_earth_bin': wv_earth_in_bin_lim, provides the Earth wavelength in microns in the bin limit
            'wv_minNep_bin': wv_minNep_in_bin_lim, provides the mini Neptune wavelength in microns in the bin limit
            'fpfs_minNep_bin': fpfs_minNep_in_bin_lim, provides the mini Neptune fpfs in the bin limit
            'fpfs_earth_bin':fpfs_earth_in_bin_lim, provides the Earth fpfs in the bin limit
            'wv_earth_full': wv_earth, provides the full wavelength spectrum of Earth (in microns)
            'fpfs_earth_full': fpfs_earth, provides the full fpfs spectrum of Earth
            
            These are repeat values of the binning wavelengths and fpfs values, but for now kept them here to keep Jupyter Notebooks Running.
            'wv_earth_bin_lim': wv_bin_earth_wv_lim, provides the Earth wavelength in microns in the bin limit 
            'fpfs_earth_bin_lim': fpfs_bin_earth_wv_lim, 
            'wv_minNep_bin_lim': wv_minNep_in_bin_lim, 
            'fpfs_minNep_bin_lim': fpfs_minNep_in_bin_lim                            
"""
    
    if total_flux_list is not None:
        total_flux = total_flux_list
    else:
        total_flux = np.linspace(0.1, 2, resolution)
    
    if planet_metal_list is not None:
        planet_metal = planet_metal_list
    else:
        planet_metal = np.linspace(0.5, 2, resolution)
        
    if tint_list is not None:
        tint = tint_list
    else:
        tint = np.linspace(20, 200, resolution)
        
    if kzz_list is not None:
        kzz = kzz_list
    else:
        kzz = np.linspace(5, 9, resolution)
        
    if phase_list is not None:
        phase = phase_list
    else:
        phase = np.linspace(0, 2.9, resolution)
    
    fpfs_dict = {}
    
    for flux in total_flux:
        for metal in planet_metal:
            for tint_val in tint:
                for kzz_val in kzz:
                    for phase_val in phase:
                        z = flux, metal, tint_val, kzz_val, phase_val
                        fpfs_difference_grid = objective(z=z, wv=wv_earth, fpfs_earth=fpfs_earth)
                        fpfs_dict[f'{z}'] = fpfs_difference_grid

    fpfs_wv_dict = None

    if bin_lim is not None and wv_lim is not None:

        wv_earth_in_bin_lim = []
        fpfs_earth_in_bin_lim = []
        
        wv_minNep_in_bin_lim = []
        fpfs_minNep_in_bin_lim = []

        fpfs_bin_earth_wv_lim = []
        wv_bin_earth_wv_lim = []

        # When solving for the fpfs difference of a single rebinned point on earth spectrum vs minNep.
        # Has capability of solving for multiple points in the grid...in theory.

        for bin_lim_range in bin_lim:

            bin_lim_range_converted = 1e+4/np.array(bin_lim_range)
            fpfs_binned_earth = flux_in_wv_range(wv=wv_earth, F=fpfs_earth, wv_range=bin_lim_range_converted)
            fpfs_earth_in_bin_lim.append(fpfs_binned_earth)
            wv_earth_in_bin_lim.append(np.mean(bin_lim_range_converted))
            
            # Then for each of those solutions, we solve for the minNep rebinned point. 
            for flux in total_flux:
                for metal in planet_metal:
                    for tint_val in tint:
                        for kzz_val in kzz:
                            for phase_val in phase:
                                z = flux, metal, tint_val, kzz_val, phase_val
                                fpfs_interp, fpfs_before, wno_before = mini_nep_model(wv=wv_earth, flux=flux, metal=metal, tint=tint_val, kzz=kzz_val, phase=phase_val)
                                
                                bin_lim_range_converted = 1e+4/np.array(bin_lim_range)
                                new_minNep_fpfs_binned = flux_in_wv_range(wv=wv_earth, F=fpfs_interp, wv_range=bin_lim_range_converted)
                                
                                fpfs_minNep_in_bin_lim.append(new_minNep_fpfs_binned)
                                wv_minNep_in_bin_lim.append(np.mean(bin_lim_range_converted))
    
                                fpfs_difference_grid = np.sqrt(np.sum((new_minNep_fpfs_binned - fpfs_earth_in_bin_lim)**2))
    
                                fpfs_dict[f'fpfsrebinned_diff_{z}_{bin_lim_range}'] = fpfs_difference_grid
                                                
        wv_earth_in_wv_lim = []
        fpfs_earth_in_wv_lim = []
        wv_minNep_in_wv_lim = []
        fpfs_minNep_in_wv_lim = []

        for wv_lim_range in wv_lim:
            for wv_earth_val in wv_earth:

                # This solves for the fpfs differences of a specific range of wavelengths for a snapshot of the spectra
                if 1e4/wv_earth_val >= (wv_lim_range[0]) and 1e4/wv_earth_val <= (wv_lim_range[1]):

                    index = list(wv_earth).index(wv_earth_val)
                    fpfs_earth_val = fpfs_earth[index]
                    fpfs_earth_in_wv_lim.append(fpfs_earth_val)
                    wv_earth_in_wv_lim.append(wv_earth_val)

        for flux in total_flux:
            for metal in planet_metal:
                for tint_val in tint:
                    for kzz_val in kzz:
                        for phase_val in phase:
                            z = flux, metal, tint_val, kzz_val, phase_val
                            fpfs_interp, fpfs_before, wno_before = mini_nep_model(wv=wv_earth_in_wv_lim, flux=flux, metal=metal, tint=tint_val, kzz=kzz_val, phase=phase_val)
                            fpfs_minNep_in_wv_lim.append(fpfs_interp)
                            wv_minNep_in_wv_lim.append(wv_earth_in_wv_lim)
                            fpfs_difference_grid = np.sqrt(np.sum((fpfs_interp - fpfs_earth_in_wv_lim)**2))
                            fpfs_dict[f'fpfs_wvlim_{z}_{wv_lim_range}'] = fpfs_difference_grid

        
        fpfs_wv_dict = {'wv_earth_wv': wv_earth_in_wv_lim, 'wv_minNep_wv': wv_minNep_in_wv_lim,
                        'fpfs_earth_wv': fpfs_earth_in_wv_lim, 'fpfs_minNep_wv': fpfs_minNep_in_wv_lim,
                        'wv_earth_bin': wv_earth_in_bin_lim, 'wv_minNep_bin': wv_minNep_in_bin_lim, 
                        'fpfs_minNep_bin': fpfs_minNep_in_bin_lim, 'fpfs_earth_bin':fpfs_earth_in_bin_lim,
                       'wv_earth_full': wv_earth, 'fpfs_earth_full': fpfs_earth, 
                       'wv_earth_bin_lim': wv_bin_earth_wv_lim, 'fpfs_earth_bin_lim': fpfs_bin_earth_wv_lim, 
                       'wv_earth_bin_lim': wv_bin_earth_wv_lim, 'fpfs_earth_bin_lim': fpfs_bin_earth_wv_lim, 
                       'wv_minNep_bin_lim': wv_minNep_in_bin_lim, 'fpfs_minNep_bin_lim': fpfs_minNep_in_bin_lim}
        
    return fpfs_dict, fpfs_wv_dict


def fpfs_minimized_wvbinlim(fpfs_diff_dict=None):

    """
    This provides the minimized mini Neptune value in the wavelength range limit, as well as the fpfs values binned.
    This was an initial attempt to see if I could order the values from smallest to largest, when next steps would be
    giving both cases equal weight in deciding the ultimate match of the mini Neptune to Earth. 

    Parameters:
        fpfs_diff_dict: dictionary
            output of calc_objective_dict_point_wv_range
    
    Returns:
        index_wv_range: list
            This provides the index of wavelength when looking at mini Neptune values limited by the wavelength range.
        index_point: list
            This provides the index of the wavelengths/fpfs/input data of the mini Neptune when limiting by binned point.
        fpfs_diff_point: list
            This provides the fpfs difference between Mini Neptune and Earth associated with the binning method.
        fpfs_diff_wv_lim: list
            This provides the fpfs value(s) differences between Mini Neptunes and Earths associated with the wavelength range method.
        minNep_inputs: list
            This logs what inputs are associated with what fpfs, fpfs_diff, and index values.

    """
    minNep_inputs_float = []
    
    fpfs_diff_point = []
    fpfs_diff_wv_lim = []
    index_min_point = []
    index_min_wv_lim = []
    minNep_inputs = []
   
    minNep_inputs_min = []

    index_value = -1
    for key in fpfs_diff_dict.keys():
        index_value += 1
        if key.startswith('fpfsrebinned_diff'):
            print(f'starts with fpfsrebin: {key}')
            value = fpfs_diff_dict[key]
            fpfs_diff_point.append(value)
        if key.startswith('fpfs_wvlim'):
            print(f'starts with fpfs_wvlim: {key}')
            wvlim_value = fpfs_diff_dict[key]
            fpfs_diff_wv_lim.append(wvlim_value)
        if key.startswith('('):
            print(f'other: {key}')
            minNep_inputs.append(key)

    for key in minNep_inputs:
        
        # Remove parentheses
        print(key)
        cleaned_s = key.strip('()')
        
        # Split the string into a list of number strings
        number_strings = cleaned_s.split(', ')
        
        # Convert each string to a float using a list comprehension
        float_array = [float(num_str) for num_str in number_strings]

        # This converts the inputs from a string back to a list with floats. 
        minNep_inputs_float.append(float_array)


    # Then sort the fpfs differences for both methods from smallest to largest, maintaining indicies, and find the closest match
    # for both cases.

    index_point = []
    index_wv_range = []

    for i, value in enumerate(fpfs_diff_point):
        index_point.append((value, i))
        index_point.sort(key=lambda x:[0])
    for i, value in enumerate(fpfs_diff_wv_lim):
        index_wv_range.append((value, i))
        index_wv_range.sort(key=lambda x:[0])

    index_wv_range.sort()
    index_point.sort()

    print(index_wv_range, index_point)

    return index_wv_range, index_point, fpfs_diff_point, fpfs_diff_wv_lim, minNep_inputs