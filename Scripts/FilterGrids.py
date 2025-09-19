import gridutils
import Photochem_grid
import Reflected_Spectra_grid as Reflected_Spectra
import PICASO_Climate_grid
import GraphsKey
import numpy as np
from scipy import interpolate


def find_PT_sol(filepath='/Users/epawelka/Documents/NASA_Ames_ProjS25/AmesProjS25Work/results/PICASO_climate_fv.h5',total_flux=None, log10_planet_metallicity=None, tint=None, grid_gridvals=PICASO_Climate_grid.get_gridvals_PICASO_TP()):

    """
    Inputs:
    log10_totalflux = np.array([1.25])
    log10_planet_metallicity = np.array([0.75])
    tint = np.array([50]) # in Kelvin
    """

    # This takes the inputs that define the grid
    gridshape = tuple(len(a) for a in grid_gridvals)

    print(f"Make sure your inputs are within the following ranges, total_flux: {np.min(grid_gridvals[0])} - {np.max(grid_gridvals[0])} xsolar, planet metallicity: {np.min((grid_gridvals[1]).astype(float))} - {np.max((grid_gridvals[1]).astype(float))} xsolar, tint: {np.min(grid_gridvals[2])} - {np.max(grid_gridvals[2])} K")

    # Check to see if there is a solution that already exists
    PT_list, convergence_values = Photochem_grid.find_PT_grid(filename=filepath, total_flux=total_flux, log10_planet_metallicity=log10_planet_metallicity, tint=tint)

    if PT_list is not None:
        print(f'All inputs chosen were directly on the grid!')
        comb_results = {}
        comb_results['pressure'] = PT_list[0]
        comb_results['temperature'] = PT_list[1]
        # comb_results['converged'] = convergence_values
        return comb_results
        

    else:

        print(f'Interpolating results...')
        
        # This notes the grid and associated inputs used to make it as the data
        PhotCh_grid = gridutils.GridInterpolator(filename=filepath, gridvals=grid_gridvals)

        # This interpolates the results based on the user input
        interp_results = {}
    
        # New grid values to interpolate
        user_gridvals = (total_flux, log10_planet_metallicity, tint)
    
        for key in PhotCh_grid.data.keys():
            if key.startswith('pressure'):
                interp_function_pressure = PhotCh_grid.make_interpolator(key=key, logspace=True)
                interp_results[key] = interp_function_pressure(user_gridvals)
            elif key.startswith('temperature'):
                interp_function_temperature = PhotCh_grid.make_interpolator(key=key, logspace=False)
                interp_results[key] = interp_function_temperature(user_gridvals)
            elif key.startswith('converged'):
                continue
            else:
                interp_function = PhotCh_grid.make_interpolator(key=key, logspace=True)
                interp_results[key] = interp_function(user_gridvals)
                
        return interp_results
        
def find_Photochem_sol(filepath='/Users/epawelka/Documents/NASA_Ames_ProjS25/AmesProjS25Work/results/Photochem_1D_fv.h5',total_flux=None, log10_planet_metallicity=None, tint=None, Kzz=None, grid_gridvals=Photochem_grid.get_gridvals_Photochem()):

    """
    Inputs:
    log10_totalflux = np.array([1.25])
    log10_planet_metallicity = np.array([0.75])
    tint = np.array([50]) # in Kelvin
    kzz = np.array([7]) # in log space
    """

    # This takes the inputs that define the grid
    gridshape = tuple(len(a) for a in grid_gridvals)

    print(f"Make sure your inputs are within the following ranges, total_flux: {np.min(grid_gridvals[0])} - {np.max(grid_gridvals[0])} xsolar, planet metallicity: {np.min((grid_gridvals[1]).astype(float))} - {np.max((grid_gridvals[1]).astype(float))} xsolar, tint: {np.min(grid_gridvals[2])} - {np.max(grid_gridvals[2])} K, kzz: {np.min(grid_gridvals[3])} - {np.max(grid_gridvals[3])} log10 of cm^2/s.")

    # Check to see if there is a solution that already exists
    sol_dict_new, soleq_dict_new, PT_list, convergence_PC, convergence_TP = Reflected_Spectra.find_Photochem_match(filename=filepath, total_flux=total_flux, log10_planet_metallicity=log10_planet_metallicity, tint=tint, Kzz=Kzz)

    if sol_dict_new is not None:
        print(f'All inputs chosen were directly on the grid!')
        comb_results = {}
        sol_dict = {}
        soleq_dict = {}
        for key, value in sol_dict_new.items():
            new_key = key + '_sol'
            sol_dict[new_key] = value
        for key, value in soleq_dict_new.items():
            new_key = key + '_soleq'
            soleq_dict[new_key] = value
        comb_results.update(sol_dict)
        comb_results.update(soleq_dict)
        #comb_results['pressure'] = PT_list[
        # comb_results['converged_PC'] = convergence_PC
        # comb_results['converged_TP'] = convergence_TP
        return comb_results
        

    else:

        print(f'Interpolating results...')
        
        # This notes the grid and associated inputs used to make it as the data
        PhotCh_grid = gridutils.GridInterpolator(filename=filepath, gridvals=grid_gridvals)

        # This interpolates the results based on the user input
        interp_results = {}
    
        # New grid values to interpolate
        user_gridvals = (total_flux, log10_planet_metallicity, tint, Kzz)
    
        for key in PhotCh_grid.data.keys():
            if key.startswith('pressure'):
                interp_function_pressure = PhotCh_grid.make_interpolator(key=key, logspace=True)
                interp_results[key] = interp_function_pressure(user_gridvals)
            elif key.startswith('temperature'):
                interp_function_temperature = PhotCh_grid.make_interpolator(key=key, logspace=False)
                interp_results[key] = interp_function_temperature(user_gridvals)
            elif key.startswith('converged'):
                continue
            else:
                interp_function = PhotCh_grid.make_interpolator(key=key, logspace=True) # This is giving us photochem abundances
                interp_results[key] = interp_function(user_gridvals)
                
        return interp_results

def find_ReflectedSpectra_sol(filepath='/Users/epawelka/Documents/NASA_Ames_ProjS25/AmesProjS25Work/results/ReflectedSpectra_fv.h5',total_flux = None, log10_planet_metallicity=None, tint=None, Kzz=None, phase=None, grid_gridvals=Reflected_Spectra.get_gridvals_RSM()):

    """
    Inputs:
    log10_totalflux = np.array([1.25])
    log10_planet_metallicity = np.array([0.75])
    tint = np.array([50]) # in Kelvin
    kzz = np.array([7]) # in log space
    phase = np.array([0]) # in radians
    """

    # This takes the inputs that define the grid
    gridshape = tuple(len(a) for a in grid_gridvals)

    print(f"Make sure your inputs are within the following ranges, total_flux: {np.min(grid_gridvals[0])} - {np.max(grid_gridvals[0])} xsolar, planet metallicity: {np.min((grid_gridvals[1]).astype(float))} - {np.max((grid_gridvals[1]).astype(float))} xsolar, tint: {np.min(grid_gridvals[2])} - {np.max(grid_gridvals[2])} K, kzz: {np.min(grid_gridvals[3])} - {np.max(grid_gridvals[3])} log10 of cm^2/s, phase: {np.min(grid_gridvals[4])} - {np.max(grid_gridvals[4])} radians.")

    # Check to see if there is a solution that already exists
    wno, albedo, fpfs = GraphsKey.find_Reflected_Spectra_values(filename=filepath, total_flux=total_flux, log10_planet_metallicity=log10_planet_metallicity, tint=tint, Kzz=Kzz, phase=phase)
    
    if wno is not None:
        print(f'All inputs chosen were directly on the grid!')
        comb_results = {}
        comb_results['wno'] = wno
        comb_results['albedo'] = albedo
        comb_results['fpfs'] = fpfs
        return comb_results

    else:

        print(f'Interpolating results...')
        
        # This notes the grid and associated inputs used to make it as the data
        PhotCh_grid = gridutils.GridInterpolator(filename=filepath, gridvals=grid_gridvals)

        # This interpolates the results based on the user input
        interp_results = {}
    
        # New grid values to interpolate
        user_gridvals = (total_flux, log10_planet_metallicity, tint, Kzz, phase)
    
        for key in PhotCh_grid.data.keys():
            if key.startswith('cloud'):
                continue
            else:
                interp_function = PhotCh_grid.make_interpolator(key=key, logspace=False)
                interp_results[key] = interp_function(user_gridvals)
                
        return interp_results

    



