import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import Photochem_grid
import Reflected_Spectra_grid as Reflected_Spectra
import PICASO_Climate_grid
import FilterGrids
from photochem.utils import stars
import pickle

# Open an h5 file
def find_Reflected_Spectra_values(filename=None, total_flux=None, log10_planet_metallicity=None, tint=None, Kzz=None, phase=None, gridvals=Reflected_Spectra.get_gridvals_RSM()):
    """
    filename = this is the output of makegrid for climate model using Photochem
    flux, metallicity, tint, Kzz are the same inputs used to create the photochemical grid
    but the gridvals are what was used for the Photochem modeling
    Also note that I might want to add a if then statement for convergence (some PT profiles didn't converge with PICASO)
    """
    gridvals_metal = [float(s) for s in gridvals[1]]
    log10_planet_metallicity = float(log10_planet_metallicity)
    gridvals_dict = {'total_flux':gridvals[0], 'planet_metallicity':gridvals_metal, 'tint':gridvals[2], 'Kzz':gridvals[3], 'phase':gridvals[4]}

    with h5py.File(filename, 'r') as f:
        input_list = np.array([total_flux, log10_planet_metallicity, tint, Kzz, phase])
        #print(f'These were the inputs used: {input_list}')
        #print(f'These were the inputs used for metallicity from the grid: {gridvals_dict['planet_metallicity']}')
        matches = (list(f['inputs'] == input_list))
        row_matches = np.all(matches, axis=1)
        matching_indicies = np.where(row_matches)

        matching_indicies_flux = np.where(list(gridvals_dict['total_flux'] == input_list[0]))
        matching_indicies_metal = np.where(list(gridvals_dict['planet_metallicity'] == input_list[1]))
        #print(f'This is the matching indicies of metallicity: {matching_indicies_metal}')
        matching_indicies_tint = np.where(list(gridvals_dict['tint'] == input_list[2]))
        matching_indicies_kzz = np.where(list(gridvals_dict['Kzz'] == input_list[3]))
        matching_indicies_phase = np.where(list(gridvals_dict['phase'] == input_list[4]))
        

        flux_index, metal_index, tint_index, kzz_index, phase_index = matching_indicies_flux[0], matching_indicies_metal[0], matching_indicies_tint[0], matching_indicies_kzz[0], matching_indicies_phase[0]

        if matching_indicies[0].size == 0:
            print(f'A match given total flux, planet metallicity, and tint does not exist')
            wno = None
            albedo = None
            fpfs = None
            return wno, albedo, fpfs
            
        else:
            wno = np.array(f['results']['wno'][flux_index[0]][metal_index[0]][tint_index[0]][kzz_index[0]][phase_index[0]])
            albedo = (f['results']['albedo'][flux_index[0]][metal_index[0]][tint_index[0]][kzz_index[0]][phase_index[0]])
            fpfs =  (f['results']['fpfs'][flux_index[0]][metal_index[0]][tint_index[0]][kzz_index[0]][phase_index[0]])
            print(f'Was able to successfully find your input parameters in the PICASO TP profile grid!')
            
            return wno, albedo, fpfs

def find_all_plotting_values_old(total_flux=None, planet_metal=None, tint=None, kzz=None, phase=None, ptop=None, pbottom=None, w0=None, g0=None, opd=None, reflected_spec=False):

    
    PT_list, convergence_values = Photochem_grid.find_PT_grid(total_flux=total_flux, log10_planet_metallicity=planet_metal, tint=tint, gridvals=PICASO_Climate_grid.get_gridvals_PICASO_TP())
    
    sol_dict, soleq_dict, PT_list_Photochem, convergence_PC, convergence_TP = Reflected_Spectra.find_Photochem_match(total_flux=total_flux, log10_planet_metallicity=planet_metal, tint=tint, Kzz=kzz, gridvals= Photochem_grid.get_gridvals_Photochem())
    
    if reflected_spec == False:
        wno = 0
        albedo = 0
        fpfs = 0
        return PT_list, sol_dict, soleq_dict, wno, albedo, fpfs, PT_list_Photochem
    
    elif reflected_spec == True:
        wno, albedo, fpfs = find_Reflected_Spectra_values(total_flux=total_flux, log10_planet_metallicity=planet_metal, tint=tint, Kzz=kzz, phase=phase, gridvals=Reflected_Spectra.get_gridvals_RSM())
        
        return PT_list, sol_dict, soleq_dict, wno, albedo, fpfs, PT_list_Photochem

def find_all_plotting_values(total_flux=None, planet_metal=None, tint=None, kzz=None, phase=None, calc_PT=True, calc_PhotCh=True, calc_RSM=True):

    # Results from PICASO Grid (Interpolates if within range of grid)
    if calc_PT == True:
        PT_results_dict = FilterGrids.find_PT_sol(total_flux=total_flux, log10_planet_metallicity=planet_metal, tint=tint)

        PT_list = [PT_results_dict['pressure'], PT_results_dict['temperature']]

    elif calc_PT == False:
        PT_list = None

    # Results from Photochem Grid (Interpolates if within range of grid)
    if calc_PhotCh == True:
        PhotCh_results_dict = FilterGrids.find_Photochem_sol(total_flux=total_flux, log10_planet_metallicity=planet_metal, tint=tint, Kzz=kzz)
    
        sol_dict = {}
        soleq_dict = {}
        for key in PhotCh_results_dict.keys():
            if key.endswith('_sol'):
                sol_dict[key] = PhotCh_results_dict[key].copy()
            elif key.endswith('_soleq'):
                soleq_dict[key] = PhotCh_results_dict[key].copy()
            else:
                continue
        
        PT_list_Photochem = [PhotCh_results_dict['pressure_sol'], PhotCh_results_dict['temperature_sol']]

    elif calc_PhotCh == False:
        PT_list_Photochem = None
        sol_dict = None
        soleq_dict = None

    if calc_RSM == True:
        # Results from Reflected Spectra Grid
        RSM_results_dict = FilterGrids.find_ReflectedSpectra_sol(total_flux=total_flux, log10_planet_metallicity=planet_metal, tint=tint, Kzz=kzz, phase=phase)
    
        wno = RSM_results_dict['wno']
        albedo = RSM_results_dict['albedo']
        fpfs = RSM_results_dict['fpfs']

    elif calc_RSM == False:
        wno = None
        albedo = None
        fpfs = None
    
    return PT_list, sol_dict, soleq_dict, wno, albedo, fpfs, PT_list_Photochem
    

def plot_PT(total_flux=None, planet_metal=None, tint=None, kzz=None, phase=None, calc_PT=True, calc_PhotCh=True, calc_RSM=False):
    
    PT_list, sol_dict, soleq_dict, wno, albedo, fpfs, PT_list_Photochem = find_all_plotting_values(total_flux=total_flux, planet_metal=planet_metal, tint=tint, kzz=kzz, phase=phase, calc_PT=calc_PT, calc_PhotCh=calc_PhotCh, calc_RSM=calc_RSM)
    
    # Plot the PT Profile from PICASO
    
    plt.figure(figsize=(10,10))
    plt.ylabel("Pressure [Bars]", fontsize=25)
    plt.xlabel('Temperature [K]', fontsize=25)
    plt.gca().invert_yaxis()
    #plt.ylim(500,1e-4)
    #plt.xlim(250,1750)
    
    plt.semilogy(PT_list[1],PT_list[0],color="r",linewidth=3, label='PICASO PT')
    plt.semilogy(PT_list_Photochem[1],PT_list_Photochem[0]/(10**6),color="blue",linewidth=3, label='PICASO PT')
    plt.minorticks_on()
    plt.tick_params(axis='both',which='major',length =30, width=2,direction='in',labelsize=23)
    plt.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)

    Tef = stars.equilibrium_temperature(total_flux*1361, 0)
    plt.title(f"Teff = {Tef} K, log(g)=4.4")

    plt.show()

def plot_PT_Photochem(total_flux=None, planet_metal=None, tint=None, kzz=None, phase=None, calc_PT=True, calc_PhotCh=True, calc_RSM=False):
    
    PT_list, sol_dict, soleq_dict, wno, albedo, fpfs, PT_list_Photochem = find_all_plotting_values(total_flux=total_flux, planet_metal=planet_metal, tint=tint, kzz=kzz, phase=phase, calc_PT=calc_PT, calc_PhotCh=calc_PhotCh, calc_RSM=calc_RSM)
    
    # Plot the Composition from Photochem
    fig, ax1 = plt.subplots(1,1,figsize=[5,4])
    #species = ['CO2','H2O','CH4','CO','NH3','H2','HCN','H2Oaer']
    species_sol = ['CO2_sol','H2O_sol','CH4_sol','CO_sol','NH3_sol','H2_sol','HCN_sol','H2Oaer_sol']
    species_soleq = ['CO2_soleq','H2O_soleq','CH4_soleq','CO_soleq','NH3_soleq','H2_soleq','HCN_soleq','H2Oaer_soleq']
    
    #for i,sp in enumerate(species):
    #    ax1.plot(sol_dict[sp],sol_dict['pressure']/1e6,label=sp, c='C'+str(i))
    #    ax1.plot(soleq_dict[sp],soleq_dict['pressure']/1e6, ls=':', c='C'+str(i), alpha=0.4)
    
    for i,sp in enumerate(species_soleq):
        ax1.plot(soleq_dict[sp],soleq_dict['pressure_soleq']/1e6, ls=':', c='C'+str(i), alpha=0.4)
    for i,sp in enumerate(species_sol):
        ax1.plot(sol_dict[sp],sol_dict['pressure_sol']/1e6, c='C'+str(i), label=sp)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(1e-8,1)
    ax1.set_ylim(1000,1e-7)
    ax1.grid(alpha=0.4)
    ax1.legend(ncol=1,bbox_to_anchor=(1,1.0),loc='upper left')
    ax1.set_xlabel('Mixing Ratio')
    ax1.set_ylabel('Pressure (bar)')
    ax1.set_yticks(10.0**np.arange(-6,2))
    #ax1.text(0.02, 1.04, 't = '+'%.2e'%pc.wrk.tn, \
    #    size = 15,ha='left', va='bottom',transform=ax1.transAxes)
    
    ax2 = ax1.twiny()
    ax2.set_xlabel('Temperature (K)')
    #print(((PT_list_Photochem[1])/(10**6)),  PT_list_Photochem[0])
    #print(np.flip(PT_list[1]), np.flip(PT_list[0]))
    ax2.plot(PT_list_Photochem[1], (PT_list_Photochem[0]/(10**6)), c='blue', ls='--',label='PT PICASO Profile')
    ax2.plot(np.flip(PT_list[1]), np.flip(PT_list[0]), c='black', ls='--',label='PT PICASO Profile')
    
    plt.title('K2-18b Around Sun (G-Star)')
    
    plt.legend()

    plt.show()
    
    # Save the plot as a PNG image
    # plt.savefig('output_graph_GStar_K218b.png', bbox_inches='tight')
    
    # Close the plot to free up memory (important if you're generating many plots)
    # plt.close()

def plot_solar_spectra(solar_file_name='GJ176_spectrum_278K.txt'):
    
    # Determine solar data:
    data_star = pd.read_csv(solar_file_name, sep='/t', engine='python')
    data_star[['Wavelength (nm)', 'Solar flux (mW/m^2/nm)']] = data_star['Wavelength(nm)      SolarFlux(mW/m^2/nm)'].str.split(expand=True)
    df_star = data_star.drop('Wavelength(nm)      SolarFlux(mW/m^2/nm)', axis=1)
    df_star['Wavelength (nm)'] = df_star['Wavelength (nm)'].astype(float)
    df_star['Solar flux (mW/m^2/nm)'] = df_star['Solar flux (mW/m^2/nm)'].astype(float)

    fig, ax1 = plt.subplots(1,1,figsize=[5,4])

    ax1.plot(df_star['Wavelength (nm)'], df_star['Solar flux (mW/m^2/nm)'], color='black', ls='--', linewidth=0.25)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Solar flux (mW/m^2/nm)')
    ax1.set_xlim(250,1400)
    
    # Plot vertical lines for EM spectral boundaries (UV - IR)
    
    # Fill the area between two vertical lines
    plt.axvspan(250, 400, color='lightblue', alpha=0.15, label='UV')
    
    # Visible
    plt.axvspan(380, 700, color='grey', alpha=0.15, label='Visible')
    
    # IR
    plt.axvspan(700, 1400, color='red', alpha=0.15, label='IR')
    
    plt.legend()
    plt.title('Star Spectra')
    plt.show()

def plot_Reflected_Spectra(total_flux=None, planet_metal=None, tint=None, kzz=None, phase=None, calc_PT=False, calc_PhotCh=False, calc_RSM=True):

    PT_list, sol_dict, soleq_dict, wno, albedo, fpfs, PT_list_Photochem = find_all_plotting_values(total_flux=total_flux, planet_metal=planet_metal, tint=tint, kzz=kzz, phase=phase, calc_PT=calc_PT, calc_PhotCh=calc_PhotCh, calc_RSM=calc_RSM)

    fig,ax = plt.subplots(1,1,figsize=[5,4])

    ax.plot(1e4/wno, fpfs, c='k', lw=1.5, label='With Cloud')
    #for key in res:
    #    if key == 'all':
    #        continue
    #    _, fpfs1 = res[key]
    #    ax.fill_between(1e4/wno,fpfs,fpfs1,label=key,alpha=0.3)
    
    #wno_no_cloud_grid_150, fpfs_no_cloud_grid_150 = jdi.mean_regrid(wno_no_cloud, fpfs_no_cloud, R=150)
    ax.plot(1e4/wno, fpfs, c='red', lw=.05, ls='--')
    #ax.set_xlim(0.3,1)
    #ax.set_ylim(0e-10,10e-9)
    ax.set_ylabel('Planet-to-star flux ratio')
    ax.set_xlabel('Wavelength (microns)')
    ax.set_title('K2-18b Reflected Spectra around G-Star')
    ax.legend()
    #plt.savefig('old_opacities.pdf',bbox_inches='tight')
    plt.show()

def plot_Reflected_Spectra_comp(total_flux=None, planet_metal=None, tint=None, kzz=None, phase=None, calc_PT=False, calc_PhotCh=False, calc_RSM=True):

    PT_list, sol_dict, soleq_dict, wno, albedo, fpfs, PT_list_Photochem = find_all_plotting_values(total_flux=total_flux, planet_metal=planet_metal, tint=tint, kzz=kzz, phase=phase, calc_PT=calc_PT, calc_PhotCh=calc_PhotCh, calc_RSM=calc_RSM)

    res1 = Reflected_Spectra.make_case_earth()
    wno_earth, fpfs_earth, albedo_earth = res1['all']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(1e4/wno, fpfs, c='k', lw=1.5, label='K218b/G-Star')
    ax1.plot(1e4/wno_earth, fpfs_earth, c='red', lw=1.5, label='Earth/G-Star')
    ax1.set_xlim(0.3,1)
    ax1.set_ylim(0e-10,10e-10)
    ax1.set_ylabel('Planet-to-star flux ratio')
    ax1.set_xlabel('Wavelength (microns)')
    ax1.set_title('Reflected Spectra w/ Earth Clouds')
    ax1.legend()

    ax2.plot(1e4/wno, fpfs, c='k', lw=1.5, label='K218b/G-Star')
    ax2.plot(1e4/wno_earth, fpfs_earth, c='red', lw=1.5, label='Earth/G-Star')
    ax2.set_xlim(0.3,1)
    ax2.set_ylim(0e-10,10e-9)
    ax2.set_ylabel('Planet-to-star flux ratio')
    ax2.set_xlabel('Wavelength (microns)')
    ax2.set_title('Reflected Spectra w/ Earth Clouds')
    ax2.legend()
    
    #plt.savefig('Earth_K218b_clouds_RS.pdf',bbox_inches='tight')
    
    #plt.savefig('Earth_K218b_clouds_RS_zoomedin.pdf',bbox_inches='tight')
    plt.show()
