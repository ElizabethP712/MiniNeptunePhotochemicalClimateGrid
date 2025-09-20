# Differentiating Mini-Neptunes from Terrestrial Atmospheres 
## (NASA Ames Summer 2025 OSTEM/HWO Intern Project)

Habitable Worlds Observatory, a NASA future flagship mission, aims to detect new exoplanets with an emphasis on Earth-like planets 
around Sun-like stars in the search for habitable conditions. However, upon initial detection, the mass and radius of these planets
is poorly constrained, meaning we could be viewing a massive gas giant with low albedo, or a less massive terrestrial planet with high
albedo. The purpose of this project was to investigate whether we could distinguish mini Neptunes from terrestrial worlds like Earth. 

This was done by developing three grids, located in the Scripts folder, to represent a parameter space of mini Neptunes. 
The PICASO_Climate_grid.py calculates a temperature and pressure profile based on energy conservation and chemical equilibrium assumptions, 
based on climate modeling software developed by Dr. Natasha Batalha.The Photochem_grid.py then takes the pressure and temperature profile calculated by
PICASO and calculates chemical abundances at a steady state in the atmosphere accounting for vertical mixing, temperature-pressure gradients, and UV photolysis that could
change chemical abundances, assuming mass conservation, which pulls from the 1D photochemical modeling software, Photochem, an open source code developed by
Dr. Nick Wogan. Finally, the Reflected_Spectra_grid.py takes the photochemical and PT models and calculates the reflected light spectra of these mini Neptunes.

The Demo&Test_Notebooks file includes a demo application and analysis using the code, whose results are summarized in the Final Presentation on the first page. By default,
the grids are set to solve for a K218-b like mini-Neptune around the Sun (G-star), ranging over the following input parameters:

<br> **Total Flux:**
<br> 0.1 - 2.0 x Solar

<br> **Planet Metallicity:**
<br> 3 - 100 x Solar (input will need to be in logspace, so 3x Solar Mettalicity = 10^0.5, just input 0.5)

<br> **Internal Temperature (Tint):**
<br> 20 - 200 K

<br> **Eddy Diffusion Coefficient (Kzz):**
<br> 10^5  −10^9  cm^2/𝑠 (input will just be 5, or 9, since this is in logspace)

<br> **Phase:** 
<br> ~ 0 - 170° (input will need to be in radians)

<br>Each of these grids have been configured to parallize computations, with the default grid results following the ranges above available
in the script/results folder, or on zenodo, but if you wish to change the input ranges to explore a new parameter space of mini-Neptunes, 
the original scripts will have to be adjusted in the get_gridvals functions of each script. 

The rest of the notebooks include demonstrations on how to apply PICASO and Photochem software to one K2-18b like case around a G-star and M-dwarf, 
how to read in results from grids in order to compare mini-Neptune and Earth spectras, as well as find optimized fits between the two. 
There are also notebooks demonstrating how to check implementation of clouds and convergence for PICASO
and Photochem grids. 

It is important to note you will have to restructure your home directory to include the notebooks, scripts in the scripts folder, and results (located in the scripts folder) 
to be under the same directory for everything to properly run. There is also an Installation Guide to provide directions in what versions of PICASO and Photochem were used in the project. 

Without further ado, enjoy exploring!

Quick Code References:
<br> PICASO - https://github.com/natashabatalha/picaso.git
<br> Photochem - https://github.com/Nicholaswogan/photochem.git
<br> Zenodo Grid Data - https://zenodo.org/records/17161837?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI1MTM5NTIwLTFiMmYtNDgxZi05NTFiLWU4Mjk3M2MxNDA3MSIsImRhdGEiOnt9LCJyYW5kb20iOiJlYjMzZTE1N2MxNzM0MjYzOTQ3YjhhMTJmYjEwYTNmMCJ9.kXJMYqbJ42Zf1ZMTfw3Sj1_fZABxIVcDMTMq34mxIpJ7WMm8jLXJZg66zu6GKyCL3UylKScAx0CG_etGlUNH9g

*Please note all code is property of NASA Ames. 
