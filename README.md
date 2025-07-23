# BandU
------------------------------------------------------------------------------------------------------- 
<h1><p align="center">BandU OVERVIEW</p></h1>

<p align="justify">A package that performs a principal component inspired analysis on the Bloch wavefunctions of 
periodic material to provide a real space visualization of the states that significantly contribute to the Fermi surface.
</p>

<p align="justify">These real space functions can then be projected onto the Fermi surface to provide a clear visual
for where a nesting vector may combine two points in reciprocal space.</p>

<p align="justify">This package is designed to be very straightforward in its use, offering Fermi surface and BandU functions 
visualizations in as little as 5 lines of Python script. This package can also be used to just visual the Fermi surface, without 
BandU projections, if provided with the necessary k-point and eigenvalue data.</p>

-------------------------------------------------------------------------------------------------------  
<h1><p align="center">INSTALLATION INSTRUCTIONS</p></h1>

1) Inside that directory type on the command line  
   "git clone https://github.com/pcross0405/BandU.git"

2) Type "cd BandU"

3) Make sure you have python's build tool up to date with  
   "python3 -m pip install --upgrade build"

4) Once up to date type  
   "python3 -m build"

5) This should create a "dist" directory with a .whl file inside

6) On the command line type  
   "pip install dist/*.whl" 
   
-------------------------------------------------------------------------------------------------------  
<h1><p align="center">DEPENDENCIES</p></h1>

REQUIRED FOR VISUALIZING FERMI SURFACE

   - [pyvista](https://pyvista.org/)

   - [numpy](https://numpy.org/)

REQUIRED FOR CUSTOM COLORS

   - [matplotlib](https://matplotlib.org/)

WAVEFUNCTIONS THAT CAN BE READ DIRECTLY

> Currently only reading directly from ABINIT 7 and 10 wavefunctions is supported.
> Reading eigenvalues from other DFT packages will come in future updates.

   - [ABINIT](https://abinit.github.io/abinit_web/)

---------------------------------------------------------------------------------------------------------  
<h1><p align="center">REPORTING ISSUES</p></h1>

Please report any issues [here](https://github.com/pcross0405/BandU/issues)  

-------------------------------------------------------------------------------------------------------------------------  
<h1><p align="center">TUTORIAL</p></h1>

An example script that can run the different functions of the BandU program is given below.
-------------------------------------------------------------------------------------------
from bandu.bandu import BandU
from bandu.abinit_reader import AbinitWFK
from bandu.isosurface_class import Isosurface
from bandu.plotter import Plotter
from bandu.colors import Colors


root_name = 'your file root name here' # root_name of WFK files and of XSF files
xsf_number = 1 # XSF file number to be read in
energy_level = 0.000 # Energy relative to the Fermi energy to be sampled
width = 0.0005 # Search half the the width above and below the specified energy level
wfk_path = f'path\to\WFK\file\{root_name}_o_WFK'
xsf_path = f'path\to\XSF\file\{root_name}_bandu_{xsf_number}'
bandu_name = f'{root_name}_bandu'

def main(
        Band_U:bool, Iso_Surf:bool, Load_Surf:bool
)->None:
    if Band_U: # create BandU principal orbital components
        wfk_gen = AbinitWFK(wfk_path).ReadWFK(
            energy_level = energy_level, 
            width=width
         )
        wfk = BandU(
            wfks = wfk_gen, 
            energy_level = energy_level, 
            width = width, 
            sym = True
         )
        wfk.ToXSF(
            xsf_name = bandu_name, 
            nums = [1,10]
         )
    elif Iso_Surf: # construct energy isosurfaces and plot them
        contours = Isosurface(
            wfk_name = wfk_path, 
            energy_level = energy_level,
            width = width
        )
        contours.Contour() # make contours
        plot = Plotter(
            isosurface = contours, 
            save_file=f'{root_name}_bandu_{xsf_number}_fermi_surf.pkl'
         ) # create plotter object
        overlap_vals = plot.SurfaceColor(
            wfk_path=wfk_path,
            xsf_path=xsf_path,
        ) # compute overlap between principal orbital component and states in Brillouin Zone
        plot.Plot(
            surface_vals = overlap_vals,
            colormap = Colors().blues,
        ) # plot contours
    elif LoadSurf:
        Plotter().Load(
            save_path='{root_name}_bandu_{xsf_number}_fermi_surf.pkl',
        )
if __name__ == '__main__':
    main(
        Band_U = False,
        Iso_Surf = False,
        Load_Surf = False,
    ) 