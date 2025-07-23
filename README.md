# BandU
------------------------------------------------------------------------------------------------------- 
<h1><p align="center">BandU OVERVIEW</p></h1>

<p align="justify">A package that performs a principal component inspired analysis on the Bloch wavefunctions of 
periodic material to provide a real space visualization of the states that significantly contribute to the Fermi surface.
</p>

<p align="justify">These real space functions can then be projected onto the Fermi surface to provide a clear visual
for where a nesting vector may combine two points in reciprocal space.</p>

<p align="justify">This package is designed to be very straightforward in its use, offering Fermi surface and BandU function 
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
from bandu.bandu import BandU<br />
from bandu.abinit_reader import AbinitWFK<br />
from bandu.isosurface_class import Isosurface<br />
from bandu.plotter import Plotter<br />
from bandu.colors import Colors<br />

root_name = 'your file root name here' # root_name of WFK files and of XSF files<br />
xsf_number = 1 # XSF file number to be read in<br />
energy_level = 0.000 # Energy relative to the Fermi energy to be sampled<br />
width = 0.0005 # Search half the the width above and below the specified energy level<br />
wfk_path = f'path\to\WFK\file\{root_name}_o_WFK'<br />
xsf_path = f'path\to\XSF\file\{root_name}\_bandu_{xsf_number}'<br />
bandu_name = f'{root_name}_bandu'<br />

<pre>
def main(<br />
        Band_U:bool, Iso_Surf:bool, Load_Surf:bool<br />
)->None:<br />
    if Band_U: # create BandU principal orbital components<br />
        wfk_gen = AbinitWFK(wfk_path).ReadWFK(<br />
            energy_level = energy_level,<br />
            width=width<br />
         )<br />
        wfk = BandU(<br />
            wfks = wfk_gen,<br />
            energy_level = energy_level,<br />
            width = width,<br />
            sym = True<br />
         )<br />
        wfk.ToXSF(<br />
            xsf_name = bandu_name,<br />
            nums = [1,10]<br />
         )<br />
    elif Iso_Surf: # construct energy isosurfaces and plot them<br />
        contours = Isosurface(<br />
            wfk_name = wfk_path,<br />
            energy_level = energy_level,<br />
            width = width<br />
        )<br />
        contours.Contour() # make contours<br />
        plot = Plotter(<br />
            isosurface = contours,<br />
            save_file=f'{root_name}_bandu_{xsf_number}_fermi_surf.pkl'<br />
         ) # create plotter object<br />
        overlap_vals = plot.SurfaceColor(<br />
            wfk_path=wfk_path,<br />
            xsf_path=xsf_path,<br />
        ) # compute overlap between principal orbital component and states in Brillouin Zone<br />
        plot.Plot(<br />
            surface_vals = overlap_vals,<br />
            colormap = Colors().blues,<br />
        ) # plot contours<br />
    elif LoadSurf:<br />
        Plotter().Load(<br />
            save_path='{root_name}_bandu_{xsf_number}_fermi_surf.pkl',<br />
        )<br />
if \__name__ == '\__main__':<br />
    main(<br />
        Band_U = False,<br />
        Iso_Surf = False,<br />
        Load_Surf = False,<br />
    )<br />
<pre>