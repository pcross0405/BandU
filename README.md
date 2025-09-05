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

<h2><p align="center">THROUGH GITHUB</p></h2>

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

<h2><p align="center">THROUGH PIP</p></h2>

pip install bandu
   
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
<pre>
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
        principal_orbital_components:bool, 
        fermi_surface:bool, 
        fermi_surface_projection:bool,
        load_fermi_surface:bool
)->None:
    # this option will generate the principal orbital components 1 through 10
    # to generate more or less, adjust the range of the "nums" keyword in the ToXSF() function
    # the energy sampled can be set, relative to the Fermi energy, by changing the the "energy_level" global variable
    # states are included in the analysis if they are within +/- 1/2*width of the set energy_level 
    # to get fewer or more states, decrease or increase, respectively, the "width" global variable
    # by default, the prinicipal orbital components are generated from an irreducible wedge of the Brillouin Zone
    # to generate from the full BZ, change the "sym" attribute in the BandU class from "False" to "True"
    if principal_orbital_components:
        wfk_gen = AbinitWFK(wfk_path).ReadWFK(
            energy_level = energy_level,
            width=width
         )
        wfk = BandU(
            wfks = wfk_gen,
            energy_level = energy_level,
            width = width,
            sym = False
         )
        wfk.ToXSF(
            xsf_name = bandu_name,
            nums = [1,10]
         )
    # this option will only generate an energy isosurface and will not project principal component overlap onto the surface
    # the "energy_level" global variable is the energy at which the isosurface will be be generated, relative to the Fermi energy
    # so energy_level = 0.0 will generate the Fermi surface
    # the "width" global variable determines how many states are included in the generation of the isosurface
    # a small width (~10 meV or ~0.5 mHa) is best here as larger widths may introduce bands that do not cross the Fermi energy
    # the color of the surface can be changed to any string compatible with the matplotlib colors 
    # see named colors here: https://matplotlib.org/stable/gallery/color/named_colors.html
    # the Plot function has many other keywords to customize the visuals to the users liking, see the docstring for more
    elif fermi_surface: 
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
        plot.Plot(
            color = 'silver',
        ) # plot contours
    # this option will generate an energy isosurface as well as project the overlap of a principal orbital component onto the surface
    # everything remains the same as the previous option, except now the principal orbtial component XSF file is needed 
    # also the color of the surface is done with the Colors module by default
    # other colors can be made with the Colors module, also any matplotlib colormap works
    elif fermi_surface_projection: 
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
    # this option will load a previously generated and saved fermi surface file
    # update the "save_path" keyword to match the path and name of your save file
    elif load_fermi_surface:
        Plotter().Load(
            save_path='{root_name}_bandu_{xsf_number}_fermi_surf.pkl',
        )
# to run any of the options above, make sure to set that option to "True"
# also be sure that the other options (or at least all options that come before) are set to "False"
# the main function will only run which ever option is the first found to be "True" in top to bottom order
# in other words, the priority follows as most to least in the order:
# principal_orbital_components -> fermi_surface -> fermi_surface_projection -> load_fermi_surface
if __name__ == '__main__':
    main(
        principal_orbital_components=True, 
        fermi_surface=True, 
        fermi_surface_projection=True,
        load_fermi_surface=True
    )
<pre>