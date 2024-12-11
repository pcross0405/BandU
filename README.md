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

> Currently only reading directly from ABINIT 7 wavefunctions is supported.
> Reading eigenvalues from other DFT packages will come in future updates.

   - [ABINIT](https://abinit.github.io/abinit_web/)

---------------------------------------------------------------------------------------------------------  
<h1><p align="center">REPORTING ISSUES</p></h1>

Please report any issues [here](https://github.com/pcross0405/BandU/issues)  

-------------------------------------------------------------------------------------------------------------------------  
<h1><p align="center">TUTORIAL</p></h1>

TBA