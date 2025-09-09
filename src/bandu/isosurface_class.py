import pyvista as pv
import numpy as np
from . import translate as trnslt
from . import abinit_reader as ar
from . import wfk_class as wc
from . import brillouin_zone as brlzn

# object for creating reciprocal space energy isosurfaces
class Isosurface():
    '''
    Class for constructing contours from WFK eigenvalues

    Parameters
    ----------
    points : np.ndarray
        Array of reciprocal space points to construct contour over
        Shape (N,3)
    values : np.ndarray
        Array of eigenvalues for each reciprocal space point
        Shape (N,1) or (N,#Bands)
    nbands : list[int]
        List of bands to create contours for, aka indices of bands that cross energy isosurface
    rec_latt : np.ndarray
        Reciprocal lattice vectors
    wfk_name : str
        If points and values arrays are not provided, then construct contours from reading in an ABINIT WFK file
        Path to wfk file
    grid_steps : tuple
        Set fineness of grid that contours are interpolated on
        Default (50,50,50)
    fermi_energy : float
        Fermi energy
        Default 0.0
    energy_level : float
        Sets energy isosurface relative to the Fermi energy
        Default 0.0 (this samples the Fermi energy)
    width : float
        Sets the range above and below the energy_level that eigenvalues are sampled from
        Default 0.005 (Assumes units of Hartree) 
    radius : float
        Radius of gaussian interpolation
        Default 0.05, this scales inversely with kpoint grid size (larger grids require smaller radius)
    sym_ops : np.ndarray
        Array of symmetry operations for generating points in Brillouin Zone
    grid_stretch : float
        Value that stretches grid that is used to interpolate bands
        If portions of Brillouin Zone are cropped off, increase this value
        Default is 0.15

    Methods
    -------
    Contour
        Generates energy isosurface contours
    '''
    def __init__(
        self, points:np.ndarray=np.zeros(1), values:np.ndarray=np.zeros(1), rec_latt:np.ndarray=np.zeros(1), 
        wfk_name:str='', grid_steps:tuple=(50,50,50), fermi_energy:float=0.0, energy_level:float=0.0, 
        width:float=0.005, radius:float=0.05, nbands:list[int]=[], sym_ops:np.ndarray=np.zeros(1), 
        grid_stretch:float=0.15
    )->None:
        # define attributes
        self.points=points
        self.values=values
        self.wfk_name=wfk_name
        self.rec_latt=rec_latt
        self.fermi_energy=fermi_energy
        self.energy_level=energy_level
        self.width=width
        self.radius=radius
        self.nbands=nbands
        self.sym_ops=sym_ops
        self.grid_stretch=grid_stretch
        # check if enough information is provided to class to construct isosurfaces
        if self.points.shape == (1,) and self.values.shape == (1,):
            if wfk_name == '':
                raise ValueError((
                    'Either the points and the values attributes must be defined or the wfk_name attribute '
                    'must be defined'
                ))
            else:
                self.points, self.values, self.nbands, self.ir_kpts = self._ReadAbinit(wfk_name)
        # check if information provided is in correct format
        if len(self.values.shape) > 2:
            raise ValueError((
                f'Provided values have dimension {len(self.values.shape)}'
                'Values must be 1 or 2 dimensional'
            ))
        else:
            self._valdim=len(self.values.shape)
        # more attributes, these rely on points and values attributes
        self.contours:list[pv.PolyData] = []
        self.grid_steps=grid_steps
        self.grid = self._MakeGrid()
#---------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ METHODS ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#
    # method for creating grid to interplote on
    def _MakeGrid(
            self
    )->pv.ImageData:
        '''
        Method for constructing PyVista grid used in interpolation prior to plotting
        *This method assumes your points are in Cartesian format*

        Parameters
        ----------
        points : np.ndarry
            Numpy array with shape (N, 3) where N is number of points\n
            These are the points that lie within your energy range, generally from GetValAndKpt method\n
            These are NOT the points of the grid
        steps : tuple
            Define the number of grid points along (x,y,z)\n
            Default is 50 points along each axis\n
            Spacing of grid is automatically calculated from steps and points
        '''
        # helper functions for finding grid origin and setting grid spacing
        def _GetMax(pts:np.ndarray):
            xmax = pts[:,0].max()   
            ymax = pts[:,1].max()
            zmax = pts[:,2].max()
            return xmax, ymax, zmax
        def _GetMin(pts:np.ndarray):
            xmin = pts[:,0].min()   
            ymin = pts[:,1].min()
            zmin = pts[:,2].min()
            return xmin, ymin, zmin
        # begin method for making grid
        xmax, ymax, zmax = _GetMax(self.points)
        xmin, ymin, zmin = _GetMin(self.points)
        dimx = self.grid_steps[0]
        dimy = self.grid_steps[1]
        dimz = self.grid_steps[2]
        grid = pv.ImageData()
        stretch_factor = self.grid_stretch
        dim_stretch = [2*stretch_factor/dim for dim in self.grid_steps]
        grid.origin = (
            xmin - 0.5*stretch_factor, 
            ymin - 0.5*stretch_factor,
            zmin - 0.5*stretch_factor
        )
        grid.spacing = (
            2*xmax/dimx + dim_stretch[0], 
            2*ymax/dimy + dim_stretch[1], 
            2*zmax/dimz + dim_stretch[2]
        )
        grid.dimensions = (dimx, dimy, dimz)
        return grid
    #-----------------------------------------------------------------------------------------------------------------#
    # method for selecting which contour to perform depending on if a single band or many bands are passed
    def Contour(
        self
    ):
        if self._valdim == 1:
            self._Contour(points=self.points, values=self.values)
        else:
            for band in self.values.T:
                self._Contour(points=self.points, values=band)
    #-----------------------------------------------------------------------------------------------------------------#
    # method for drawing contour from list of values
    def _Contour(
        self, points:np.ndarray, values:np.ndarray
    ):
        # setup grid interpolation
        null_value = self.fermi_energy + self.energy_level + self.width/2 * 1.05
        trans_pts, trans_vals = trnslt.TranslatePoints(
            points=points, 
            values=values.reshape((values.shape[0],1)), 
            lattice_vecs=self.rec_latt
        )
        trans_pts = pv.PolyData(trans_pts)
        trans_pts['values'] = trans_vals
        inter_grid:pv.ImageData = self.grid.interpolate(
            trans_pts, 
            radius=self.radius, 
            sharpness=2.0, 
            strategy='null_value', 
            null_value=null_value
        )
        # create contour
        iso_range = [self.fermi_energy + self.energy_level, self.fermi_energy + self.energy_level]
        contour:pv.PolyData = inter_grid.contour(
            isosurfaces=2,
            rng=iso_range,
            method='contour'
        )
        self.contours.append(contour)
    #-----------------------------------------------------------------------------------------------------------------#
    # method for reading kpoints and eigenvalues from ABINIT WFK file
    def _ReadAbinit(
        self, filename:str
    )->tuple[np.ndarray,np.ndarray, list, np.ndarray]:
        # collect kpoints and eigenvalues from wfk file
        wfk = ar.AbinitWFK(filename=filename)
        eigs = []
        syms = np.array(wfk.symrel)
        nsym = wfk.nsym
        kpoints = np.array(wfk.kpts)
        ir_kpts = kpoints
        nbands = wfk.bands[0]
        nkpt = wfk.nkpt
        self.fermi_energy = wfk.fermi
        self.rec_latt = wc.WFK(lattice=wfk.real_lattice).Real2Reciprocal()
        for wfk_obj in wfk.ReadEigenvalues():
            eigs.append(wfk_obj.eigenvalues)
        eigs = np.array(eigs).reshape((nkpt,nbands))
        # look through eigenvalues to find which bands to contour
        min_val = self.fermi_energy + self.energy_level - self.width/2
        max_val = self.fermi_energy + self.energy_level + self.width/2
        band_num = []
        for i in range(nbands):
            for eigval in eigs[:,i]:
                if min_val <= eigval <= max_val:
                    band_num.append(i)
                    break            
        # return kpoints and bands of interest
        bands = np.take(eigs, band_num, axis=1)
        shifts = brlzn.BZ(self.rec_latt).GetShifts(kpoints)
        kpoints = kpoints - shifts
        ir_kpts -= shifts
        ir_kpts = np.matmul(ir_kpts,self.rec_latt)
        wfk = wc.WFK(symrel=syms, nsym=nsym, nbands=len(band_num), nkpt=nkpt)
        all_kpts = np.zeros((1,3))
        all_eigs = np.zeros((1,bands.shape[1]))
        # symmetrize kpoints
        for i, kpt in enumerate(kpoints):
            unique_kpts, _ = wfk.Symmetrize(
                points=kpt,
                reciprocal=True,
            )
            all_kpts = np.concatenate((all_kpts, unique_kpts), axis=0)
            new_eigs = np.repeat(bands[i,:].reshape((1,-1)), unique_kpts.shape[0], axis=0)
            all_eigs = np.concatenate((all_eigs, new_eigs), axis=0)
        kpoints = np.delete(all_kpts, 0, axis=0)
        bands = np.delete(all_eigs, 0, axis=0)
        kpoints = np.matmul(kpoints, self.rec_latt)
        return kpoints, bands, band_num, ir_kpts