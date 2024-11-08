from wfk_class import WFK, bytes2float, bytes2int
import numpy as np
from scipy.fft import fftn
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from typing import Generator

class Isosurface(WFK):
    def __init__(self, filename):
        super().__init__(filename)
        self.p = BackgroundPlotter(window_size=(600,400))
    def _ReadEigenvalues(self):
        wfk = open(self.filename, 'rb')
        #-----------#
        # skip header
        wfk.read(298)
        wfk.read(4*(2*self.nkpt + self.nkpt*self.nsppol + self.npsp + 10*self.nsym + self.natom))
        wfk.read(8*(4*self.nkpt + self.bandtot + 3*self.nsym + self.ntypat + 3*self.natom))
        wfk.read(self.npsp*(176))
        #-------------------------------#
        # begin reading wavefunction body
        for i in range(self.nsppol):
            for j in range(self.nkpt):
                print(f'Reading kpoint {j+1} of {self.nkpt}', end='\r')
                if j+1 == self.nkpt:
                    print('\n', end='')
                eigenvalues = []
                wfk.read(4)
                npw = bytes2int(wfk.read(4))
                self.nspinor = bytes2int(wfk.read(4))
                nband_temp = bytes2int(wfk.read(4))
                wfk.read(8)
                wfk.read(12*npw)
                wfk.read(8)
                #------------------------------------------------------------------#
                # only need eigenvalues for Fermi surface, skip over everything else
                for nband in range(nband_temp):
                    eigenval = bytes2float(wfk.read(8))
                    eigenvalues.append(eigenval)
                
                wfk.read(nband_temp*8)
                wfk.read(4)
                wfk.read(nband_temp*(8 + npw*16))
                yield eigenvalues
    #-----------------------------------------------------------------------------------------------------------------#
    # method for getting eigenvalues and kpoints within specified energy range
    def _GetValAndKpt(self, energy_level:float=None, width:float=0.0005)->tuple[np.ndarray, np.ndarray, list]:
        # default energy is fermi energy
        if energy_level == None:
            energy_level = self.fermi
        # define upper and lower energy bounds
        min_nrg = energy_level - width/2
        max_nrg = energy_level + width/2
        # lists energy values to for making isosurface
        iso_values = np.zeros((self.nkpt,self.bands[0]))
        band_num = []
        for i, eigenvalues in enumerate(self._ReadEigenvalues()):
            iso_values[i,:] = eigenvalues
            for band, eigval in enumerate(eigenvalues):
                if min_nrg <= eigval <= max_nrg:
                    band_num.append(band)
                    # only 1 value per kpoint (if there is a value) move to next kpt when found
                    break
        iso_pts = np.array(self.kpts).reshape((self.nkpt,3))
        band_num = set(band_num)
        return iso_pts, iso_values, band_num
    #-----------------------------------------------------------------------------------------------------------------#
    # method for getting real space wavefunctions
    def _FindOverlap(self, coeffs:np.ndarray, wfk_grid:np.ndarray, BandU:np.ndarray)->np.ndarray:
        ngfft_grid = np.zeros((self.ngfftx, self.ngffty, self.ngfftz), dtype=complex)
        overlaps = []
        for wfk in coeffs:
            for i, points in enumerate(wfk_grid):
                x = points[0]
                y = points[1]
                z = points[2]
                ngfft_grid[z][x][y] = wfk[i]
            ngfft_grid = fftn(ngfft_grid, norm='ortho')
            overlap = np.sum(BandU*ngfft_grid)
            overlaps.append(np.abs(overlap))
        return np.array(overlaps)
    #-----------------------------------------------------------------------------------------------------------------#
    # method getting eigenvalues and kpts within defined energy range, also get overlap of BandU eigfuncs w/ states
    def _GetOverlapValandKpt(self, BandU:np.ndarray, energy_level:float=None, width:float=0.0005
        )->tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        # default energy is fermi energy
        if energy_level == None:
            energy_level = self.fermi
        # define upper and lower energy bounds
        min_nrg = energy_level - width/2
        max_nrg = energy_level + width/2
        # lists energy values to for making isosurface
        iso_values = np.zeros((self.nkpt,self.bands[0]))
        overlaps = np.zeros((self.nkpt,self.bands[0]))
        band_num = []
        for i, (eigenvalues, coeffs, kpoints) in enumerate(self._ReadWFK()):
            overlap_values = self._FindOverlap(coeffs, kpoints, BandU)
            overlaps[i,:] = overlap_values
            iso_values[i,:] = eigenvalues
            for band, eigval in enumerate(eigenvalues):
                if min_nrg <= eigval <= max_nrg:
                    band_num.append(band)
                    # only 1 value per kpoint (if there is a value) move to next kpt when found
                    break
        iso_pts = np.array(self.kpts).reshape((self.nkpt,3))
        band_num = set(band_num)
        return iso_pts, iso_values, band_num, overlaps
    #-----------------------------------------------------------------------------------------------------------------#
    # method constructing PyVista grid
    def MakeGrid(self, points:np.ndarray, steps:tuple=(50,50,50))->pv.ImageData:
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
        # define functions to be used by MakeGrid only
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
        # begin script for making grid
        xmax, ymax, zmax = _GetMax(points)
        xmin, ymin, zmin = _GetMin(points)
        dimx = steps[0]
        dimy = steps[1]
        dimz = steps[2]
        grid = pv.ImageData()
        grid.origin = (xmin, ymin, zmin)
        grid.spacing = (2*xmax/(dimx-1), 2*ymax/(dimy-1), 2*zmax/(dimz-1))
        grid.dimensions = (dimx, dimy, dimz)
        return grid
    #-----------------------------------------------------------------------------------------------------------------#
    # method translating points to fill grid space
    def _TranslatePoints(self, points:np.ndarray, values:np.ndarray, lattice_vecs:np.ndarray
        )->tuple[np.ndarray, np.ndarray]:
        npts = len(points)
        # translate points along each lattice direction to ensure grid space is filled
        trans_points = np.zeros((27*npts,3))
        trans_values = np.zeros((27*npts,1))
        for i in range(3):
            even_ind = 9*i
            odd_ind = 9*i+1 
            # translate points along z axis
            trans_points[even_ind*npts:(even_ind+1)*npts,:] = layer_pts = points + (i-1)*lattice_vecs[2]
            # translate along +x and -x
            trans_points[odd_ind*npts:(odd_ind+1)*npts,:] = layer_pts + lattice_vecs[0]
            trans_points[(even_ind+2)*npts:(even_ind+3)*npts,:] = layer_pts - lattice_vecs[0]
            # along +y and -y
            trans_points[(odd_ind+2)*npts:(odd_ind+3)*npts,:] = layer_pts + lattice_vecs[1]
            trans_points[(even_ind+4)*npts:(even_ind+5)*npts,:] = layer_pts - lattice_vecs[1]
            # along +x+y and +x-y
            trans_points[(odd_ind+4)*npts:(odd_ind+5)*npts,:] = layer_pts + lattice_vecs[0] + lattice_vecs[1]
            trans_points[(even_ind+6)*npts:(even_ind+7)*npts,:] = layer_pts - lattice_vecs[0] + lattice_vecs[1]
            # along -x+y and -x-y
            trans_points[(odd_ind+6)*npts:(odd_ind+7)*npts,:] = layer_pts + lattice_vecs[0] - lattice_vecs[1]
            trans_points[(even_ind+8)*npts:(even_ind+9)*npts,:] = layer_pts - lattice_vecs[0] - lattice_vecs[1]
            # repeat translations for values array
            trans_values[even_ind*npts:(even_ind+1)*npts,:] = values
            trans_values[odd_ind*npts:(odd_ind+1)*npts,:] = values
            trans_values[(even_ind+2)*npts:(even_ind+3)*npts,:] = values
            trans_values[(odd_ind+2)*npts:(odd_ind+3)*npts,:] = values
            trans_values[(even_ind+4)*npts:(even_ind+5)*npts,:] = values
            trans_values[(odd_ind+4)*npts:(odd_ind+5)*npts,:] = values
            trans_values[(even_ind+6)*npts:(even_ind+7)*npts,:] = values
            trans_values[(odd_ind+6)*npts:(odd_ind+7)*npts,:] = values
            trans_values[(even_ind+8)*npts:(even_ind+9)*npts,:] = values
        return trans_points, trans_values
    #-----------------------------------------------------------------------------------------------------------------#
    # method for creating symmetrically equivalent points
    def _SymPtsAndVals(self, points:np.ndarray, values:np.ndarray, symtol:float=1.001
        )->tuple[np.ndarray, np.ndarray]:
        '''
        Method for generating full reciprocal space cell from irreducible kpoints\n
        *REQUIRES KPOINTS TO BE IN REDUCED FORMAT, CARTESIAN FORMAT WILL NOT WORK*

        Parameters
        ----------
        points : np.ndarray
            Irreducible set of kpoints, generally from GetValAndKpt
        values : np.ndarray
            Eigenvalues for irreducible kpoints, generally from GetValAndKpt
        bands : np.ndarray
            Band number that each kpoint belongs to
        symtol : float
            Used for checking symmetrically equivalent points outside of unit cell\n
            Should be greater than 1, default is 1.001
        '''
        # get symmetry operations and initialize symmetrically equivalent point and value arrays
        sym_ops = self.symrel
        ind_len = self.nkpt
        sym_pts = np.zeros((self.nsym*ind_len,3))
        sym_vals = np.zeros((self.nsym*ind_len,self.bands[0]))
        for i, op in enumerate(sym_ops):
            new_pts = np.matmul(points, op)
            sym_pts[i*ind_len:(i+1)*ind_len,:] = new_pts
            sym_vals[i*ind_len:(i+1)*ind_len,:] = values
        # points overlap on at edges of each symmetric block, remove duplicates
        sym_pts, unique_inds = np.unique(sym_pts, return_index=True, axis=0)
        sym_vals = np.take(sym_vals, unique_inds, axis=0)
        return sym_pts, sym_vals
    #-----------------------------------------------------------------------------------------------------------------#
    # method for interpolating grid with Scipy Interpolation
    def _ScipyInterpolate(self, grid_points:np.ndarray, interp_points:np.ndarray, values:np.ndarray, null_value:float
        )->np.ndarray:
        from scipy.interpolate import griddata
        interp_vals = griddata(interp_points, values, grid_points, fill_value=null_value)
        return interp_vals
    #-----------------------------------------------------------------------------------------------------------------#
    # method for computing isosurface color from BandU overlaps
    def _GetOverlapColor(self, grid:pv.ImageData, iso_surf_pts:np.ndarray, points:np.ndarray, overlap:np.ndarray,
                         radius:float, sharpness:float, strategy:str
        )->np.ndarray:
        color_grid = pv.ImageData()
        color_grid.origin = grid.origin
        color_grid.dimensions = grid.dimensions
        color_grid.spacing = grid.spacing
        trans_color_pts, trans_overlap = self._TranslatePoints(points, overlap, self.rec_lattice)
        trans_color_pts = pv.PolyData(trans_color_pts)
        trans_color_pts['values'] = trans_overlap
        color_grid = color_grid.interpolate(trans_color_pts, sharpness=sharpness, radius=radius, strategy=strategy)
        color_inds = []
        for iso_point in iso_surf_pts:
            ind = color_grid.find_closest_point(iso_point)
            color_inds.append(ind)
        color_values = np.take(color_grid['values'], color_inds)
        return color_values
    #-----------------------------------------------------------------------------------------------------------------#
    # method for making isosurface and creating PyVista plotter
    def _MakeIsosurface(self, points:np.ndarray, values:np.ndarray, translation:np.ndarray=None, 
                        grid:pv.ImageData=None, steps:tuple=None, radius:float=0.25, sharpness:float=2.0, 
                        strategy:str='null_value', null_value:float=None, isosurfaces:int=10, rng:list=None, 
                        smooth:bool=True, show_outline:bool=False, show_points:bool=False, show_isosurf:bool=True, 
                        show_vol:bool=False, scipy_interpolation:bool=False, width:float=0.0002, color:np.ndarray=None, 
                        overlap:np.ndarray=None
        )->None:
        # translate points to fill grid space
        if translation == None:
            translation = self.rec_lattice
        trans_pts, trans_vals = self._TranslatePoints(points, values, translation)
        # create grid if one is not provided
        if grid == None and steps == None:
            grid = self.MakeGrid(points)
        elif grid == None:
            grid = self.MakeGrid(points, steps=steps)
        # interpolate grid
        trans_pts = pv.PolyData(trans_pts)
        trans_pts['values'] = trans_vals
        if null_value == None:
                null_value = self.fermi+width/2*1.05
        if scipy_interpolation:
            grid['values'] = self._ScipyInterpolate(grid.points, 
                                                    trans_pts.points, 
                                                    trans_pts['values'], 
                                                    null_value
            )
        else:
            grid = grid.interpolate(trans_pts, 
                                    radius=radius, 
                                    sharpness=sharpness, 
                                    strategy=strategy, 
                                    null_value=null_value
            )
        # create isosurface
        if rng == None:
            rng = [self.fermi, self.fermi]
        iso_surf = grid.contour(isosurfaces=isosurfaces, rng=rng, method='contour')
        # apply Taubin smoothing
        if smooth:
            iso_surf = iso_surf.smooth_taubin(n_iter=100, pass_band=0.05)
        # find overlap colors on surface
        if type(overlap) == np.ndarray:
            color = self._GetOverlapColor(grid, 
                                          iso_surf.points, 
                                          points, 
                                          overlap,
                                          radius,
                                          sharpness,
                                          strategy
                    )
            print(color)
        # open plotter and add meshes
        if show_outline:
            self.p.add_mesh(grid.outline())
        if show_points:
            pts = pv.PolyData(points)
            self.p.add_mesh(pts.points, color='black')
        if show_vol:
            actor = self.p.add_volume(grid)
            actor.prop.interpolation_type = 'linear'
        if show_isosurf:
            self.p.add_mesh(iso_surf, 
                            style='surface',
                            smooth_shading=smooth, 
                            lighting=True,
                            ambient=1.0,
                            diffuse=1.0,
                            specular=0.8,
                            pbr=True,
                            metallic=0.7,
                            roughness=0.5,
                            scalars=color
                            )
    #-----------------------------------------------------------------------------------------------------------------#
    # method for getting isosurface color from BandU overlap with isosurface states
    def _BandUColor(self, energy_level:float, width:float, states:int
        )->tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        eigfuncs = []
        for eigfunc in self.BandU(energy_level=energy_level, width=width, states=states, XSFFormat=False):
            eigfuncs.append(eigfunc)
        if states > len(eigfuncs):
            states = len(eigfuncs)
            print(f'There are only {states} BandU functions, setting BandU parameter to {states}')
        states -= 1
        BandU_func = eigfuncs[states]
        kpts, eigvals, bands, overlaps = self._GetOverlapValandKpt(energy_level=energy_level, 
                                                                    width=width, 
                                                                    BandU=BandU_func
        )
        return kpts, eigvals, bands, overlaps
    #-----------------------------------------------------------------------------------------------------------------#
    # Render isosurfaces
    def _Render(self)->None:
        self.p.add_text(f'Fermi Energy:{self.fermi} H', font_size=12)
        self.p.show()
        self.p.app.exec_()
    #-----------------------------------------------------------------------------------------------------------------#
    # wrapper around class methods to plot isosurface with one function call
    def PlotIsosurface(self, translation:np.ndarray=None, grid:pv.ImageData=None,
                       steps:tuple=None, radius:float=0.25, sharpness:float=2.0, strategy:str='null_value', 
                       null_value:float=None, isosurfaces:int=2, rng:list=None, smooth:bool=True, 
                       show_outline:bool=False, show_points:bool=False, show_isosurf:bool=True, show_vol:bool=False,
                       scipy_interpolation:bool=False, width:float=0.0002, color:np.ndarray=None, BandU:int=0,
                       energy_level:float=None
        )->None:
        # find BandU eigen fxns and compute overlaps of eigen fxns with reciprocal space states
        # these overlap values will be used to color isosurface
        if BandU > 0:
            kpts, eigvals, bands, overlaps = self._BandUColor(energy_level, width, BandU)
            _, overlaps = self._SymPtsAndVals(kpts, overlaps)
        # if BandU is not specified, just recover kpts, eigenvals, and bands
        # this will construct the isosurface with monochromatic coloration
        else:
            kpts, eigvals, bands = self._GetValAndKpt(energy_level=energy_level, width=width)
        kpts, eigvals = self._SymPtsAndVals(kpts, eigvals)
        kpts = np.matmul(kpts, self.rec_lattice)
        for band in bands:
            energies = eigvals[:,band].reshape((len(eigvals[:,band]),1))
            if BandU > 0:
                overlap = overlaps[:,band].reshape((len(overlaps[:,band]),1))
            else:
                overlap = None
            self._MakeIsosurface(kpts, energies, width=width, translation=translation, grid=grid, steps=steps, 
                                 radius=radius, sharpness=sharpness, strategy=strategy, null_value=null_value, 
                                 isosurfaces=isosurfaces, rng=rng, smooth=smooth, show_outline=show_outline, 
                                 show_points=show_points, show_isosurf=show_isosurf, show_vol=show_vol, 
                                 scipy_interpolation=scipy_interpolation, color=color, overlap=overlap
            )
        self._Render()