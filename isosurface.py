from wfk_class import WFK
from xsf_reader import XSF
from typing import Generator
import numpy as np
from copy import copy
from scipy.spatial import Voronoi, ConvexHull, Delaunay
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import pickle as pkl

class Isosurface():
    def __init__(
            self, wfks:Generator=None, depth_peeling:bool=True, save:bool=True, save_file:str='Fermi_surface.pkl',
            empty_mesh:bool=False, energy_level:float=0.0, width:float=0.005
    )->None:
        '''
        Class for creating energy isosurfaces with the PyVista library from WFK object.

        Parameters
        ----------
        wfks : Generator
            An iterable generator of WFK objects.
        depth_peeling : bool
            Enables parts of surface to remain invisible without effecting visible surface opacity.
            Default enables depth peeling (True).
        save : bool
            Saves surface as pickle for rerendering later and post analysis.
            Default saves pickle file (True).
        save_file : str
            Name of save file.
            Default "Fermi_surface.pkl".
        empty_mesh : bool
            If no surface can be constructed, then proceed with plotting anyways.
            Default is to throw exception if surface cannot be constructed (False).
        energy_level : float
            This determines what energy the isosurface is created at.
            Set relative to the Fermi energy, default is 0.0. 
        width : float
            Determines how far above and below to look for states about an energy_level.
            Looks width/2 above and below energy_level, default is 0.005 (Hartree is assumed).
        '''
        self.wfks = wfks
        self.p:pv.Plotter = BackgroundPlotter(window_size=(600,400))
        self.save = save
        self.save_file = save_file
        pv.global_theme.allow_empty_mesh = empty_mesh
        if depth_peeling:
            self.p.enable_depth_peeling(number_of_peels=10)
    #-----------------------------------------------------------------------------------------------------------------#
    # method for getting eigenvalues and kpoints within specified energy range
    def Surface(
            self, energy_level:float, width:float
    )->tuple[np.ndarray, np.ndarray, list]:
        # lists energy values to for making isosurface
        band_num = []
        # loop through generator of WFK objects
        state:WFK
        for i, state in enumerate(self.wfks):
            # first loop collect necessary attributes from WFK object
            if i == 0:
                self.ngfftx = state.ngfftx
                self.ngffty = state.ngffty
                self.ngfftz = state.ngfftz
                self.nbands = state.nbands
                self.kpoints = state.kpoints
                self.nkpt = state.nkpt
                self.fermi_energy = state.fermi_energy
                self.symrel = state.symrel
                self.nsym = state.nsym
                energy_level += self.fermi_energy
                min_nrg = energy_level - width/2
                max_nrg = energy_level + width/2
                iso_values = np.zeros((self.nkpt,self.nbands))
                self.lattice = state.lattice
                self.rec_lattice = state.Real2Reciprocal()
            # pick up eigenvalues
            iso_values[i,:] = state.eigenvalues
            # check if band crosses into energy width
            for band, eigval in enumerate(iso_values[i,:]):
                if min_nrg <= eigval <= max_nrg:
                    band_num.append(band)
                    # only 1 value per kpoint (if there is a value) move to next kpt when found
                    break
        # collect kpoints and unique bands that cross into width
        iso_pts = np.array(self.kpoints).reshape((self.nkpt,3))
        band_num = set(band_num)
        return iso_pts, iso_values, band_num
    #-----------------------------------------------------------------------------------------------------------------#
    # method for finding overlap value of BandU function with k-states
    def _FindOverlap(
            self, state:WFK, bandu_fxn:np.ndarray
    )->np.ndarray:
        overlap_vals = []
        for i in range(state.nbands):
            temp_state = copy(state)
            temp_state = temp_state.GridWFK(band_index=i)
            temp_state = temp_state.FFT()
            overlap = np.sum(np.conj(bandu_fxn)*temp_state.wfk_coeffs)
            overlap_vals.append(np.square(np.abs(overlap)))
        return np.array(overlap_vals).real
    #-----------------------------------------------------------------------------------------------------------------#
    # method getting eigenvalues and kpts within defined energy range, also get overlap of BandU eigfuncs w/ states
    def SurfaceAndColor(
            self, bandu_fxn:np.ndarray, energy_level:float=None, width:float=0.0005
    )->tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        # list for picking which bands cross energy width
        band_num = []
        # loop through generator of WFK objects
        state:WFK
        for i, state in enumerate(self.wfks):
            # first loop collect necessary attributes from WFK object
            if i == 0:
                self.ngfftx = state.ngfftx
                self.ngffty = state.ngffty
                self.ngfftz = state.ngfftz
                self.nbands = state.nbands
                self.kpoints = state.kpoints
                self.nkpt = state.nkpt
                self.symrel = state.symrel
                self.nsym = state.nsym
                self.fermi_energy = state.fermi_energy
                energy_level += self.fermi_energy
                self.lattice = state.lattice
                self.rec_lattice = state.Real2Reciprocal()
                min_nrg = energy_level - width/2
                max_nrg = energy_level + width/2
                iso_values = np.zeros((self.nkpt, self.nbands))
                overlap_vals = np.zeros((self.nkpt, self.nbands))
            # pick up eigenvalues
            iso_values[i,:] = state.eigenvalues
            # compute overlap of eigenstates with BandU function
            overlap_vals[i,:] = self._FindOverlap(state, bandu_fxn)
            # check if band crosses into energy width
            for band, eigval in enumerate(iso_values[i,:]):
                if min_nrg <= eigval <= max_nrg:
                    band_num.append(band)
                    # only 1 value per kpoint (if there is a value) move to next kpt when found
                    break
        # collect kpoints and unique bands that cross into width
        iso_pts = np.array(self.kpoints).reshape((self.nkpt,3))
        band_num = set(band_num)
        return iso_pts, iso_values, band_num, overlap_vals
    #-----------------------------------------------------------------------------------------------------------------#
    # method constructing PyVista grid
    def MakeGrid(
            self, points:np.ndarray, steps:tuple=(50,50,50)
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
        grid.origin = (xmin-0.035, ymin-0.035, zmin-0.035)
        grid.spacing = (2*(xmax+0.05)/dimx, 2*(ymax+0.05)/dimy, 2*(zmax+0.05)/dimz)
        grid.dimensions = (dimx, dimy, dimz)
        return grid
    #-----------------------------------------------------------------------------------------------------------------#
    # method translating points to fill grid space
    def _TranslatePoints(
            self, points:np.ndarray, values:np.ndarray, lattice_vecs:np.ndarray
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
    # method for interpolating grid with Scipy Interpolation
    def _ScipyInterpolate(
            self, grid_points:np.ndarray, interp_points:np.ndarray, values:np.ndarray, null_value:float
    )->np.ndarray:
        from scipy.interpolate import griddata
        interp_vals = griddata(interp_points, values, grid_points, fill_value=null_value)
        return interp_vals
    #-----------------------------------------------------------------------------------------------------------------#
    # method for computing isosurface color from BandU overlaps
    def _GetOverlapColor(
            self, grid:pv.ImageData, iso_surf_pts:np.ndarray, points:np.ndarray, overlap:np.ndarray, radius:float, 
            sharpness:float, strategy:str
    )->np.ndarray:
        # make grid for interpolating overlap values
        color_grid = pv.ImageData()
        # same parameters as eigenvalue grid
        color_grid.origin = grid.origin
        color_grid.dimensions = grid.dimensions
        color_grid.spacing = grid.spacing
        trans_color_pts, trans_overlap = self._TranslatePoints(points, overlap, self.rec_lattice)
        trans_color_pts = pv.PolyData(trans_color_pts)
        trans_color_pts['values'] = trans_overlap
        color_grid = color_grid.interpolate(
            trans_color_pts, 
            sharpness=sharpness, 
            radius=radius, 
            strategy=strategy
        )
        # to project colors onto isosurface, use a closest point strategy
        # since grids have the same parameters, this gives accurate projection
        color_inds = []
        for iso_point in iso_surf_pts:
            ind = color_grid.find_closest_point(iso_point)
            color_inds.append(ind)
        color_values = np.take(color_grid['values'], color_inds)
        return color_values
    #-----------------------------------------------------------------------------------------------------------------#
    # method for determining which parts of the isosurface are within the Brillouin zone
    def _SetOpacity(
            self, points:np.ndarray, hull_points:np.ndarray, delaunay:bool
    )->np.ndarray:
        if delaunay:
            opacities = Delaunay(hull_points).find_simplex(points)
            opacities[opacities >= 0] = 1
            opacities[opacities < 0] = 0
        else:
            hull = ConvexHull(hull_points)
            opacities = np.ones(len(points))
            for i, pt in enumerate(points):
                test_hull = ConvexHull(np.concatenate((hull_points, [pt]), axis=0))
                if not np.array_equal(test_hull.vertices, hull.vertices):
                    opacities[i] = 0
        return opacities
    #-----------------------------------------------------------------------------------------------------------------#
    # method for making isosurface and creating PyVista plotter
    def _MakeIsosurface(
            self, points:np.ndarray, values:np.ndarray, translation:np.ndarray=None, 
            grid:pv.ImageData=None, steps:tuple=None, radius:float=0.25, sharpness:float=2.0, 
            strategy:str='null_value', null_value:float=None, isosurfaces:int=10, rng:list=None, 
            smooth:bool=True, show_outline:bool=False, show_points:bool=False, show_isosurf:bool=True, 
            show_vol:bool=False, scipy_interpolation:bool=False, width:float=0.0002, color:str='green',
            scalars:np.ndarray=None, overlap:np.ndarray=None, colormap:str='plasma', 
            hull:np.ndarray=None, delaunay:bool=True, lighting:bool=True, ambient:float=0.5, 
            diffuse:float=0.5, specular:float=1.0, specular_power:float=128.0, pbr:bool=False, 
            metallic:float=0.5, roughness:float=0.5, energy_level:float=0.0
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
            null_value = self.fermi_energy+energy_level+width/2*1.05
        if scipy_interpolation:
            grid['values'] = self._ScipyInterpolate(
                grid.points, 
                trans_pts.points, 
                trans_pts['values'], 
                null_value
            )
        else:
            grid = grid.interpolate(
                trans_pts, 
                radius=radius, 
                sharpness=sharpness, 
                strategy=strategy, 
                null_value=null_value,
            )
        # create isosurface
        if rng == None:
            rng = [self.fermi_energy, self.fermi_energy]
        else:
            rng = [self.fermi_energy + rng[0], self.fermi_energy + rng[1]]
        iso_surf:pv.PolyData = grid.contour(isosurfaces=isosurfaces, rng=rng, method='contour')
        opacities = self._SetOpacity(iso_surf.points, hull, delaunay)
        # apply Taubin smoothing
        if smooth:
            iso_surf = iso_surf.smooth_taubin(n_iter=100, pass_band=0.05)
        # find overlap colors on surface
        if type(overlap) == np.ndarray:
            scalars = self._GetOverlapColor(
                grid, 
                iso_surf.points, 
                points, 
                overlap,
                radius,
                sharpness,
                strategy
            )
            scalars += 10**(-15)
        if self.save:
            with open(self.save_file, 'ab') as f:
                pkl.dump(iso_surf, f)
                pkl.dump(opacities, f)
                pkl.dump(scalars, f)
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
            self.p.add_mesh(
                iso_surf, 
                style='surface',
                smooth_shading=smooth, 
                lighting=lighting,
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
                specular_power=specular_power,
                pbr=pbr,
                metallic=metallic,
                roughness=roughness,
                scalars=scalars,
                cmap=colormap,
                opacity=opacities,
                color=color,
                show_scalar_bar=True
            )
    #-----------------------------------------------------------------------------------------------------------------#
    # method for getting isosurface color from BandU overlap with isosurface states
    def _BandUColor(
            self, energy_level:float, width:float, states:int, xsf_root:str=None, xsf_nums:list=[1]
    )->tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        # function only used by _BandUColor for reading in all BandU XSF files necessary
        def _ReadXSFs()->np.ndarray:
            for num in xsf_nums:
                real_path = xsf_root + f'_bandu_{num}_real.xsf'
                real_xsf = XSF(real_path)
                real_func = real_xsf.ReadDensity()
                imag_path = xsf_root + f'_bandu_{num}_imag.xsf'
                imag_xsf = XSF(imag_path)
                imag_func = 1j*imag_xsf.ReadDensity()
            eigfunc = real_func + imag_func
            return eigfunc
        # read in BandU eigenfunction from XSF file
        if xsf_root != None:
            BandU_func = _ReadXSFs()
            BandU_func = WFK(
                wfk_coeffs=BandU_func,
                ngfftx=np.shape(BandU_func)[1],
                ngffty=np.shape(BandU_func)[2],
                ngfftz=np.shape(BandU_func)[0]
            ).RemoveXSF()
        # or calculate eigenfunction from WFK
        else:
            raise NotImplementedError(
                '''Currently this program only supports reading in density from an XSF file.
                Action: Set 'read_xsf' to True and provide 'xsf_root' with the root name of the XSF file.
                '''
            )
            eigfuncs = []
            for eigfunc in self.BandU(energy_level=energy_level, width=width, states=states, XSFFormat=False):
                eigfuncs.append(eigfunc)
            if states > len(eigfuncs):
                states = len(eigfuncs)
                print(f'There are only {states} BandU functions, setting BandU parameter to {states}')
            states -= 1
            BandU_func = eigfuncs[states]
        kpts, eigvals, bands, overlaps = self.SurfaceAndColor(
            energy_level=energy_level, 
            width=width, 
            bandu_fxn=BandU_func.wfk_coeffs
        )
        return kpts, eigvals, bands, overlaps
    #-----------------------------------------------------------------------------------------------------------------#
    # method for finding Brillouin zone
    def _GetBZ(
            self, BZ_width:float
    )->np.ndarray:
        # function used only by _GetBZ
        def CheckPts(
                pts:np.ndarray, verts:np.ndarray
        )->np.ndarray:
            # get cartesian coordinates
            verts = np.take(verts, pts, axis=0)
            # last point is the furthest, check if it is in same plane as other points
            pt = verts[-1,:]
            check = False
            # truncate floats to avoid floating point errors when checking for equivalence
            pt = [np.sign(n)*float(str(np.abs(n))[:8]) for n in pt]
            for i, vert in enumerate(verts):
                verts[i,:] = [np.sign(n)*float(str(np.abs(n))[:8]) for n in vert]
            # check if points are in same plane
            if pt[0] == verts[0,:][0] and pt[0] == verts[1,:][0] and pt[0] == verts[2,:][0]:
                check = True 
            if pt[1] == verts[0,:][1] and pt[1] == verts[1,:][1] and pt[1] == verts[2,:][1]:
                check = True
            if pt[2] == verts[0,:][2] and pt[2] == verts[1,:][2] and pt[2] == verts[2,:][2]:
                check = True
            return check
        # get reciprocal space lattice points
        vor_points, _ = self._TranslatePoints(np.zeros(3), np.zeros(1), self.rec_lattice)
        vor_points = np.unique(vor_points, axis=0)
        # Voronoi tessallate the lattice points
        vor_tessellation = Voronoi(vor_points)
        # find Voronoi region that encapsulates Brillouin zone
        for i, region in enumerate(vor_tessellation.regions):
            if -1 not in region and region != []:
                vor_cell = i
                break
        # get vertices of BZ Voronoi region
        vor_cell_verts = vor_tessellation.regions[vor_cell]
        vor_verts = np.take(vor_tessellation.vertices, vor_cell_verts, axis=0)
        # find which vertices are closest to each other
        point_cloud = pv.PolyData(vor_verts)
        vert_inds = []
        for i, vert in enumerate(vor_verts):
            nearest_pts = point_cloud.find_closest_point(vert, n=4)
            c = 4
            # iterate check until points are no longer in same plane
            while CheckPts(nearest_pts, vor_verts):
                c += 1
                nearest_pts = point_cloud.find_closest_point(vert, n=c)
                len_arr = c - 4
                arr = -1*np.arange(len_arr) - 2
                nearest_pts = np.delete(nearest_pts, arr, axis=0)
            # current index is a repeated entry every other point since this is how PyVista interprets lines
            nearest_pts = np.insert(nearest_pts, [2,4], [i,i])
            vert_inds.append(nearest_pts)
        # convert nested list to flat list
        vert_inds = [int(ind) for verts in vert_inds for ind in verts]
        # get cartesian coordinates
        vor_verts = np.take(vor_verts, vert_inds, axis=0)
        # add BZ edges as lines to plotter
        self.p.add_lines(vor_verts, color='black', width=BZ_width)
        return vor_verts
    #-----------------------------------------------------------------------------------------------------------------#
    # Render isosurfaces
    def _Render(
            self, show_energy:bool=False
    )->None:
        if show_energy:
            self.p.add_text(f'Fermi Energy:{self.fermi_energy} H', font_size=12)
        self.p.enable_parallel_projection()
        self.p.enable_custom_trackball_style(left='rotate',
                                             shift_left='spin',
                                             right='pan',
        )
        self.p.set_focus([0.0,0.0,0.0])
        self.p.show()
        self.p.app.exec_()
    #-----------------------------------------------------------------------------------------------------------------#
    # wrapper around class methods to plot isosurface with one function call
    def PlotIsosurface(
            self, translation:np.ndarray=None, grid:pv.ImageData=None,
            steps:tuple=None, radius:float=0.25, sharpness:float=2.0, strategy:str='null_value', 
            null_value:float=None, isosurfaces:int=2, rng:list=None, smooth:bool=True, 
            show_outline:bool=False, show_points:bool=False, show_isosurf:bool=True, show_vol:bool=False,
            scipy_interpolation:bool=False, width:float=0.0002, color:str='green', BandU:int=1,
            energy_level:float=None, read_xsf:bool=True, xsf_root:str=None, xsf_nums:list=[1],
            bandu_width:float=None, BZ_width:float=2.5, delaunay:bool=True, colormap:str='plasma', 
            lighting:bool=True, ambient:float=0.5, diffuse:float=0.5, specular:float=1.0, 
            specular_power:float=128.0, pbr:bool=False, metallic:float=0.5, roughness:float=0.5
    )->None:
        # find BandU eigen fxns and compute overlaps of eigen fxns with reciprocal space states
        # these overlap values will be used to color isosurface
        if read_xsf:
            if bandu_width == None:
                bandu_width = width
            kpts, eigvals, bands, overlaps = self._BandUColor(
                energy_level, 
                bandu_width, 
                BandU, 
                xsf_root=xsf_root,
                xsf_nums=xsf_nums
            )
            temp = WFK(kpoints=kpts, eigenvalues=overlaps, symrel=self.symrel, nsym=self.nsym, nkpt=self.nkpt, 
                       nbands=self.nbands)
            _, overlaps = temp.Symmetrize(temp.kpoints, temp.eigenvalues)
        # if BandU is not specified, just recover kpts, eigenvals, and bands
        # this will construct the isosurface with monochromatic coloration
        else:
            kpts, eigvals, bands = self.Surface(energy_level=energy_level, width=width)
        # apply symmetry operations to kpoints
        temp = WFK(kpoints=kpts, eigenvalues=eigvals, symrel=self.symrel, nsym=self.nsym, nkpt=self.nkpt, 
                   nbands=self.nbands)
        kpts, eigvals = temp.Symmetrize(temp.kpoints, temp.eigenvalues)
        # convert reduce coordinates to Cartesian coordinates
        kpts = np.matmul(kpts, self.rec_lattice)
        # get Brillouin zone and add it to plotter
        hull = self._GetBZ(BZ_width)
        # plot isosurface by band
        if self.save:
            with open(self.save_file, 'ab') as f:
                pkl.dump(bands, f)
                pkl.dump(hull, f)
                pkl.dump(self.rec_lattice, f)
        for band in bands:
            # get all eigenvalues for current iterated band
            energies = eigvals[:,band].reshape((len(eigvals[:,band]),1))
            # get all overlap values if desired
            if read_xsf:
                overlap = overlaps[:,band].reshape((len(overlaps[:,band]),1))
            else:
                overlap = None
            # make PyVista surface objext
            self._MakeIsosurface(
                kpts, energies, width=width, translation=translation, grid=grid, steps=steps, 
                radius=radius, sharpness=sharpness, strategy=strategy, null_value=null_value, 
                isosurfaces=isosurfaces, rng=rng, smooth=smooth, show_outline=show_outline, 
                show_points=show_points, show_isosurf=show_isosurf, show_vol=show_vol, 
                scipy_interpolation=scipy_interpolation, color=color, overlap=overlap, hull=hull,
                delaunay=delaunay, colormap=colormap, lighting=lighting, ambient=ambient, 
                diffuse=diffuse, specular=specular, specular_power=specular_power, pbr=pbr, 
                metallic=metallic, roughness=roughness, energy_level=energy_level
            )
        # render plot
        self._Render()