import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from copy import copy
import pickle as pkl
from matplotlib.colors import ListedColormap
from . import brillouin_zone as brlzn
from . import isosurface_class as ic
from . import wfk_class as wc
from . import translate as trslt

class Plotter():
    '''
    Class for creating 3D plot of Fermi surface and projection of BandU functions on Fermi surface.

    Parameters
    ----------
    isosurface : Isosurface
        An Isosurface object that contains the contours to be plotted
    save : bool
        Create save file of isosurface plot that can be loaded with the Load method
        Default will save the plot (True)
    save_file : str
        Name of save file
        Default is "Fermi_surface.pkl"
    empty_mesh : bool
        Allow PyVista plotter to plot meshes even if no surface is present
        Default will throw exception of an empty surface is being plotted (False)
    plot : bool
        Enable creation of PyVista Plotter
        Must be enabled for plotting, default enables plotter (True)

    Methods
    -------
    Plot
        Generates 3D plot of Fermi surface from contours of Isosurface object
    SurfaceColor
        Reads in a BandU XSF file and Fermi surface ABINIT WFK file to calculate the overlap between the BandU
        function and the states at the Fermi energy
    Load
        Loads a save file
    '''
    def __init__(
        self, isosurface:ic.Isosurface=ic.Isosurface(points=np.ones((1,3))), save:bool=True,
        empty_mesh:bool=False, _debug:bool=False, save_file:str='Fermi_surface.pkl', plot:bool=True
    ):
        self.isosurface=isosurface
        self.save=save
        self.save_file=save_file
        self._debug=_debug
        self.plot=plot
        if self.plot:
            self.p:pv.Plotter=BackgroundPlotter(window_size=(600,400))
            pv.global_theme.allow_empty_mesh=empty_mesh
            self.p.enable_depth_peeling(number_of_peels=10)
#---------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ METHODS ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#
    # method for turning opacity to zero outside of BZ
    def _SetOpacities(
        self, contour:pv.PolyData
    )->np.ndarray:
        bz = brlzn.BZ(self.isosurface.rec_latt)
        opacities = bz.PointLocate(contour.points, cart=False)
        opacities[opacities >= 0] = 1
        opacities[opacities < 0] = 0
        return opacities
    #-----------------------------------------------------------------------------------------------------------------#
    # method to add nesting vector to isosurface plot
    def _AddArrow(
            self, arrow:list, rec_lattice:np.ndarray, show_endpoints:bool, color:str
    )->None:
        tail = np.array(arrow[0])
        shift = np.array(arrow[1])
        tail = np.matmul(tail, rec_lattice)
        shift = np.matmul(shift, rec_lattice)
        scale = np.linalg.norm(shift).astype(float)
        py_arrow = pv.Arrow(
            start=tail,
            direction=shift,
            tip_radius=0.05/scale,
            tip_length=0.15/scale,
            shaft_radius=0.025/scale,
            scale=scale
        )
        if show_endpoints:
            points = np.array([tail, shift+tail])
            points = pv.PolyData(points)
            self.p.add_mesh(points.points, point_size=20, color='black', render_points_as_spheres=True)
        self.p.add_mesh(py_arrow, color=color)
    #-----------------------------------------------------------------------------------------------------------------#
    # method to visualize cross-section of surface
    def _CrossSection(
            self, vecs:list, points:np.ndarray, width:float, rec_lattice:np.ndarray, bz_points:np.ndarray,
            linear:bool=False, two_dim:bool=False
    )->np.ndarray:
        from scipy.spatial import Delaunay
        # cross section is defined by plane of two perpendicular vectors
        if len(vecs) == 2:
            vec1 = np.matmul(vecs[0], rec_lattice)
            vec2 = np.matmul(vecs[1], rec_lattice)
            norm = np.cross(vec1, vec2)
            norm /= np.linalg.norm(norm)
        # cross section is defined by normal vector
        else:
            vec = np.matmul(vecs, rec_lattice)
            norm = vec/np.linalg.norm(vec)
        # surface points can be thought of vectors in 3D space, here we normalize the vectors
        norm_points = np.array(points/np.linalg.norm(points, axis=1).reshape((len(points),1)))
        # find dot product between vector normal to cross section and all normalized points
        angs = np.matmul(norm, norm_points.T)
        # opacity linearly fades out as points get farther from cross section
        if linear:
            angs[angs <= width] = 2
            angs -= 1
            angs = np.abs(angs)
        # opacity is zero beyond width of cross section
        else:
            # opacity is zero beyond width on both sides of cross section
            if two_dim:
                angs = np.abs(angs)
            opacities = np.zeros(len(points))
            opacities[angs <= width] = 1
            angs = opacities  
        # set opacity to zero for points outside of the BZ
        beyond_bz = Delaunay(bz_points).find_simplex(points)
        angs[beyond_bz < 0] = 0
        return angs
    #-----------------------------------------------------------------------------------------------------------------#
    # method for plotting isosurface contours
    def Plot(
        self, show_points:bool=False, show_outline:bool=False, show_axes:bool=True, show_isosurf:bool=True,
        smooth:bool=True, lighting:bool=True, ambient:float=0.5, diffuse:float=0.5, specular:float=1.0, 
        specular_power:float=128.0, pbr:bool=False, metallic:float=0.5, roughness:float=0.5,
        colormap:str|ListedColormap='plasma', color:str='white', bz_show:bool=True, bz_width:int=3, arrow:list=[], 
        show_endpoints:bool=False, arrow_color:str='yellow', periodic_arrow:list=[], camera_position:list=[], 
        cross_section:list=[], cross_width:float=0.1, linear:bool=False, two_dim:bool=True, show_bands:float|list=1.0,
        surface_vals:list|None=None, show_ird_points:bool=False
    ):
        '''
        Method for plotting contours made from the Isosurface object

        Parameters
        ----------
        colormap : str | ListedColormap
            Choose colormap for isosurface that colors according to assigned scalars of surface\n
            Can use Colors class to use default or create custom colormaps\n
            Default is matplotlib's plasma
        bz_width : int
            Line width of Brillouin Zone\n
            Default is 3
        bz_show : bool
            Show the Brillouin Zone
            Default is to show (True)
        smooth : bool
            Use smooth lightning techniques\n
            Default is to use smoothing (True)
        lighting : bool
            Apply directional lighting to surface\n
            Default is to enable directional lighting (True)
        ambient : float
            Intensity of light on surface\n
            Default is 0.5
        diffuse : float
            Amount of light scattering\n
            Default is 0.5
        specular : float
            Amount of reflected light\n
            Default is 1.0 (max)
        specular_power : float
            Determines how sharply light is reflected\n
            Default is 128.0 (max)
        pbr : bool
            Apply physics based rendering\n
            Default is no physics based rendering (False)
        metallic : float
            Determine how metallic-looking the surface is, only considered with pbr\n
            Default is 0.5
        roughness : float
            Determine how smooth/rough surface appear, only considered with pbr\n
            Default is 0.5
        color : str
            Sets color of surface (colormap overwrites this)\n
            Sets color of reflected light (colormap does not overwrite this)\n
            Default is white
        arrow : list
            Parameters for plotting nesting vector on top of Fermi surface\n
            Element_0 of list should be starting (or tail) position of arrow\n
            Element_1 of list should be orientation of arrow with desired magnitude\n
            Both the tail and orientation should be specified in reduced reciprocal space coordinates
        arrow_color : str
            Color of nesting arrow\n
            Default is black
        show_endpoints : bool
            Plot points on the end of the arrow to make visualizing start and end easier\n
            Default is to not show endpoints (False)
        periodic_arrow : list
            Adds periodic image of arrow that is translated [X,Y,Z] cells\n
            Where X, Y, and Z are the cell indices
        show_bands : float | list
            Specifies the opacity of each band\n
            If a single float is provided, all bands will be plotted with the same opacity\n
            If a list is provided, each band will be plotted with the opacity of the respective list element
        show_axes : bool
            Plots reciprocal cell axes with a* as red, b* as green, and c* as blue\n
            Default is to show axes (True)
        cross_section : list
            Plot cross section through surface\n
            If one vector is provided, it is assumed to be the normal to the cross section plane\n
            Else, cross section is defined by plane made by two vectors\n
            Vectors should be specified in reduced coordinates
        cross_width : float
            Width of cross section\n
            Default is 0.15
        linear : bool
            Cross section linearly fades out\n
            Default is not fade out linearly (False)
        two_dim : bool
            Cross section is a 2D slice instead of a section\n
            Default is to show cross section as 2D slice (True)
        surface_vals : np.ndarray
            A list of values defining the coloration of the isosurface
            Default plots no coloration
        '''
        # save file
        if self.save:
            with open(self.save_file, 'wb') as f:
                kwargs = locals()
                kwargs.pop('self', None)
                kwargs.pop('f', None)
                pkl.dump(kwargs, f)
                pkl.dump(self.isosurface, f)
        if not self.plot:
            raise SystemExit()
        # plot BZ boundary
        vor_verts = brlzn.BZ(self.isosurface.rec_latt).vertices
        if bz_show:
            self.p.add_lines(vor_verts, color='black', width=bz_width)
        # set opacity of each band
        if type(show_bands) is float:
            band_ops = show_bands*np.ones(len(self.isosurface.contours), dtype=float)
            show_bands = band_ops.tolist()
        # loop through contours of isosurface object and plot each
        for i, contour in enumerate(self.isosurface.contours):
            if surface_vals is None:
                scalars = None
                surf_max = 0.0
            else:
                scalars = surface_vals[i]
                surf_max = np.max(surface_vals[i])
            # opacities can be adjusted to show a cross section of contours
            if cross_section != []:
                opacities = self._CrossSection(
                    cross_section,
                    contour.points, 
                    cross_width, 
                    self.isosurface.rec_latt, 
                    vor_verts,
                    linear=linear,
                    two_dim=two_dim
                )
            # otherwise opacities are set to just render contours in the BZ
            else:
                opacities = self._SetOpacities(contour=contour)
            # set opacity of bands
            opacities = [op*show_bands[i] for op in opacities] # type: ignore
            # lighting is set to make surface appear more smooth
            if smooth:
                contour = contour.smooth_taubin(n_iter=100, pass_band=0.05)
            # plot contours
            if show_isosurf:
                self.p.add_mesh(
                    contour, 
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
                    clim=[0.0,surf_max],
                    cmap=colormap,
                    opacity=opacities,
                    color=color,
                    show_scalar_bar=True,
                )
        # plot irreducible kpoints
        if show_ird_points:
            pts = pv.PolyData(self.isosurface.ir_kpts)
            self.p.add_mesh(pts.points, color='black')
        # plot points that are used to construct isosurface
        if show_points:
            pts = pv.PolyData(self.isosurface.points)
            self.p.add_mesh(pts.points, color='black')
        # plot outline of grid used in interpolation
        if show_outline:
            self.p.add_mesh(self.isosurface.grid.outline())
        # plot reciprocal space axes
        if show_axes:
            axes_colors = ['red', 'green', 'blue']
            for i in range(3):
                axis = self.isosurface.rec_latt[i,:]
                color = axes_colors[i]
                self._AddArrow([[0,0,0], axis], np.identity(3), False, color)
        # plot nesting vector
        if arrow != []:
            self._AddArrow(arrow, self.isosurface.rec_latt, show_endpoints, arrow_color)
        # plot a periodic image of the nesting vector
        if periodic_arrow != []:
            tail = np.array(arrow[0])
            cell = np.array(periodic_arrow, dtype=float)
            tail += cell
            arrow[0] = tail
            self._AddArrow(arrow, self.isosurface.rec_latt, show_endpoints, arrow_color)
        # set camera position
        if camera_position != []:
            camera_pos = np.array(camera_position).reshape((3,3))
            camera_pos = np.matmul(camera_pos, self.isosurface.rec_latt)
            self.p.camera_position = camera_pos
        self._Render()
    #-----------------------------------------------------------------------------------------------------------------#
    # Render isosurfaces
    def _Render(
        self
    ):
        self.p.enable_parallel_projection() # type: ignore
        self.p.enable_custom_trackball_style(
            left='rotate',
            shift_left='spin',
            right='pan',
        ) # type: ignore
        self.p.set_focus([0.0,0.0,0.0]) # type: ignore
        self.p.show()
        self.p.app.exec_() # type: ignore
    #-----------------------------------------------------------------------------------------------------------------#
    # method to compute bandu fxn and one electron wfk overlaps
    def _OverlapsWithSym(
        self, ir_wfk:wc.WFK, bandu:wc.WFK
    )->np.ndarray:
        # kpoint to symmetrically generate
        kpt = ir_wfk.kpoints
        # find number of distinct symmetries 
        sym_kpoints, _ = ir_wfk.Symmetrize(kpt, unique=False, reciprocal=True)
        dupes, unique_inds = ir_wfk._FindOrbit(sym_kpoints)
        count = sum([1 for i, _ in enumerate(unique_inds) if i not in dupes])
        # initialize array for overlap values
        num_bands = len(self.isosurface.nbands)
        overlap_vals = np.zeros((count,num_bands), dtype=float)
        # loop over bands
        for i, band in enumerate(self.isosurface.nbands):
            # generate symmetric coefficients for each band
            for j, wfk in enumerate(ir_wfk.SymWFKs(kpoint=kpt, band=band)):
                wfk = wfk.GridWFK()
                wfk = wfk.IFFT()
                wfk = wfk.Normalize()
                overlap = np.sum(np.conj(bandu.wfk_coeffs)*wfk.wfk_coeffs)
                overlap = np.square(np.abs(overlap))
                overlap_vals[j,i] = overlap
        return overlap_vals
    #-----------------------------------------------------------------------------------------------------------------#
    # method to compute bandu fxn and one electron wfk overlaps
    def _OverlapsNoSym(
        self, ir_wfk:wc.WFK, bandu:wc.WFK
    )->np.ndarray:
        # initialize array for overlap values
        num_bands = len(self.isosurface.nbands)
        overlap_vals = np.zeros((1,num_bands), dtype=float)
        # loop over bands
        for i, band in enumerate(self.isosurface.nbands):
            wfk = ir_wfk.GridWFK(band_index=band)
            wfk = wfk.IFFT()
            wfk = wfk.Normalize()
            overlap = np.sum(np.conj(bandu.wfk_coeffs)*wfk.wfk_coeffs)
            overlap = np.square(np.abs(overlap))
            overlap_vals[0,i] = overlap
        return overlap_vals
    #-----------------------------------------------------------------------------------------------------------------#
    # method to interpolate overlap values
    def _InterpolateOverlaps(
        self, contour:pv.PolyData, overlap_values:np.ndarray
    )->np.ndarray:
        # recreate eigenvalue grid as base for surface color grid
        color_grid = copy(self.isosurface.grid)
        # translate overlap values to 3x3x3 grid
        trans_color_points, trans_color_values = trslt.TranslatePoints(
            self.isosurface.points,
            overlap_values.reshape((-1,1)),
            self.isosurface.rec_latt
        )
        # construct PyVista object from translated overlap grid
        trans_color_points = pv.PolyData(trans_color_points)
        trans_color_points['values'] = trans_color_values
        # interpolate translated grid
        color_grid = color_grid.interpolate(
            trans_color_points,
            sharpness=2.0,
            radius=self.isosurface.radius,
            strategy='null_value'
        )
        # sample color values from interpolated color grid
        color_sample = contour.sample(color_grid)
        return color_sample.active_scalars
    #-----------------------------------------------------------------------------------------------------------------#
    # method to calculate surface color values from XSF and WFK
    def SurfaceColor(
        self, wfk_path:str, xsf_path:str, sym:bool=False
    )->list:
        '''
        Method for calculating BandU function overlap with states at a specified isoenergy.
        Requires BandU XSF file and ABINIT WFK file.

        Parameters
        ----------
        wfk_path : str
            Path to ABINIT WFK file
        xsf_path : str
            Path to BandU XSF file
        sym : bool
            Symmetrically generate full Brillouin Zone
            Default will not generate Brillouin Zone and instead symmetrize surface colors (False)
        '''
        from . import abinit_reader as ar
        from . import xsf_reader as xsfr
        # list of overlap values
        overlaps = np.zeros(1)
        # read fermi surface wavefunction
        fermi_wfk = ar.AbinitWFK(filename=wfk_path)   
        # get number of bands
        nbands = len(self.isosurface.nbands)
        # paths to real and imaginary bandu xsf files
        real_path = xsf_path + '_real.xsf'
        imag_path = xsf_path + '_imag.xsf'
        # read in xsf
        real_fxn = xsfr.XSF(xsf_file=real_path)
        imag_fxn = xsfr.XSF(xsf_file=imag_path)
        print('XSF read')
        # convert xsf to wfk object
        bandu_fxn = wc.WFK(
            wfk_coeffs=real_fxn.ReadGrid() + 1j*imag_fxn.ReadGrid(),
            ngfftx=real_fxn.ngfftx,
            ngffty=real_fxn.ngffty,
            ngfftz=real_fxn.ngfftz
        )
        # remove xsf format
        bandu_fxn = bandu_fxn.RemoveXSF()
        # loop through fermi surface kpoints and calc overlap with bandu fxn
        for i, kpt in enumerate(fermi_wfk.ReadWFK()):
            if sym:
                vals = self._OverlapsWithSym(kpt, bandu_fxn)
            else:
                vals = self._OverlapsNoSym(kpt, bandu_fxn)
            if i == 0:
                overlaps = vals
            else:
                overlaps = np.concatenate((overlaps, vals), axis=0)
        # symmetrically permute overlap values if they were only calculated on irreducible BZ wedge when sym==False
        if not sym:
            symrel = fermi_wfk.symrel
            nsym = fermi_wfk.nsym
            kpts = fermi_wfk.kpts
            all_kpts = np.zeros((1,3))
            all_overlaps = np.zeros((1,nbands))
            new_wfk = wc.WFK(symrel=np.array(symrel), nsym=nsym, nbands=nbands)
            for i, kpt in enumerate(kpts):
                unique_kpts, _ = new_wfk.Symmetrize(
                    points=kpt,
                    reciprocal=True
                )
                all_kpts = np.concatenate((all_kpts, unique_kpts), axis=0)
                new_overlaps = np.repeat(overlaps[i,:].reshape((1,-1)), unique_kpts.shape[0], axis=0)
                all_overlaps = np.concatenate((all_overlaps, new_overlaps), axis=0)
            kpts = np.delete(all_kpts, 0, axis=0)
            overlaps = np.delete(all_overlaps, 0, axis=0)
            self.isosurface.points = np.matmul(kpts, self.isosurface.rec_latt)
        # interpolate overlap values for smooth coloration
        scalars = []
        for i in range(overlaps.shape[1]):
            interp_vals = self._InterpolateOverlaps(self.isosurface.contours[i], overlaps[:,i])
            scalars.append(interp_vals)
        return scalars
    #-----------------------------------------------------------------------------------------------------------------#
    # method to load save file
    def Load(
        self, save_path:str|list, **kwargs
    ):
        '''
        Method for loading saved contours 

        Parameters
        ----------
        colormap : str | ListedColormap
            Choose colormap for isosurface that colors according to assigned scalars of surface\n
            Can use Colors class to use default or create custom colormaps\n
            Default is matplotlib's plasma
        bz_width : int
            Line width of Brillouin Zone\n
            Default is 3
        bz_show : bool
            Show the Brillouin Zone
            Default is to show (True)
        smooth : bool
            Use smooth lightning techniques\n
            Default is to use smoothing (True)
        lighting : bool
            Apply directional lighting to surface\n
            Default is to enable directional lighting (True)
        ambient : float
            Intensity of light on surface\n
            Default is 0.5
        diffuse : float
            Amount of light scattering\n
            Default is 0.5
        specular : float
            Amount of reflected light\n
            Default is 1.0 (max)
        specular_power : float
            Determines how sharply light is reflected\n
            Default is 128.0 (max)
        pbr : bool
            Apply physics based rendering\n
            Default is no physics based rendering (False)
        metallic : float
            Determine how metallic-looking the surface is, only considered with pbr\n
            Default is 0.5
        roughness : float
            Determine how smooth/rough surface appear, only considered with pbr\n
            Default is 0.5
        color : str
            Sets color of surface (colormap overwrites this)\n
            Sets color of reflected light (colormap does not overwrite this)\n
            Default is white
        arrow : list
            Parameters for plotting nesting vector on top of Fermi surface\n
            Element_0 of list should be starting (or tail) position of arrow\n
            Element_1 of list should be orientation of arrow with desired magnitude\n
            Both the tail and orientation should be specified in reduced reciprocal space coordinates
        arrow_color : str
            Color of nesting arrow\n
            Default is black
        show_endpoints : bool
            Plot points on the end of the arrow to make visualizing start and end easier\n
            Default is to not show endpoints (False)
        periodic_arrow : list
            Adds periodic image of arrow that is translated [X,Y,Z] cells\n
            Where X, Y, and Z are the cell indices
        show_bands : float | list
            Specifies the opacity of each band\n
            If a single float is provided, all bands will be plotted with the same opacity\n
            If a list is provided, each band will be plotted with the opacity of the respective list element
        show_axes : bool
            Plots reciprocal cell axes with a* as red, b* as green, and c* as blue\n
            Default is to show axes (True)
        cross_section : list
            Plot cross section through surface\n
            If one vector is provided, it is assumed to be the normal to the cross section plane\n
            Else, cross section is defined by plane made by two vectors\n
            Vectors should be specified in reduced coordinates
        cross_width : float
            Width of cross section\n
            Default is 0.15
        linear : bool
            Cross section linearly fades out\n
            Default is not fade out linearly (False)
        two_dim : bool
            Cross section is a 2D slice instead of a section\n
            Default is to show cross section as 2D slice (True)
        surface_vals : np.ndarray
            A list of values defining the coloration of the isosurface
            Default plots no coloration
        '''
        # functionality for adding multiple projections onto one surface
        if type(save_path) == list:
            all_scalars = []
            for save in save_path:
                with open(save, 'rb') as f:
                    kwargs_dict:dict = pkl.load(f)
                    self.isosurface:ic.Isosurface = pkl.load(f)
                all_scalars.append(kwargs_dict['surface_vals'])
            sizes = []
            for surface_vals in all_scalars[0]:
                sizes.append(surface_vals.shape)
            new_scalars = [np.zeros((size)) for size in sizes]
            for scalars in all_scalars:
                for i, surface_vals in enumerate(scalars):
                    new_scalars[i] += surface_vals
            kwargs_dict['surface_vals'] = new_scalars # type: ignore
        # replot a single surface
        else:
            with open(save_path, 'rb') as f: # type: ignore
                kwargs_dict:dict = pkl.load(f)
                self.isosurface:ic.Isosurface = pkl.load(f)
        for k, val in kwargs.items(): # type: ignore
            kwargs_dict[k] = val # type: ignore       
        self.save = False
        self.plot = True
        self.Plot(**kwargs_dict) # type: ignore