import pickle as pkl
import pyvista as pv
import numpy as np
from isosurface import Isosurface
from typing import Union

class AnalyzeSurface(Isosurface):
    def __init__(
            self, depth_peeling=True
    )->None:
        super().__init__(depth_peeling=depth_peeling)
    #-----------------------------------------------------------------------------------------------------------------#
    # method to add nesting vector to isosurface plot
    def _AddArrow(
            self, arrow:list, rec_lattice:np.ndarray, show_endpoints:bool, color:str
    )->None:
        tail = np.array(arrow[0])
        shift = np.array(arrow[1])
        tail = np.matmul(tail, rec_lattice)
        shift = np.matmul(shift, rec_lattice)
        scale = np.linalg.norm(shift)
        py_arrow = pv.Arrow(start=tail,
                            direction=shift,
                            tip_radius=0.05/scale,
                            tip_length=0.15/scale,
                            shaft_radius=0.025/scale,
                            scale=scale
        )
        if show_endpoints:
            points = np.array([tail, shift+tail])
            points = pv.PolyData(points)
            self.p.add_mesh(points.points, point_size=10, color='red')
        self.p.add_mesh(py_arrow, color=color)
    #-----------------------------------------------------------------------------------------------------------------#
    # method to add reciprocal axes
    def _AddAxes(
            self, axes:np.ndarray
    )->None:
        axes_colors = ['red', 'green', 'blue']
        for i in range(3):
            axis = axes[i,:]
            color = axes_colors[i]
            self._AddArrow([[0,0,0], axis], np.identity(3), False, color)
    #-----------------------------------------------------------------------------------------------------------------#
    # method to add periodic image of nesting vector
    def _PeriodicArrow(
            self, arrow:list, cell:list, rec_lattice:np.ndarray, show_endpoints:bool
    )->None:
        tail = np.array(arrow[0])
        cell = np.array(cell, dtype=float)
        tail += cell
        arrow[0] = tail
        self._AddArrow(arrow, rec_lattice, show_endpoints, color='gray')
    #-----------------------------------------------------------------------------------------------------------------#
    # method to visualize cross-section of surface
    def _CrossSection(
            self, vecs:list, points:np.ndarray, width:float, rec_lattice:np.ndarray, bz_points:np.ndarray,
            linear:bool=False, two_dim:bool=False
    )->np.ndarray:
        from scipy.spatial import Delaunay
        if len(vecs) == 2:
            vec1 = np.matmul(vecs[0], rec_lattice)
            vec2 = np.matmul(vecs[1], rec_lattice)
            norm = np.cross(vec1, vec2)
            norm /= np.linalg.norm(norm)
        else:
            vec = np.matmul(vecs, rec_lattice)
            norm = vec/np.linalg.norm(vec)
        norm_points = np.array(points/np.linalg.norm(points, axis=1).reshape((len(points),1)))
        angs = np.matmul(norm, norm_points.T)
        if linear:
            angs[angs <= width] = 2
            angs -= 1
            angs = np.abs(angs)
        else:
            if two_dim:
                angs = np.abs(angs)
            opacities = np.zeros(len(points))
            opacities[angs <= width] = 1
            angs = opacities  
        beyond_bz = Delaunay(bz_points).find_simplex(points)
        angs[beyond_bz < 0] = 0
        return angs
    #-----------------------------------------------------------------------------------------------------------------#
    # method to load previously calculated surface
    def Analysis(
            self, colormap:str='plasma', BZ_width:float=2.5, smooth:bool=True, lighting:bool=True, 
            ambient:float=0.5, diffuse:float=0.5, specular:float=1.0, specular_power:float=128.0, pbr:bool=False, 
            metallic:float=0.5, roughness:float=0.5, color:str='white', file_name:str='Fermi_surface.pkl',
            arrow:list=[], arrow_color:str='black', show_endpoints:bool=False, periodic:list=[], 
            opacity:Union[float,list]=1.0, add_axes:bool=False, cross_section:list=[], cross_width:float=0.15,
            linear:bool=False, two_dim:bool=False, camera_position:list=[]
    )->None:
        '''
        Method for loading saved isosurface pickle and changing colors/adding nesting vectors

        Parameters
        ----------
        colormap : str
            Choose colormap for isosurface that colors according to assigned scalars of surface\n
            Can use Colors class to use default or create custom colormaps\n
            Default is matplotlib's plasma
        BZ_width : float
            Line width of Brillouin Zone\n
            Default is 2.5
        smooth : bool
            Use smooth lightning techniques\n
            Default is True
        lighting : bool
            Apply directional lighting to surface\n
            Default is True
        ambient : float
            Intensity of light on surface\n
            Default is 0.5
        diffuse : float
            Amount of light scattering\n
            Default is 0.5
        specular : float
            Amount of reflected light\n
            Default is 1.0
        specular_power : float
            Determines how sharply light is reflected\n
            Default is 128.0 (max)
        pbr : bool
            Apply physics based rendering\n
            Default is False
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
        file_name : str
            Name of isosurface pickle file\n
            Default is Fermi_surface.pkl
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
            Default is false
        periodic : list
            Adds periodic image of arrow that is translated [X,Y,Z] cells\n
            Where X, Y, and Z are the cell indices
        opacity : float | list
            Specifies the opacity of each band\n
            If a single float is provided, all bands will be plotted with the same opacity\n
            If a list is provided, each band will be plotted with the opacity of the respective list element
        add_axes : bool
            Plots reciprocal cell axes with a* as red, b* as green, and c* as blue\n
            Default is false
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
            Default is False
        two_dim : bool
            Cross section is a 2D slice instead of a section\n
            Default is False
        '''
        with open(file_name, 'rb') as f:
            bands = pkl.load(f)
            if type(opacity) == float:
                opacity = np.ones(len(bands))*opacity            
            bz_points = pkl.load(f)
            rec_lattice = pkl.load(f)
            self.p.add_lines(bz_points, color='black', width=BZ_width)
            for i, _ in enumerate(bands):
                iso_surf = pkl.load(f)
                opacities = pkl.load(f)
                scalars = pkl.load(f)
                if cross_section != []:
                    opacities = self._CrossSection(cross_section, 
                                                   iso_surf.points, 
                                                   cross_width, 
                                                   rec_lattice, 
                                                   bz_points,
                                                   linear=linear,
                                                   two_dim=two_dim
                    )
                opacities = [opacity[i]*op for op in opacities]
                self.p.add_mesh(iso_surf, 
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
                                color=color
                )
        if arrow != []:
            self._AddArrow(arrow, rec_lattice, show_endpoints, arrow_color)
        if periodic != []:
            self._PeriodicArrow(arrow, periodic, rec_lattice, show_endpoints)
        if add_axes:
            self._AddAxes(rec_lattice)
        if camera_position != []:
            camera_position = np.array(camera_position).reshape((3,3))
            camera_position = np.matmul(camera_position, rec_lattice)
            self.p.camera_position = camera_position
        self._Render()