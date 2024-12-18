import pickle as pkl
import pyvista as pv
import numpy as np
from isosurface import Isosurface
from typing import Union

class LoadIsosurface(Isosurface):
    def __init__(self, depth_peeling=True):
        super().__init__(depth_peeling=depth_peeling)
    #-----------------------------------------------------------------------------------------------------------------#
    # method to add nesting vector to isosurface plot
    def _AddArrow(self, arrow:list, rec_lattice:np.ndarray, color:str='black')->None:
        if arrow != []:
            tail = np.array(arrow[0])
            shift = np.array(arrow[1])
            tail = np.matmul(tail, rec_lattice)
            shift = np.matmul(shift, rec_lattice)
            scale = np.linalg.norm(shift)*(0.88)
            py_arrow = pv.Arrow(start=tail,
                                direction=shift,
                                tip_radius=0.05/scale,
                                tip_length=0.15,
                                shaft_radius=0.025/scale,
                                scale=scale
            )
        self.p.add_mesh(py_arrow, color=color)
    #-----------------------------------------------------------------------------------------------------------------#
    # method to load previously calculated surface
    def LoadFile(self, colormap:str='plasma', BZ_width:float=2.5, smooth:bool=True, lighting:bool=True, 
                    ambient:float=0.5, diffuse:float=0.5, specular:float=1.0, specular_power:float=128.0, pbr:bool=False, 
                    metallic:float=0.5, roughness:float=0.5, color:str='white', file_name:str='Fermi_surface.pkl',
                    arrow:list=[], arrow_color:str='black', opacity:Union[float,list]=1.0
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
        opacity : float | list
            Specifies the opacity of each band\n
            If a single float is provided, all bands will be plotted with the same opacity\n
            If a list is provided, each band will be plotted with the opacity of the respective list element
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
                opacities = [opacity[i]*op for op in opacities]
                scalars = pkl.load(f)
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
            self._AddArrow(arrow, rec_lattice, arrow_color)
        self._Render()