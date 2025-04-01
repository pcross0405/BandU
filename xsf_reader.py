import numpy as np

atom_labels = {1:'H', 2:'He', 3:'Li', 4:'Be', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 10:'Ne', 11:'Na', 12:'Mg', 13:'Al', 
               14:'Si', 15:'P', 16:'S', 17:'Cl', 18:'Ar', 19:'K', 20:'Ca', 21:'Sc', 22:'Ti', 23:'V', 24:'Cr', 25:'Mn',
               26:'Fe', 27:'Co', 28:'Ni', 29:'Cu', 30:'Zn', 31:'Ga', 32:'Ge', 33:'As', 34:'Se', 35:'Br', 36:'Kr', 
               37:'Rb', 38:'Sr', 39:'Y', 40:'Zr', 41:'Nb', 42:'Mo', 43:'Tc', 44:'Ru', 45:'Rh', 46:'Pd', 47:'Ag', 
               48:'Cd', 49:'In', 50:'Sn', 51:'Sb', 52:'Te', 53:'I', 54:'Xe', 55:'Cs', 56:'Ba', 57:'La', 58:'Ce', 
               59:'Pr', 60:'Nd', 61:'Pm', 62:'Sm', 63:'Eu', 64:'Gd', 65:'Tb', 66:'Dy', 67:'Ho', 68:'Er', 69:'Tm', 
               70:'Yb', 71:'Lu', 72:'Hf', 73:'Ta', 74:'W', 75:'Re', 76:'Os', 77:'Ir', 78:'Pt', 79:'Au', 80:'Hg', 
               81:'Tl', 82:'Pb', 83:'Bi', 84:'Po', 85:'At', 86:'Rn', 87:'Fr', 88:'Ra', 89:'Ac', 90:'Th', 91:'Pa', 
               92:'U'}

class XSF():
    def __init__(
            self, 
            xsf_file:str=None
        )->None:
    # if xsf file is supplied, read in parameters
        self.xsf_file = xsf_file
        with open(xsf_file, 'r') as xsf:
            self.xsf_lines = xsf.readlines()
        self.lattice = np.zeros((3,3))
        for i, line in enumerate(self.xsf_lines):
            # get lattice vectors from XSF file
            if line.strip() == 'PRIMVEC':
                self.lattice[0,:] = [float(val) for val in self.xsf_lines[i+1].strip().split(' ')]
                self.lattice[1,:] = [float(val) for val in self.xsf_lines[i+2].strip().split(' ')]
                self.lattice[2,:] = [float(val) for val in self.xsf_lines[i+3].strip().split(' ')]
            # get number of atoms and atomic symbols
            # get atomic coordinates from XSF file
            if line.strip() == 'PRIMCOORD':
                self.natoms = int(self.xsf_lines[i+1].strip().split(' ')[0])
                self.coords = np.zeros((self.natoms, 3))
                self.elements = []
                for atom in range(self.natoms):
                    coord = self.xsf_lines[i+atom+2].strip().split(' ')
                    coord = [float(val) for val in coord]
                    element = atom_labels[int(coord[0])]
                    self.elements.append(element)
                    del coord[0]
                    self.coords[atom,:] = coord
            # once density block is reached, get ngfft spacing and end init
            if line.strip() == 'DATAGRID_3D_DENSITY':
                ngfft_spacing = self.xsf_lines[i+1].strip().split(' ')
                ngfft_spacing = [int(val) for val in ngfft_spacing]
                self.ngfftx = ngfft_spacing[0]
                self.ngffty = ngfft_spacing[1]
                self.ngfftz = ngfft_spacing[2]
                return None
    #-----------------------------------------------------------------------------------------------------------------#
    # method reading in BandU eigenfunction from XSF
    def ReadDensity(
        self
    )->np.ndarray:
        '''
        Method for reading in density grid from XSF file. Returns grid as N dimensional numpy array.

        Parameters
        ----------
        undo_xsf : bool
            If True (which is the default), then the XSF formatting is undone by removing end points.
        '''
        for i, line in enumerate(self.xsf_lines):
            # get density block, this assumes density is the end most data grid in the XSF
            if line.strip() == 'DATAGRID_3D_DENSITY':
                # density starts 6 lines down from DATAGRID_3D_DENSITY header
                density_lines = self.xsf_lines[i+6:]
                # last line indicates end of data block, remove it
                del density_lines[-1]
                # last entry in density is a string indicating end of density, remove it
                last_line = density_lines[-1].strip().split(' ')
                last_line = [val for val in last_line if val != 'END_DATAGRID_3D']
                # cast line back to a single string
                last_line = ' '.join(last_line)
                density_lines[-1] = last_line
        # convert density to 3D array of floats
        density_lines = [line.strip().split(' ') for line in density_lines]
        density_lines = [val for line in density_lines for val in line]
        density_lines = np.array(density_lines, dtype=float)
        density_lines = density_lines.reshape((self.ngfftz, self.ngfftx, self.ngffty))
        return density_lines