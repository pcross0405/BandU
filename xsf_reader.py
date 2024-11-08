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
    def __init__(self, xsf_file):
        self.xsf_file = xsf_file
        with open(xsf_file, 'r') as xsf:
            self.xsf_lines = xsf.readlines()
        self.lattice = np.zeros((3,3))
        for i, line in enumerate(self.xsf_lines):
            if line.strip() == 'PRIMVEC':
                self.lattice[0,:] = [float(val) for val in self.xsf_lines[i+1].strip().split(' ')]
                self.lattice[1,:] = [float(val) for val in self.xsf_lines[i+2].strip().split(' ')]
                self.lattice[2,:] = [float(val) for val in self.xsf_lines[i+3].strip().split(' ')]
            if line.strip() == 'PRIMCOORD':
                self.natoms = int(self.xsf_lines[i+1].strip().split(' ')[0])
                self.coords = np.zeros((self.natoms, 3))
                self.elements = []
                for atom in range(self.natoms):
                    coord = self.xsf_lines[i+atom+1].strip().split(' ')
                    coord = [float(val) for val in coord]
                    element = atom_labels[int(coord[0])]
                    self.elements.append(element)
                    del coord[0]
                    self.coords[atom,:] = coord
            if line.strip() == 'DATAGRID_3D_DENSITY':
                ngfft_spacing = self.xsf_lines[i+1].strip().split(' ')
                ngfft_spacing = [int(val) for val in ngfft_spacing]
                self.ngfftx = ngfft_spacing[0]
                self.ngffty = ngfft_spacing[1]
                self.ngfftz = ngfft_spacing[2]
                break
    #-----------------------------------------------------------------------------------------------------------------#
    # method removing XSF formatting from density grid
    def _RemoveXSF(self, grid:np.ndarray)->np.ndarray:
        to_be_del = np.ones((self.ngfftz, self.ngfftx, self.ngffty), dtype=bool)
        for z in range(self.ngfftz):
            for x in range(self.ngfftx):
                for y in range(self.ngffty):
                    if y == self.ngffty - 1 or x == self.ngfftx - 1 or z == self.ngfftz - 1:
                        to_be_del[z,x,y] = False
        grid = grid[to_be_del]
        return grid
    #-----------------------------------------------------------------------------------------------------------------#
    # method reading in BandU eigenfunction from XSF
    def _ReadDensity(self)->np.ndarray:
        for i, line in enumerate(self.xsf_lines):
            if line.strip() == 'DATAGRID_3D_DENSITY':
                density_lines = self.xsf_lines[i+6:]
                del density_lines[-1]
                last_line = density_lines[-1].strip().split(' ')
                last_line = [val for val in last_line if val != 'END_DATAGRID_3D']
                last_line = ' '.join(last_line)
                density_lines[-1] = last_line
        density_lines = [line.strip().split(' ') for line in density_lines]
        density_lines = [val for line in density_lines for val in line]
        density_lines = np.array(density_lines, dtype=float)
        density_lines = density_lines.reshape((self.ngfftz, self.ngfftx, self.ngffty))
        return density_lines
    
if __name__ == '__main__':
    xsf = XSF('RbV3Sb5_eig_1_real.xsf')
    grid = xsf._ReadDensity()
    grid = xsf._RemoveXSF(grid)