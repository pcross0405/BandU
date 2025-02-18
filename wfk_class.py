import numpy as np
from scipy.fft import fftn, ifftn
import sys
from typing import Self
from copy import copy
np.set_printoptions(threshold=sys.maxsize)

class WFK():
    '''
    A class manipulating wavefunctions from DFT calculations

    Parameters
    ----------
    wfk_coeffs : np.ndarray
        The planewave coefficients of the wavefunction
        These should be complex values
    kpoints : np.ndarray
        A multidimensional array of 3D kpoints
        Entries along axis 0 should be individual kpoints
        Entries along axis 1 should be the kx, ky, and kz components, in that order
        The kpoints should be in reduced form
    syrmel : np.ndarray
        A multidimensional array of 3x3 arrays of symmetry operations
    nsym : int
        Total number of symmetry operations
    nkpt : int
        Total number of kpoints
        If a kpoints array is provided, then nkpt will be acquired the its length
    nbands : int
        Total number of bands
    ngfftx : int
        x dimension of Fourier transform grid
    ngffty : int
        y dimension of Fourier transform grid
    ngfftz : int
        z dimension of Fourier transform grid
    eigenvalues : list
        List of the eigenvalues for wavefunction at each band
        Should be ordered from least -> greatest
    fermi_energy : float
        Fermi energy 
    lattice : np.ndarray
        3x3 array containing lattice parameters
    natom : int
        Total number of atoms in unit cell
    xred : np.ndarray
        Reduced coordinates of all atoms in unit cell
        Individual atomic coordinates fill along axis 0
        X, Y, and Z components fill along axis 1, in that order
    typat : list
        Numeric labels starting from 1 and incrementing up to natom
        Order of labels should follow xred
    znucltypat : list
        List of element names
        First element of list should correspond to typat label 1, second element to label 2 and so on
    rec_latt_pts : np.ndarray
        Array of reciprocal lattice points.
        Necessary for arranging wavefunction coefficients in 3D array.
    '''
    def __init__(
        self, 
        wfk_coeffs:np.ndarray=None, kpoints:np.ndarray=None, symrel:np.ndarray=None, nsym:int=None, nkpt:int=None, 
        nbands:int=None, ngfftx:int=None, ngffty:int=None, ngfftz:int=None, eigenvalues:list=None, 
        fermi_energy:float=None, lattice:np.ndarray=None, natom:int=None, xred:np.ndarray=None, typat:list=None,
        znucltypat:list=None, rec_latt_pts:np.ndarray=None
    )->None:
        self.wfk_coeffs=wfk_coeffs
        self.kpoints=kpoints
        self.rec_latt_pts=rec_latt_pts,
        self.symrel=symrel
        self.nsym=nsym
        self.nkpt=nkpt
        self.nbands=nbands
        self.ngfftx=ngfftx
        self.ngffty=ngffty
        self.ngfftz=ngfftz
        self.eigenvalues=eigenvalues
        self.fermi_energy=fermi_energy
        self.lattice=lattice
        self.natom=natom
        self.xred=xred
        self.typat=typat,
        self.znucltypat=znucltypat
#---------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ METHODS ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#
    # method transforming reciprocal space wfks to real space
    def GridWFK(
            self, band_index:int=None
    )->Self:
        '''
        Returns copy of WFK object with coefficients in numpy 3D array grid.
        Grid is organized in (ngfftz, ngfftx, ngffty) dimensions.
        Where ngfft_ represents the _ Fourier transform grid dimension.

        Parameters
        ----------
        band_index : int
            Integer represent the band index of the wavefunction coefficients to be transformed.
            If nothing is passed, it is assumed the coefficients of a single band are supplied.
        '''
        # initialize 3D grid
        gridded_wfk = np.zeros((self.ngfftz, self.ngfftx, self.ngffty), dtype=complex)
        # update grid with wfk coefficients
        for k, kpt in enumerate(self.rec_latt_pts):
            kx = kpt[0]
            ky = kpt[1]
            kz = kpt[2]
            if band_index != None:
                gridded_wfk[kz, kx, ky] = self.wfk_coeffs[band_index][k]
            else:
                gridded_wfk[kz, kx, ky] = self.wfk_coeffs[k]
        new_WFK = copy(self)
        new_WFK.wfk_coeffs = gridded_wfk
        return new_WFK
    #-----------------------------------------------------------------------------------------------------------------#
    # method transforming reciprocal space wfks to real space
    def FFT(
            self
    )->Self:
        '''
        Returns copy of WFK with wavefunction coefficients in expressed in real space.
        Assumes existing wavefunction coefficients are expressed in reciprocal space. 
        '''
        # Fourier transform reciprocal grid to real space grid
        real_coeffs = fftn(self.wfk_coeffs, norm='ortho')
        new_WFK = copy(self)
        new_WFK.wfk_coeffs = np.array(real_coeffs).reshape((self.ngfftz, self.ngfftx, self.ngffty))
        return new_WFK
    #-----------------------------------------------------------------------------------------------------------------#
    # method transforming real space wfks to reciprocal space
    def IFFT(
            self
    )->Self:
        '''
        Returns copy of WFK with wavefunction coefficients in expressed in reciprocal space.
        Assumes existing wavefunction coefficients are expressed in real space. 
        '''
        # Fourier transform real space grid to reciprocal space grid
        reciprocal_coeffs = ifftn(self.wfk_coeffs, norm='ortho')
        new_WFK = copy(self)
        new_WFK.wfk_coeffs = reciprocal_coeffs
        return new_WFK
    #-----------------------------------------------------------------------------------------------------------------#
    # method for normalizing wfks
    def Normalize(
            self
    )->Self:
        '''
        Returns copy of WFK object with normalized wavefunction coefficients such that <psi|psi> = 1.
        '''
        coeffs = np.array(self.wfk_coeffs)
        # calculate normalization constant and apply to wfk
        norm = np.dot(coeffs.flatten(), np.conj(coeffs).flatten())
        norm = np.sqrt(norm)
        new_WFK = copy(self)
        new_WFK.wfk_coeffs /= norm
        return new_WFK
    #-----------------------------------------------------------------------------------------------------------------#
    # method for converting real space lattice vectors to reciprocal space vectors
    def Real2Reciprocal(
        self
    )->np.ndarray:
        '''
        Method for converting the real space lattice parameters to reciprocal lattice parameters

        Parameters
        ----------
        real_lat : np.ndarray
            3x3 numpy array containing real space lattice parameters.
        '''
        # conversion by default converts Angstrom to Bohr since ABINIT uses Bohr
        a = self.lattice[0,:]
        b = self.lattice[1,:]
        c = self.lattice[2,:]
        vol = np.dot(a,np.cross(b,c))
        b1 = 2*np.pi*(np.cross(b,c))/vol
        b2 = 2*np.pi*(np.cross(c,a))/vol
        b3 = 2*np.pi*(np.cross(a,b))/vol
        return np.array([b1,b2,b3]).reshape((3,3))
    #-----------------------------------------------------------------------------------------------------------------#
    # method for creating symmetrically equivalent points
    def Symmetrize(
            self, points:np.ndarray=None, values:np.ndarray=None
    )->tuple[np.ndarray, np.ndarray]:
        '''
        Method for generating full reciprocal space cell from irreducible kpoints\n
        *REQUIRES KPOINTS TO BE IN REDUCED FORMAT, CARTESIAN FORMAT WILL NOT WORK*

        Parameters
        ----------
        points : np.ndarray
            Irreducible set of kpoints
        values : np.ndarray
            Eigenvalues for irreducible kpoints
        '''
        # get symmetry operations and initialize symmetrically equivalent point and value arrays
        sym_ops = self.symrel
        ind_len = self.nkpt
        sym_pts = np.zeros((self.nsym*ind_len,3))
        sym_vals = np.zeros((self.nsym*ind_len,self.nbands))
        for i, op in enumerate(sym_ops):
            new_pts = np.matmul(points, op)
            sym_pts[i*ind_len:(i+1)*ind_len,:] = new_pts
            sym_vals[i*ind_len:(i+1)*ind_len,:] = values
        # points overlap on at edges of each symmetric block, remove duplicates
        sym_pts, unique_inds = np.unique(sym_pts, return_index=True, axis=0)
        sym_vals = np.take(sym_vals, unique_inds, axis=0)
        return sym_pts, sym_vals
    #-----------------------------------------------------------------------------------------------------------------#
    # method for expanding a grid into XSF format
    def XSFFormat(
            self
    )->Self:
        '''
        Returns copy of WFK object XSF formatted coefficients.
        Requires wfk_coeffs to be in gridded format, i.e. (ngfftz, ngfftx, ngffty) shape.
        '''
        # append zeros to ends of all axes in grid_wfk
        # zeros get replaced by values at beginning of each axis
        # this is repetition is required by XSF format
        if np.shape(self.wfk_coeffs) != (self.ngfftz, self.ngfftx, self.ngffty):
            raise ValueError(
                f'''Passed array is not the correct shape:
                Expected: ({self.ngfftz}, {self.ngfftx}, {self.ngffty}),
                Received: {np.shape(self.wfk_coeffs)}
            ''')
        else:
            grid_wfk = self.wfk_coeffs
        grid_wfk = np.append(grid_wfk, np.zeros((1, self.ngfftx, self.ngffty)), axis=0)
        grid_wfk = np.append(grid_wfk, np.zeros((self.ngfftz+1, 1, self.ngffty)), axis=1)
        grid_wfk = np.append(grid_wfk, np.zeros((self.ngfftz+1, self.ngfftx+1, 1)), axis=2)
        for x in range(self.ngfftx+1):
            for y in range(self.ngffty+1):
                for z in range(self.ngfftz+1):
                    if x == self.ngfftx:
                        grid_wfk[z][x][y] = grid_wfk[z][0][y]
                    if y == self.ngffty:
                        grid_wfk[z][x][y] = grid_wfk[z][x][0]
                    if z == self.ngfftz:
                        grid_wfk[z][x][y] = grid_wfk[0][x][y]
                    if x == self.ngfftx and y == self.ngffty:
                        grid_wfk[z][x][y] = grid_wfk[z][0][0]
                    if x == self.ngfftx and z == self.ngfftz:
                        grid_wfk[z][x][y] = grid_wfk[0][0][y]
                    if z == self.ngfftz and y == self.ngffty:
                        grid_wfk[z][x][y] = grid_wfk[0][x][0]
                    if x == self.ngfftx and y == self.ngffty and z == self.ngfftz:
                        grid_wfk[z][x][y] = grid_wfk[0][0][0]
        new_WFK = copy(self)
        new_WFK.wfk_coeffs = grid_wfk
        return new_WFK
    #-----------------------------------------------------------------------------------------------------------------#
    # method removing XSF formatting from density grid
    def RemoveXSF(
        self
    )->Self:
        '''
        Returns copy of WFK object without XSF formatting.
        '''
        grid = self.wfk_coeffs
        # to_be_del will be used to remove all extra data points added for XSF formatting
        to_be_del = np.ones((self.ngfftz, self.ngfftx, self.ngffty), dtype=bool)
        for z in range(self.ngfftz):
            for x in range(self.ngfftx):
                for y in range(self.ngffty):
                    # any time you reach the last density point it is a repeat of the first point
                    # remove the end points along each axis
                    if y == self.ngffty - 1 or x == self.ngfftx - 1 or z == self.ngfftz - 1:
                        to_be_del[z,x,y] = False
        # remove xsf entries from array
        grid = grid[to_be_del]
        # restore grid shape
        grid = grid.reshape((self.ngfftz-1, self.ngfftx-1, self.ngffty-1))
        new_WFK = copy(self)
        new_WFK.wfk_coeffs = grid
        return new_WFK
    #-----------------------------------------------------------------------------------------------------------------#
    # method for writing wavefunctions to XSF file
    def WriteXSF(
            self, xsf_file:str, _component:bool=True
    )->None:
        '''
        A method for writing numpy grids to an XSF formatted file

        Parameters
        ----------
        xsf_file : str
            The file name
        '''
        # check if typat is packed as tuple or not
        if type(self.typat) == tuple:
            self.typat = self.typat[0]
        # first run writes out real part of eigenfunction to xsf
        if _component:
            xsf_file += '_real.xsf'
        # second run writes out imaginary part
        else:
            xsf_file += '_imag.xsf'
        with open(xsf_file, 'w') as xsf:
            print('DIM-GROUP', file=xsf)
            print('3 1', file=xsf)
            print('PRIMVEC', file=xsf)
            print(f'{self.lattice[0,0]} {self.lattice[0,1]} {self.lattice[0,2]}', file=xsf)
            print(f'{self.lattice[1,0]} {self.lattice[1,1]} {self.lattice[1,2]}', file=xsf)
            print(f'{self.lattice[2,0]} {self.lattice[2,1]} {self.lattice[2,2]}', file=xsf)
            print('PRIMCOORD', file=xsf)
            print(f'{self.natom} 1', file=xsf)
            for i, coord in enumerate(self.xred):
                atomic_num = int(self.znucltypat[self.typat[i] - 1])
                cart_coord = np.dot(coord, self.lattice)
                print(f'{atomic_num} {cart_coord[0]} {cart_coord[1]} {cart_coord[2]}', file=xsf)
            print('ATOMS', file=xsf)
            for i, coord in enumerate(self.xred):
                atomic_num = int(self.znucltypat[self.typat[i] - 1])
                cart_coord = np.dot(coord, self.lattice)
                print(f'{atomic_num} {cart_coord[0]} {cart_coord[1]} {cart_coord[2]}', file=xsf)
            print('BEGIN_BLOCK_DATAGRID3D', file=xsf)
            print('datagrids', file=xsf)
            print('DATAGRID_3D_DENSITY', file=xsf)
            print(f'{self.ngfftx+1} {self.ngffty+1} {self.ngfftz+1}', file=xsf)
            print('0.0 0.0 0.0', file=xsf)
            print(f'{self.lattice[0,0]} {self.lattice[0,1]} {self.lattice[0,2]}', file=xsf)
            print(f'{self.lattice[1,0]} {self.lattice[1,1]} {self.lattice[1,2]}', file=xsf)
            print(f'{self.lattice[2,0]} {self.lattice[2,1]} {self.lattice[2,2]}', file=xsf)
            count = 0
            for z in range(self.ngfftz+1):
                for x in range(self.ngfftx+1):
                    for y in range(self.ngffty+1):
                        count += 1
                        if _component:
                            print(self.wfk_coeffs[z,x,y].real, file=xsf, end=' ')
                        else:
                            print(self.wfk_coeffs[z,x,y].imag, file=xsf, end=' ')
                        if count == 6:
                            count = 0
                            print('\n', file=xsf, end='')
            print('END_DATAGRID_3D', file=xsf)
            print('END_BLOCK_DATAGRID3D', file=xsf)
        # rerun method to write out imaginary part
        if _component:
            xsf_file = xsf_file.split('_real')[0]
            self.WriteXSF(xsf_file, _component=False)