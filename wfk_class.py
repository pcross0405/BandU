import numpy as np
from scipy.fft import fftn, ifftn
import sys
from typing import Self, Generator
from copy import copy
from brillouin_zone import BZ
np.set_printoptions(threshold=sys.maxsize)

class WFK():
    '''
    A class for working with wavefunctions from DFT calculations

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
    pw_indices : np.ndarray
        Array of H, K, L indices for the planwave basis set.
        Arrays of (1,3) [H,K,L] should fill axis 0, and H, K, L values fill axis 1 in that order.
        Necessary for arranging wavefunction coefficients in 3D array.
    syrmel : np.ndarray
        A multidimensional array of 3x3 arrays of symmetry operations
    non_symm_vec : np.ndarray
        A multidimensional array of 1x3 arrays of the nonsymmorphic translation vectors for each-
        symmetry operation.
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
    time_reversal : bool
        Select whether system has time reversal symmetry or not
        If the system is time reversal symmetric, then reciprocal space electronic states will share inversion 
        symmetry even if the real space symmetries do not include inversion
        Default assumes noncentrosymmetric systems have time reversal symmetry (True)
    '''
    def __init__(
        self, 
        wfk_coeffs:np.ndarray=np.zeros(1), kpoints:np.ndarray=np.zeros(1), symrel:np.ndarray=np.zeros(1), 
        nsym:int=0, nkpt:int=0, nbands:int=0, ngfftx:int=0, ngffty:int=0, ngfftz:int=0, 
        eigenvalues:np.ndarray=np.zeros(1),fermi_energy:float=0.0, lattice:np.ndarray=np.zeros(1), natom:int=0, 
        xred:np.ndarray=np.zeros(1), typat:list=[], znucltypat:list=[], pw_indices:np.ndarray=np.zeros(1), 
        non_symm_vecs:np.ndarray=np.zeros(1), time_reversal:bool=True
    )->None:
        self.wfk_coeffs=wfk_coeffs
        self.kpoints=kpoints
        self.pw_indices=pw_indices
        self.symrel=symrel
        self.nsym=nsym
        self.non_symm_vecs=non_symm_vecs
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
        self.typat=typat
        self.znucltypat=znucltypat
        self.time_reversal=time_reversal
#---------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ METHODS ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#
    # method transforming reciprocal space wfks to real space
    def GridWFK(
            self, band_index:int=-1
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
        for k, kpt in enumerate(self.pw_indices):
            kx = kpt[0]
            ky = kpt[1]
            kz = kpt[2]
            if band_index >= 0:
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
        Returns copy of WFK with wavefunction coefficients expressed in real space.
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
        Method for converting the real space lattice parameters to reciprocal lattice parameters.
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
            self, points:np.ndarray, values:np.ndarray=np.empty([]), unique:bool=True, reciprocal:bool=False,
            inverse:bool=False
    )->tuple[np.ndarray, np.ndarray]:
        '''
        Method for generating symmetric data from irreducible data.

        Parameters
        ----------
        points : np.ndarray
            Irreducible set of points.
            Shape of (N,3).
        values : np.ndarray
            Values corresponding to irreducible points (such as energy eigenvalues w/ kpoints).
            Shape of (N,1).
        unique : bool
            Check for duplicate points.
            Default is to check (True).
        reciprocal : bool
            Calculate reciprocal space symmetry matrices from real space matrices.
            Default uses real space matrices (False).
        inverse : bool
            Use inverse symmetry operations.
            Default applies forwards operation (False).
        '''
        # check if system is noncentrosymmetric
        if self.time_reversal:
            # if system is centrosymmetric, do not double reciprocal symmetry operations
            if -3.0 in [np.trace(mat) for mat in self.symrel]:
                self.time_reversal = False
            else:
                print((
                'Noncentrosymmetric system identified, assuming time reversal symmetry\n'
                'To change this, set time_reversal attribute to False'
                ))
        # check if reciprocal or real space symmetries will be used
        sym_num = self.nsym
        if reciprocal:
            tnons = False
            sym_mats = [np.linalg.inv(mat).T for mat in self.symrel]
            if self.time_reversal:
                sym_mats = np.concatenate((sym_mats, [-mat for mat in sym_mats]), axis=0)
                sym_num *= 2
        else: 
            tnons = True
            sym_mats = self.symrel
        # initialize symmetrically equivalent point and value arrays
        ind_len = np.shape(points)[0]
        if values is np.empty([]):
            values = np.zeros((ind_len,1))
        sym_pts = np.zeros((sym_num*ind_len,3))
        sym_vals = np.zeros((sym_num*ind_len,self.nbands))
        # apply symmetry operations to points
        if inverse:
            for i, op in enumerate(sym_mats):
                if tnons:
                    points -= self.non_symm_vecs[i]
                new_pts:np.ndarray = np.matmul(np.linalg.inv(op), points.T).T
                sym_pts[i*ind_len:(i+1)*ind_len,:] = new_pts
                sym_vals[i*ind_len:(i+1)*ind_len,:] = values
        else:
            for i, op in enumerate(sym_mats):
                if tnons:
                    points -= self.non_symm_vecs[i]
                new_pts:np.ndarray = np.matmul(op, points.T).T
                sym_pts[i*ind_len:(i+1)*ind_len,:] = new_pts
                sym_vals[i*ind_len:(i+1)*ind_len,:] = values
        # points overlap on at edges of each symmetric block, remove duplicates
        if unique:
            sym_pts, unique_inds = np.unique(sym_pts, return_index=True, axis=0)
            sym_vals = np.take(sym_vals, unique_inds, axis=0)
        return sym_pts, sym_vals
    #-----------------------------------------------------------------------------------------------------------------#
    # method for creating symmetrically equivalent functions at specified kpoint
    def SymWFKs(
        self, kpoint:np.ndarray, band:int=-1
    )->Generator[Self, None, None]:
        '''
        Method for generating wavefunction planewave coefficients from coefficients of the irreducible BZ.

        Parameters
        ----------
        kpoint : np.ndarray
            A single reciprocal space point is provided to generate symmetrically equivalent coefficients.
            Shape (1,3).
        band : int
            Choose which band to pull coefficients from (indexed starting from zero).
            Default assumes coefficients from a single band are provided (-1).
        '''
        # helper functions for symmetrizing WFKs
        #-----#
        # function for calculating phase imparted by nonsymmorphic translation
        def _FindPhase(
            nonsymmvec:np.ndarray, g_vecs:np.ndarray, kpt:np.ndarray
        )->np.ndarray:
            if self.non_symm_vecs is np.zeros(1):
                return np.ones(len(g_vecs))
            elif np.sum(np.abs(nonsymmvec)) < 10**(-8):
                return np.ones(len(g_vecs))
            else:
                return np.exp(-1j*np.dot((kpt+g_vecs), nonsymmvec.T))
        #-----#
        # function for finding symmetrically distinct k points
        def _FindOrbit(
            sym_kpts:np.ndarray
        )->tuple[list,list]:
            sym_kpts = np.round(sym_kpts, decimals=15)
            _, unique_inds = np.unique(sym_kpts, return_index=True, axis=0)
            # for each unique kpoint check original point is related by reciprocal lattice vector
            dupes = []
            for i, ind1 in enumerate(unique_inds):
                if i in dupes:
                    continue
                for j, ind2 in enumerate(unique_inds):
                    if i == j or j in dupes:
                        continue
                    diff = np.abs(sym_kpoints[ind1] - sym_kpoints[ind2])
                    diff[diff < 10**(-12)] = 0.0
                    diff[diff > 0.999] = 1.0
                    mask = np.isin(diff, np.array([0.0,1.0]))
                    if mask.all():
                        dupes.append(j)
            return dupes, unique_inds.tolist()
        #-----#
        # start SymWFks method
        # find symmetric kpoints
        kpoint = kpoint.reshape((1,3))
        sym_kpoints, _ = self.Symmetrize(kpoint, unique=False, reciprocal=True)
        dupes, unique_inds = _FindOrbit(sym_kpoints)
        # find symmetric planewave indices
        sym_pw_inds, _ = self.Symmetrize(self.pw_indices, unique=False, reciprocal=True)
        sym_pw_inds = sym_pw_inds.astype(int)
        ind_range = self.pw_indices.shape[0]
        # find reciprocal lattice shifts to move all points into BZ
        bz = BZ(rec_latt=self.Real2Reciprocal())
        shifts = bz.GetShifts(sym_kpoints)
        # create WFK copies with new planewave indices
        for i, ind in enumerate(unique_inds):
            if i in dupes:
                continue
            ind1 = ind*ind_range
            ind2 = (ind+1)*ind_range
            new_pw_inds = sym_pw_inds[ind1:ind2,:]
            new_pw_inds += shifts[ind,:]
            new_coeffs = copy(self)
            new_coeffs.pw_indices = new_pw_inds
            phase_factor = _FindPhase(
                    self.non_symm_vecs[ind % len(self.non_symm_vecs)],
                    self.pw_indices,
                    sym_kpoints[ind,:]
            )
            if band >= 0:
                new_coeffs.wfk_coeffs = new_coeffs.wfk_coeffs[band] * phase_factor
                yield new_coeffs
            else:
                new_coeffs.wfk_coeffs *= phase_factor
                yield new_coeffs   
    #-----------------------------------------------------------------------------------------------------------------#
    # method that returns BZ kpoints and eigenvalues
    def GetBZPtsEigs(
        self
    )->tuple[np.ndarray,np.ndarray]:
        bz = BZ(rec_latt=self.Real2Reciprocal())
        bz_kpts, bz_eigs = self.Symmetrize(points=self.kpoints, values=self.eigenvalues, reciprocal=True)
        bz_kpts -= bz.GetShifts(bz_kpts)
        return bz_kpts, bz_eigs
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
        # this repetition is required by XSF format
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
        new_WFK.ngfftx += 1
        new_WFK.ngffty += 1
        new_WFK.ngfftz += 1
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
        new_WFK.ngfftx -= 1
        new_WFK.ngffty -= 1
        new_WFK.ngfftz -= 1
        return new_WFK
    #-----------------------------------------------------------------------------------------------------------------#
    # method for writing wavefunctions to XSF file
    def WriteXSF(
            self, xsf_file:str, _component:bool=True
    )->None:
        '''
        A method for writing numpy grids to an XSF formatted file.

        Parameters
        ----------
        xsf_file : str
            The file name.
        '''
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
            print(f'{self.ngfftx} {self.ngffty} {self.ngfftz}', file=xsf)
            print('0.0 0.0 0.0', file=xsf)
            print(f'{self.lattice[0,0]} {self.lattice[0,1]} {self.lattice[0,2]}', file=xsf)
            print(f'{self.lattice[1,0]} {self.lattice[1,1]} {self.lattice[1,2]}', file=xsf)
            print(f'{self.lattice[2,0]} {self.lattice[2,1]} {self.lattice[2,2]}', file=xsf)
            count = 0
            for z in range(self.ngfftz):
                for x in range(self.ngfftx):
                    for y in range(self.ngffty):
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