import numpy as np
from typing import Generator
from wfk_class import WFK

class BandU():
    def __init__(
        self, wfks:Generator, energy_level:float, width:float, grid:bool=True, fft:bool=True, norm:bool=True
    )->None:
        '''
        BandU object with methods for finding states and computing BandU functions from states.

        Parameters
        ----------
        wfks : Generator
            An iterable generator of WFK objects with wavefunction coefficients, k-points, and eigenvalue attributes.
        energy_level : float
            The energy level of interest relative to the Fermi energy.
        width : float
            Defines how far above and below the energy_level is searched for states.
            Search is done width/2 above and below, so total states captured are within 'width' energy.
        grid : bool
            Determines whether or not wavefunction coefficients are converted to 3D numpy grid.
            Default converts to grid (True).
        fft : bool
            Determines whether or not wavefunction coefficients are Fourier transformed to real space.
            Default converts from reciprocal space to real space (True).
        norm : bool
            Determines whether or not wavefunction coefficients are normalized.
            Default normalizes coefficients (True)
        '''
        # find all states within width
        self.bandu_fxns, total_states, fermi_energy = self._FindStates(energy_level, width, wfks, grid, fft, norm)
        # find overlap of states and diagonalize
        principal_vals, principal_vecs = self._FindOverlap()
        # linear combination of states weighted by principal components
        self.bandu_fxns = np.matmul(principal_vecs, self.bandu_fxns)
        # normalize bandu functions
        for i in range(total_states):
            normal_coeffs = WFK(wfk_coeffs=self.bandu_fxns[i,:]).Normalize()
            self.bandu_fxns[i,:] = normal_coeffs.wfk_coeffs
        # write output file
        with open('eigenvalues.out', 'w') as f:
            print(f'Width is {width}', file=f)
            print(f'Energy level: {energy_level+fermi_energy}, Fermi energy: {fermi_energy}', file=f)
            print(principal_vals, file=f)
#---------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ METHODS ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#
    # method transforming reciprocal space wfks to real space
    def _FindStates(
        self, energy_level:float, width:float, wfks:Generator, grid:bool, fft:bool, norm:bool
    )->tuple[np.ndarray, int, float]:
        # total number of states found within width
        total_states = 0
        # list for capturing all wfk coeffs of states within width
        u_vecs = []
        # state will be a WFK type
        state:WFK
        # loop through every state
        for i, state in enumerate(wfks):
            # pick necessary attributes for XSF writing
            if i == 0:
                self.ngfftx = state.ngfftx
                self.ngffty = state.ngffty
                self.ngfftz = state.ngfftz
                self.lattice = state.lattice
                self.natom = state.natom
                self.typat = state.typat
                self.znucltypat = state.znucltypat
                self.xred = state.xred
                fermi_energy = state.fermi_energy
                energy_level += fermi_energy
            # check if state has a band that crosses the width
            for i, band in enumerate(state.eigenvalues):
                if energy_level - width/2 <= band <= energy_level + width/2:
                    total_states += 1
                    # convert state to real space and add to u_vec list
                    coeffs = WFK(
                        wfk_coeffs=np.array(state.wfk_coeffs[i]),
                        pw_indices=np.array(state.pw_indices),
                        ngfftx=self.ngfftx,
                        ngffty=self.ngffty,
                        ngfftz=self.ngfftz
                    )
                    if grid:
                        coeffs = coeffs.GridWFK()
                    if fft:
                        coeffs = coeffs.FFT()
                    if norm:
                        coeffs = coeffs.Normalize()
                    u_vecs.append(coeffs.wfk_coeffs)
        if total_states == 0:
            raise ValueError(
            f'''Identified 0 states within provided width.
            Action: Increase width or increase fineness of k-point mesh.
            ''')
        # after FFT the wfk coefficients are reshaped from grid to vector for subsequent matrix operations
        u_vecs = np.array(u_vecs).reshape(total_states, self.ngfftx*self.ngffty*self.ngfftz)
        return u_vecs, total_states, fermi_energy
    #-----------------------------------------------------------------------------------------------------------------#
    # find overlap of states
    def _FindOverlap(
        self
    )->tuple[np.ndarray, np.ndarray]:
        # compute overlap matrix
        print('Computing overlap matrix')
        overlap_mat = np.matmul(np.conj(self.bandu_fxns), self.bandu_fxns.T)
        # diagonlize matrix
        principal_vals, principal_vecs = np.linalg.eig(overlap_mat)
        # organize eigenvectors and eigenvalues
        sorted_inds = np.flip(principal_vals.argsort())
        principal_vecs = principal_vecs.T
        principal_vals = np.take(principal_vals, sorted_inds)
        principal_vecs = np.take(principal_vecs, sorted_inds, axis=0)
        return principal_vals, principal_vecs
    #-----------------------------------------------------------------------------------------------------------------#
    # make xsf of BandU functions
    def MakeXSF(
        self, function_number:list, xsf_file:str
    )->None:
        if len(function_number) != 2:
            raise ValueError(f'function_number should contain two values, {len(function_number)} were received.')
        count = 1
        while function_number[0] <= count <= function_number[1]:
            # fetch nth bandu function coefficients
            bandu_fxn = self.bandu_fxns[count-1,:]
            # grid coefficients into 3D numpy array
            bandu_fxn = bandu_fxn.reshape((self.ngfftz, self.ngfftx, self.ngffty))
            # create WFK object from coefficients and provided attributes
            bandu_fxn = WFK(
                wfk_coeffs=bandu_fxn, 
                lattice=self.lattice,
                natom=self.natom,
                xred=self.xred,
                typat=self.typat,
                znucltypat=self.znucltypat,
                ngfftx=self.ngfftx,
                ngffty=self.ngffty,
                ngfftz=self.ngfftz
            )
            # convert to XSF format
            bandu_fxn = bandu_fxn.XSFFormat()
            # print out XSF file
            xsf_name = xsf_file + f'_{count}'
            bandu_fxn.WriteXSF(xsf_file=xsf_name)
            # update count and move to next bandu function
            count += 1