import numpy as np
from typing import Generator
from wfk_class import WFK
from copy import copy
import pickle as pkl

class BandU():
    def __init__(
        self, wfks:Generator, energy_level:float, width:float, grid:bool=True, fft:bool=True, norm:bool=True,
        sym:bool=True, low_mem:bool=False
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
        low_mem : bool
            Run the program on a lower memory setting
            The low_mem tag will print plane wave cofficients to a Python pickle to read from disk later.
            Default does not run in low memory mode (False)
        '''
        self.grid:bool=grid
        self.fft:bool=fft
        self.norm:bool=norm
        self.sym:bool=sym
        self.low_mem:bool=low_mem
        self.total_states:int=0
        self.bandu_fxns:list[WFK]=[]
        # find all states within width
        self._FindStates(energy_level, width, wfks)
        print(f'{self.total_states} states found within specified energy range')
        # construct principal orbital components
        principal_vals = self._PrincipalComponents()
        # normalize bandu functions
        for i in range(self.total_states):
            self.bandu_fxns[i] = self.bandu_fxns[i].Normalize()
        # compute ratios
        omega_vals = self._GetOmega()
        # write output file
        fermi = self.bandu_fxns[0].fermi_energy
        with open('eigenvalues.out', 'w') as f:
            print(f'Width: {width}, total states: {self.total_states}', file=f)
            print(f'Energy level: {energy_level+fermi}, Fermi energy: {fermi}', file=f)
            print(np.abs(principal_vals), file=f)   
            print('Omega Values', file=f)
            print(omega_vals, file=f)         
#---------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ METHODS ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#
    # method transforming reciprocal space wfks to real space
    def _FindStates(
        self, energy_level:float, width:float, wfks:Generator[WFK,None,None]
    ):
        # loop through every state
        for i, state in enumerate(wfks):
            # check if state has a band that crosses the width
            min_en = state.fermi_energy + energy_level - width/2
            max_en = state.fermi_energy + energy_level + width/2
            for j, band in enumerate(state.eigenvalues):
                if min_en <= band <= max_en:
                    self.total_states += 1
                    coeffs = copy(state)
                    coeffs.wfk_coeffs = coeffs.wfk_coeffs[j]
                    coeffs.kpoints = coeffs.kpoints[i]
                    for wfk in self._Process(coeffs):
                        self.bandu_fxns.append(wfk)
        if self.bandu_fxns is []:
            raise ValueError(
            '''Identified 0 states within provided width.
            Action: Increase width or increase fineness of kpoint grid.
            ''')
    #-----------------------------------------------------------------------------------------------------------------#
    # method for processing planewave coefficient data from FindStates
    def _Process(
            self, state:WFK
    )->Generator[WFK,None,None]:
        funcs:list[WFK] = []
        # generate symmetrically equivalent coefficients
        if self.sym:
            for sym_coeffs in state.SymWFKs(kpoint=state.kpoints):
                self.total_states += 1
                funcs.append(sym_coeffs)
            self.total_states -= 1
        else:
            funcs.append(state)
        # apply desired transformations
        for wfk in funcs:
            if self.grid:
                wfk = wfk.GridWFK()
            if self.fft:
                wfk = wfk.FFT()
            if self.norm:
                wfk = wfk.Normalize()
            yield wfk
    #-----------------------------------------------------------------------------------------------------------------#
    # find principal components
    def _PrincipalComponents(
        self
    )->np.ndarray:
        # organize wfk coefficients 
        x = self.bandu_fxns[0].ngfftx
        y = self.bandu_fxns[0].ngffty
        z = self.bandu_fxns[0].ngfftz
        mat = np.zeros((self.total_states,x*y*z), dtype=complex)
        for i in range(self.total_states):
            mat[i,:] = self.bandu_fxns[i].wfk_coeffs.reshape((1,x*y*z))
        # compute overlap matrix
        print('Computing overlap matrix')
        overlap_mat = np.matmul(np.conj(mat), mat.T)
        # diagonlize matrix
        principal_vals, principal_vecs = np.linalg.eig(overlap_mat)
        principal_vecs = principal_vecs.T
        # organize eigenvectors and eigenvalues
        sorted_inds = np.flip(principal_vals.argsort())
        principal_vals = np.take(principal_vals, sorted_inds)
        principal_vecs = np.take(principal_vecs, sorted_inds, axis=0)
        mat = np.matmul(principal_vecs, mat)
        for i in range(self.total_states):
            self.bandu_fxns[i].wfk_coeffs = mat[i,:]
            self.bandu_fxns[i].wfk_coeffs *= np.exp(1j*np.pi/4)
        return principal_vals 
    #-----------------------------------------------------------------------------------------------------------------#
    # find ratio of real and imaginary components
    def _GetOmega(
        self
    )->np.ndarray:
        omega_vals = np.zeros(self.total_states, dtype=float)
        for i in range(self.total_states):
            coeffs = self.bandu_fxns[i].wfk_coeffs
            coeffs *= np.exp(1j*2*np.pi)
            omega = np.sum(coeffs.real*coeffs)/np.sum(coeffs.imag*coeffs)
            omega = np.abs(omega)
            omega_vals[i] = omega
        return omega_vals
    #-----------------------------------------------------------------------------------------------------------------#
    # make xsf of BandU functions
    def ToXSF(
        self, nums:list[int]=[], xsf_name:str='Principal_orbital_component'
    ):
        if nums is []:
            nums = [0,self.total_states-1]
        else:
            # check if list has only 2 elements
            if len(nums) != 2:
                raise ValueError(f'nums should contain two values, {len(nums)} were received.')
            # check if function number is within defined range
            if nums[0] < 1:
                print('First element of nums cannot be lower than 1, changing to 1 now.')
                nums[0] = 1
            # update function number list if it exceeds maximum number of bandu functions
            if nums[1] > self.total_states:
                print(f'Printing up to max Band-U function number: {self.total_states}')
                nums[1] = self.total_states
            # check if lower limit is within defined range
            if nums[0] > nums[1]:
                nums[0] = nums[1]
        print(f'Writing XSF files for Band-U functions {nums[0]} through {nums[1]}.')
        # write xsf files
        x = self.bandu_fxns[0].ngfftx
        y = self.bandu_fxns[0].ngffty
        z = self.bandu_fxns[0].ngfftz
        for i in range(nums[0]-1, nums[1]):
            file_name = xsf_name + f'_{i+1}'
            wfk = copy(self.bandu_fxns[i])
            wfk.wfk_coeffs = wfk.wfk_coeffs.reshape((z,x,y))
            wfk.WriteXSF(xsf_file=file_name)