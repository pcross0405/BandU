import numpy as np
from typing import Generator
from wfk_class import WFK

class BandU():
    def __init__(
        self, wfks:Generator, energy_level:float, width:float, grid:bool=True, fft:bool=True, norm:bool=True,
        sym:bool=True
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
        self.grid:bool=grid
        self.fft:bool=fft
        self.norm:bool=norm
        self.sym:bool=sym
        self.total_states:int=0
        self.bandu_fxns:np.ndarray
        self.ngfftx:int
        self.ngffty:int
        self.ngfftz:int
        self.fermi_energy:float
        self.lattice:np.ndarray
        self.natom:int
        self.xred:np.ndarray
        self.typat:list[int]
        self.znucltypat:list[float]
        # apply grid, fourier transform, and normalize functions if necessary
        self._Process(
            # find all states within width
            self._FindStates(energy_level, width, wfks)
        )
        # find overlap of states and diagonalize
        principal_vals, principal_vecs = self._PrincipalComponents()
        # linear combination of states weighted by principal components
        self.bandu_fxns = np.matmul(principal_vecs, self.bandu_fxns)
        # normalize bandu functions
        for i in range(self.total_states):
            normal_coeffs = WFK(wfk_coeffs=self.bandu_fxns[i,:]).Normalize()
            self.bandu_fxns[i,:] = normal_coeffs.wfk_coeffs
        # write output file
        with open('eigenvalues.out', 'w') as f:
            print(f'Width: {width}, total states: {self.total_states}', file=f)
            print(f'Energy level: {energy_level+self.fermi_energy}, Fermi energy: {self.fermi_energy}', file=f)
            print(np.abs(principal_vals), file=f)            
#---------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ METHODS ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#
    # method transforming reciprocal space wfks to real space
    def _FindStates(
        self, energy_level:float, width:float, wfks:Generator
    )->list[WFK]:
        # list of WFK coefficients within specified range
        found_fxns = []
        # state will be a WFK type
        state:WFK
        # variable for getting info during first iteration
        first_iter = True
        # loop through every state
        for i, state in enumerate(wfks):
            # pick necessary attributes for XSF writing
            if first_iter:
                self.ngfftx = state.ngfftx
                self.ngffty = state.ngffty
                self.ngfftz = state.ngfftz
                self.lattice = state.lattice
                self.natom = state.natom
                self.typat = state.typat
                self.znucltypat = state.znucltypat
                self.xred = state.xred
                self.fermi_energy = state.fermi_energy
                energy_level += self.fermi_energy
                first_iter = False
            # check if state has a band that crosses the width
            for j, band in enumerate(state.eigenvalues):
                if energy_level - width/2 <= band <= energy_level + width/2:
                    # convert state to real space and add to u_vec list
                    coeffs = WFK(
                        wfk_coeffs=np.array(state.wfk_coeffs[j], dtype=complex),
                        pw_indices=np.array(state.pw_indices, dtype=int),
                        ngfftx=self.ngfftx,
                        ngffty=self.ngffty,
                        ngfftz=self.ngfftz,
                        kpoints=state.kpoints[i],
                        nsym=state.nsym,
                        symrel=state.symrel,
                        non_symm_vecs=state.non_symm_vecs,
                        lattice=self.lattice,
                        xred=self.xred,
                        znucltypat=self.znucltypat,
                        typat=self.typat,
                        natom=self.natom
                    )
                    found_fxns.append(coeffs)
        if found_fxns is []:
            raise ValueError(
            '''Identified 0 states within provided width.
            Action: Increase width or increase fineness of kpoint grid.
            ''')
        else:
            return found_fxns
    #-----------------------------------------------------------------------------------------------------------------#
    # method for processing planewave coefficient data from FindStates
    def _Process(
            self, found_fxns:list[WFK]
    )->None:
        funcs = []
        if self.sym:
            for coeffs in found_fxns:
                for sym_coeffs in coeffs.SymWFKs(kpoint=coeffs.kpoints):
                    self.total_states += 1
                    funcs.append(sym_coeffs)
        else:
            self.total_states = len(found_fxns)
            funcs = found_fxns
        del found_fxns
        if self.grid:
            for i, coeffs in enumerate(funcs):
                funcs[i] = coeffs.GridWFK()
        if self.fft:
            for i, coeffs in enumerate(funcs):
                funcs[i] = coeffs.FFT()
        if self.norm:
            for i, coeffs in enumerate(funcs):
                funcs[i] = coeffs.Normalize()
        funcs = [state.wfk_coeffs for state in funcs]
        funcs = np.array(funcs, dtype=complex).reshape(self.total_states, self.ngfftx*self.ngffty*self.ngfftz)
        print(f'{self.total_states} states found within specified range.')
        self.bandu_fxns = funcs
    #-----------------------------------------------------------------------------------------------------------------#
    # find principal components
    def _PrincipalComponents(
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
        '''
        Method for writing Band-U function as XSF file.

        Parameters
        ----------
        function_number : list
            Defines range of Band-U functions that will be printed.\n
            List should have only 2 elements.\n
            First being the first Band-U function to be printed.\n
            Second being the function number to be printed up to.
        xsf_file : str
            Root name given to all XSF files that are printed.
        '''
        # check if list has only 2 elements
        if len(function_number) != 2:
            raise ValueError(f'function_number should contain two values, {len(function_number)} were received.')
        # check if function number is within defined range
        if function_number[0] < 1:
            print('First element of function_number cannot be lower than 1, changing to 1 now.')
            function_number[0] = 1
        # update function number list if it exceeds maximum number of bandu functions
        if function_number[1] > self.bandu_fxns.shape[0]:
            print(f'Printing up to max Band-U function number: {self.bandu_fxns.shape[0]}')
            function_number[1] = self.bandu_fxns.shape[0]
        # check if lower limit is within defined range
        if function_number[0] > self.bandu_fxns.shape[0]:
            count = self.bandu_fxns.shape[0]
        else:
            count = function_number[0]
        print(f'Writing XSF files for Band-U functions {function_number[0]} through {function_number[1]}.')
        # loop over range and print XSF files
        while count <= function_number[1]:
            # fetch nth bandu function coefficients
            bandu_fxn = self.bandu_fxns[function_number[0]+count-2,:]
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
            # normalize
            bandu_fxn = bandu_fxn.Normalize()
            # convert to XSF format
            bandu_fxn = bandu_fxn.XSFFormat()
            # print out XSF file
            xsf_name = xsf_file + f'_{function_number[0]+count-1}'
            bandu_fxn.WriteXSF(xsf_file=xsf_name)
            # update count and move to next bandu function
            count += 1