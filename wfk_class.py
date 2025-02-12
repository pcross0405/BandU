import struct
import numpy as np
from scipy.fft import fftn
from typing import Generator
import sys
np.set_printoptions(threshold=sys.maxsize)

def bytes2float(bin_data):
    return struct.unpack('<d', bin_data)[0]

def bytes2int(bin_data):
    return int.from_bytes(bin_data, 'little', signed=True)

# method for converting real space lattice vectors to reciprocal space vectors
def Real2Reciprocal(real_lat:np.ndarray)->np.ndarray:
    '''
    Method for converting the real space lattice parameters to reciprocal lattice parameters

    Parameters
    ----------
    real_lat : np.ndarray
    '''
    # conversion by default converts Angstrom to Bohr since ABINIT uses Bohr
    a = real_lat[0,:]
    b = real_lat[1,:]
    c = real_lat[2,:]
    vol = np.dot(a,np.cross(b,c))
    b1 = 2*np.pi*(np.cross(b,c))/vol
    b2 = 2*np.pi*(np.cross(c,a))/vol
    b3 = 2*np.pi*(np.cross(a,b))/vol
    return np.array([b1,b2,b3]).reshape((3,3))

class WFK:
    '''
    A class for storing all variables from ABINIT WFK file
    '''
    def __init__(self, filename):
        self.filename = filename
        self.header = None
        self.version = None
        self.headform = None
        self.fform = None
        self.bandtot = None
        self.date = None
        self.intxc = None
        self.ixc = None
        self.natom = None
        self.ngfftx = None
        self.ngffty = None
        self.ngfftz = None
        self.nkpt = None
        self.nspden = None
        self.nspinor = None
        self.nsppol = None 
        self.nsym = None
        self.npsp = None
        self.ntypat = None
        self.occopt = None
        self.pertcase = None
        self.usepaw = None
        self.ecut = None
        self.ecutdg = None
        self.ecutsm = None
        self.ecut_eff = None
        self.qptnx = None
        self.qptny = None
        self.qptnz = None
        self.real_lattice = None
        self.rec_lattice = None
        self.stmbias = None
        self.tphysel = None
        self.tsmear = None
        self.usewvl = None
        self.istwfk = None
        self.bands = None
        self.npwarr = None
        self.so_psp = None
        self.symafm = None
        self.symrel = None
        self.typat = None
        self.kpts = None
        self.occ = None
        self.tnons = None
        self.znucltypat = None
        self.wtk = None
        self.title = None
        self.znuclpsp = None
        self.zionpsp = None
        self.pspso = None
        self.pspdat = None
        self.pspcod = None
        self.pspxc = None
        self.lmn_size = None
        self.residm = None
        self.xred = None
        self.etotal = None
        self.fermi = None
        self._ReadHeader(self.filename)
#---------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ METHODS ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#
    # method to read wavefunction header
    def _ReadHeader(self, filename)->None:
        wfk = open(filename, 'rb')
        print('Reading WFK header')
        #---------------#
        # unpack integers
        self.header = bytes2int(wfk.read(4))
        self.version = wfk.read(6).decode()
        if self.version.strip().split('.')[0] != '7':
            print(f'WARNING: currently only WFK files from ABINIT version 7 are supported')
        self.headform = bytes2int(wfk.read(4))
        self.fform = bytes2int(wfk.read(4))
        wfk.read(8)
        self.bandtot = bytes2int(wfk.read(4))
        self.date = bytes2int(wfk.read(4))
        self.intxc = bytes2int(wfk.read(4))
        self.ixc = bytes2int(wfk.read(4))
        self.natom = bytes2int(wfk.read(4))
        self.ngfftx = bytes2int(wfk.read(4))
        self.ngffty = bytes2int(wfk.read(4))
        self.ngfftz = bytes2int(wfk.read(4))
        self.nkpt = bytes2int(wfk.read(4))
        self.nspden = bytes2int(wfk.read(4))
        self.nspinor = bytes2int(wfk.read(4))
        self.nsppol = bytes2int(wfk.read(4))
        self.nsym = bytes2int(wfk.read(4))
        self.npsp = bytes2int(wfk.read(4))
        self.ntypat = bytes2int(wfk.read(4))
        self.occopt = bytes2int(wfk.read(4))
        self.pertcase = bytes2int(wfk.read(4))
        self.usepaw = bytes2int(wfk.read(4))
        if self.usepaw != 0:
            print(f'WARNING: usepaw is {self.usepaw}, support has not been added for PAW potentials yet')
        #--------------#
        # unpack doubles
        self.ecut = bytes2float(wfk.read(8))
        self.ecutdg = bytes2float(wfk.read(8))
        self.ecutsm = bytes2float(wfk.read(8))
        self.ecut_eff = bytes2float(wfk.read(8))
        self.qptnx = bytes2float(wfk.read(8))
        self.qptny = bytes2float(wfk.read(8))
        self.qptnz = bytes2float(wfk.read(8))
        rprimd_ax = bytes2float(wfk.read(8))
        rprimd_ay = bytes2float(wfk.read(8))
        rprimd_az = bytes2float(wfk.read(8))
        rprimd_bx = bytes2float(wfk.read(8))
        rprimd_by = bytes2float(wfk.read(8))
        rprimd_bz = bytes2float(wfk.read(8))
        rprimd_cx = bytes2float(wfk.read(8))
        rprimd_cy = bytes2float(wfk.read(8))
        rprimd_cz = bytes2float(wfk.read(8))
        #------------------------#
        # convert Bohr to Angstrom
        self.real_lattice = 0.529177*np.array([[rprimd_ax, rprimd_ay, rprimd_az],
                                    [rprimd_bx, rprimd_by, rprimd_bz], 
                                    [rprimd_cx, rprimd_cy, rprimd_cz]]).reshape((3,3))
        self.rec_lattice = Real2Reciprocal(self.real_lattice)
        self.stmbias = bytes2float(wfk.read(8))
        self.tphysel = bytes2float(wfk.read(8))
        self.tsmear = bytes2float(wfk.read(8))
        #---------------#
        # unpack integers
        self.usewvl = bytes2int(wfk.read(4))
        wfk.read(8)
        self.istwfk = []
        for i in range(self.nkpt):
            val = bytes2int(wfk.read(4))
            self.istwfk.append(val)
            if val > 1:
                print(f'WARNING: istwfk value {i} is greater than 1, considering rerunning ABINIT with kptopt 3')       
        self.bands = []
        for i in range(self.nkpt*self.nsppol):
            val = bytes2int(wfk.read(4))
            self.bands.append(val)
        self.npwarr = []
        for i in range(self.nkpt):
            val = bytes2int(wfk.read(4))
            self.npwarr.append(val)
        self.so_psp = []
        for i in range(self.npsp):
            val = bytes2int(wfk.read(4))
            self.so_psp.append(val)
        self.symafm = []
        for i in range(self.nsym):
            val = bytes2int(wfk.read(4))
            self.symafm.append(val)
        self.symrel = []
        for i in range(self.nsym):
            arr = np.zeros((3,3))
            arr[0,0] = bytes2int(wfk.read(4))
            arr[1,0] = bytes2int(wfk.read(4))
            arr[2,0] = bytes2int(wfk.read(4))
            arr[0,1] = bytes2int(wfk.read(4))
            arr[1,1] = bytes2int(wfk.read(4))
            arr[2,1] = bytes2int(wfk.read(4))
            arr[0,2] = bytes2int(wfk.read(4))
            arr[1,2] = bytes2int(wfk.read(4))
            arr[2,2] = bytes2int(wfk.read(4))
            self.symrel.append(arr)
        self.typat = []
        for i in range(self.natom):
            val = bytes2int(wfk.read(4))
            self.typat.append(val)
        #--------------#
        # unpack doubles
        self.kpts = []
        for i in range(self.nkpt):
            vec = np.zeros(3)
            vec[0] = bytes2float(wfk.read(8))
            vec[1] = bytes2float(wfk.read(8))
            vec[2] = bytes2float(wfk.read(8))
            self.kpts.append(vec)
        self.occ = []
        for i in range(self.bandtot):
            val = bytes2float(wfk.read(8))
            self.occ.append(val)
        self.tnons = []
        for i in range(self.nsym):
            vec = np.zeros(3)
            vec[0] = bytes2float(wfk.read(8))
            vec[1] = bytes2float(wfk.read(8))
            vec[2] = bytes2float(wfk.read(8))
            self.tnons.append(vec)
        self.znucltypat = []
        for i in range(self.ntypat):
            val = bytes2float(wfk.read(8))
            self.znucltypat.append(val)
        self.wtk = []           
        for i in range(self.nkpt):
            val = bytes2float(wfk.read(8))
            self.wtk.append(val)
        #-----------------------#
        # unpack pseudopotentials
        wfk.read(4)
        self.title = self.znuclpsp = self.zionpsp = self.pspso = self.pspdat = self.pspcod = []
        self.pspxc = self.lmn_size = []
        for i in range(self.npsp):
            wfk.read(4)
            self.title.append(wfk.read(132).decode())
            self.znuclpsp.append(bytes2float(wfk.read(8)))
            self.zionpsp.append(bytes2float(wfk.read(8)))
            self.pspso.append(bytes2int(wfk.read(4)))
            self.pspdat.append(bytes2int(wfk.read(4)))
            self.pspcod.append(bytes2int(wfk.read(4)))
            self.pspxc.append(bytes2int(wfk.read(4)))
            self.lmn_size.append(bytes2int(wfk.read(4)))
            wfk.read(4)
        #------------------#
        # unpack coordinates
        wfk.read(4)
        self.residm = bytes2float(wfk.read(8))
        self.xred = []
        for i in range(self.natom):
            vec = np.zeros(3)
            vec[0] = bytes2float(wfk.read(8))
            vec[1] = bytes2float(wfk.read(8))
            vec[2] = bytes2float(wfk.read(8))
            self.xred.append(vec)
        #------------------------------------#
        # unpack total energy and fermi energy
        self.etotal = bytes2float(wfk.read(8))
        self.fermi = bytes2float(wfk.read(8))
        wfk.read(4)
        print('WFK header read')
        wfk.close()
    #-----------------------------------------------------------------------------------------------------------------#
    # method to read entire body of wavefunction file
    def _ReadWFK(self)->Generator[list, list, list]:
        wfk = open(self.filename, 'rb')
        #-----------#
        # skip header
        wfk.read(298)
        wfk.read(4*(2*self.nkpt + self.nkpt*self.nsppol + self.npsp + 10*self.nsym + self.natom))
        wfk.read(8*(4*self.nkpt + self.bandtot + 3*self.nsym + self.ntypat + 3*self.natom))
        wfk.read(self.npsp*(176))
        #-------------------------------#
        # begin reading wavefunction body
        for i in range(self.nsppol):
            for j in range(self.nkpt):
                print(f'Reading kpoint {j+1} of {self.nkpt}', end='\r')
                if j+1 == self.nkpt:
                    print('\n', end='')
                kpoints = []
                eigenvalues = []
                occupancies = []
                coeffs = []
                wfk.read(4)
                npw = bytes2int(wfk.read(4))
                self.nspinor = bytes2int(wfk.read(4))
                nband_temp = bytes2int(wfk.read(4))
                wfk.read(8)
                for pw in range(npw):
                    kx = bytes2int(wfk.read(4))
                    ky = bytes2int(wfk.read(4))
                    kz = bytes2int(wfk.read(4))
                    kpoints.append((kx, ky, kz))
                wfk.read(8)
                for nband in range(nband_temp):
                    eigenval = bytes2float(wfk.read(8))
                    eigenvalues.append(eigenval)
                for nband in range(nband_temp):
                    occ = bytes2float(wfk.read(8))
                    occupancies.append(occ)
                wfk.read(4)
                for nband in range(nband_temp):
                    wfk.read(4)
                    cg = []
                    for pw in range(npw):
                        cg1 = bytes2float(wfk.read(8))
                        cg2 = bytes2float(wfk.read(8))
                        cg.append(cg1 + 1j*cg2)
                    coeffs.append(cg)
                    wfk.read(4)            
                yield eigenvalues, coeffs, kpoints
        print('WFK body read')
        wfk.close()
    #-----------------------------------------------------------------------------------------------------------------#
    # method to read only eigenvalues from body of wavefunction file
    def _ReadEigenvalues(self):
        wfk = open(self.filename, 'rb')
        #-----------#
        # skip header
        wfk.read(298)
        wfk.read(4*(2*self.nkpt + self.nkpt*self.nsppol + self.npsp + 10*self.nsym + self.natom))
        wfk.read(8*(4*self.nkpt + self.bandtot + 3*self.nsym + self.ntypat + 3*self.natom))
        wfk.read(self.npsp*(176))
        #-------------------------------#
        # begin reading wavefunction body
        for i in range(self.nsppol):
            for j in range(self.nkpt):
                print(f'Reading kpoint {j+1} of {self.nkpt}', end='\r')
                if j+1 == self.nkpt:
                    print('\n', end='')
                eigenvalues = []
                wfk.read(4)
                npw = bytes2int(wfk.read(4))
                self.nspinor = bytes2int(wfk.read(4))
                nband_temp = bytes2int(wfk.read(4))
                wfk.read(8)
                wfk.read(12*npw)
                wfk.read(8)
                #------------------------------------------------------------------#
                # only need eigenvalues for Fermi surface, skip over everything else
                for nband in range(nband_temp):
                    eigenval = bytes2float(wfk.read(8))
                    eigenvalues.append(eigenval)
                
                wfk.read(nband_temp*8)
                wfk.read(4)
                wfk.read(nband_temp*(8 + npw*16))
                yield eigenvalues
        print('WFK body read')
        wfk.close()
    #-----------------------------------------------------------------------------------------------------------------#
    # method transforming reciprocal space wfks to real space
    def _FFT(
            self, energy_level:float, width:float
    )->tuple[np.ndarray, int]:
        # fermi_states is the number of states identified from the WFK file within the width about the energy_level
        # u_vecs are vectors of planewave coefficients for each state identified 
        # kpts are the H, K, L indices for the planewaves
        fermi_states = 0
        u_vecs = []
        for eigenvals, coeffs, kpts in self._ReadWFK():
            for i, band in enumerate(eigenvals):
                if energy_level - width/2 <= band <= energy_level + width/2:
                    fermi_states += 1
                    # planewave coefficients are first mapped to grid for FFT
                    grid_wfk = np.zeros((self.ngfftz, self.ngfftx, self.ngffty), dtype=complex)
                    for k, kpoint in enumerate(kpts):
                        kx = kpoint[0]
                        ky = kpoint[1]
                        kz = kpoint[2]
                        grid_wfk[kz][kx][ky] = coeffs[i][k]
                    grid_wfk = fftn(grid_wfk, norm='ortho')
                    u_vecs.append(grid_wfk)
        # after FFT the wfk coefficients are reshaped from grid to vector for subsequent matrix operations
        u_vecs = np.array(u_vecs).reshape(fermi_states, self.ngfftx*self.ngffty*self.ngfftz)
        if fermi_states == 0:
            raise ValueError(f'''Identified 0 states within provided width.
            Action: Increase width or increase fineness of kpoint mesh.
            ''')
        return u_vecs, fermi_states
    #-----------------------------------------------------------------------------------------------------------------#
    # method for normalizing wfks
    def _Normalize(
            self, wfk:np.ndarray
    )->np.ndarray:
        # calculate normalization constant and apply to wfk
        norm = np.dot(wfk.flatten(), np.conj(wfk).flatten())
        wfk = wfk/np.sqrt(norm)
        return wfk
    #-----------------------------------------------------------------------------------------------------------------#
    # method for expanding a grid into XSF format
    def _XSFFormat(
            self, eigfunc:np.ndarray
    )->np.ndarray:
        # append zeros to ends of all axes in eigfunc
        # zeros get replaced by values at beginning of each axis
        # this is repetition is required by XSF format
        eigfunc = np.append(eigfunc, np.zeros((1, self.ngfftx, self.ngffty)), axis=0)
        eigfunc = np.append(eigfunc, np.zeros((self.ngfftz+1, 1, self.ngffty)), axis=1)
        eigfunc = np.append(eigfunc, np.zeros((self.ngfftz+1, self.ngfftx+1, 1)), axis=2)
        for x in range(self.ngfftx+1):
            for y in range(self.ngffty+1):
                for z in range(self.ngfftz+1):
                    if x == self.ngfftx:
                        eigfunc[z][x][y] = eigfunc[z][0][y]
                    if y == self.ngffty:
                        eigfunc[z][x][y] = eigfunc[z][x][0]
                    if z == self.ngfftz:
                        eigfunc[z][x][y] = eigfunc[0][x][y]
                    if x == self.ngfftx and y == self.ngffty:
                        eigfunc[z][x][y] = eigfunc[z][0][0]
                    if x == self.ngfftx and z == self.ngfftz:
                        eigfunc[z][x][y] = eigfunc[0][0][y]
                    if z == self.ngfftz and y == self.ngffty:
                        eigfunc[z][x][y] = eigfunc[0][x][0]
                    if x == self.ngfftx and y == self.ngffty and z == self.ngfftz:
                        eigfunc[z][x][y] = eigfunc[0][0][0]
        return eigfunc
    #-----------------------------------------------------------------------------------------------------------------#
    # method for calculating overlaps of states at specified energy level
    def BandU(
            self, energy_level:float=None, states:int=None, width:float=0.005, XSFFormat:bool=True
    )->Generator:
        '''
        A generator for writing states to a numpy grid.

        Parameters
        ----------
        energy_level : float
            This defines what energy level to look for states at, default is the Fermi energy
        states : int
            This defines how many grids are generated, default is all states found
        width : float
            This defines the upper and lower bounds on the energy_level, default is 0.005 Hartree
        '''
        if energy_level == None:
            energy_level = self.fermi
        else:
            energy_level += self.fermi
        # apply FFT to reciprocal space wfks
        u_vecs, fermi_states = self._FFT(energy_level, width)
        if states == None or states > fermi_states:
            states = fermi_states
        # normalize wfks
        for i in range(fermi_states):
            u_vecs[i,:] = self._Normalize(u_vecs[i,:])
        # begin computing overlap of u_vecs
        print('Computing overlap matrix')
        overlap_mat = np.matmul(np.conj(u_vecs), u_vecs.T)
        # diagonalize matrix, since matrix is Hermitian we can use eigh function
        print('Diagonalizing overlap matrix')
        principal_vals, principal_vecs = np.linalg.eig(overlap_mat)
        # sort eigenvalues and vectors by order of eigenvalue
        sorted_inds = np.flip(principal_vals.argsort())
        principal_vecs = principal_vecs.T
        principal_vals = np.take(principal_vals, sorted_inds)
        principal_vecs = np.take(principal_vecs, sorted_inds, axis=0)
        # write output file
        with open('eigenvalues.out', 'w') as f:
            print(f'Width is {width}', file=f)
            print(f'Calculated for states with energy {energy_level}', file=f)
            print(principal_vals, file=f)
        # find new wavefunctions from combinations of u_vec wavefunctions
        eigfuncs = np.copy(u_vecs)
        eigfuncs = np.matmul(principal_vecs, u_vecs)
        # normalize eigfuncs
        for i in range(fermi_states):
            eigfuncs[i,:] = self._Normalize(eigfuncs[i,:])
        # convert to XSF format
        for i in range(states):
            print(f'Finding BandU eigenfunction {i+1} of {states}')
            # reshape eigfunc to a grid for the purpose of writing to XSF file
            eigfunc = eigfuncs[i].reshape((self.ngfftz, self.ngfftx, self.ngffty))
            # convert grid to XSF format
            if XSFFormat:
                eigfunc = self._XSFFormat(eigfunc)
            yield eigfunc
    #-----------------------------------------------------------------------------------------------------------------#
    # method for writing wavefunctions to XSF file
    def WriteXSF(
            self, xsf_file:str, state:np.ndarray, component:bool=True
    )->None:
        '''
        A method for writing numpy grids to an XSF formatted file

        Parameters
        ----------
        xsf_file : str
            The file name
        state : numpy ndarry
            The grid to be written to an XSF
        '''
        # first run writes out real part of eigenfunction to xsf
        if component:
            xsf_file += '_real.xsf'
        # second run writes out imaginary part
        else:
            xsf_file += '_imag.xsf'
        with open(xsf_file, 'w') as xsf:
            print('DIM-GROUP', file=xsf)
            print('3 1', file=xsf)
            print('PRIMVEC', file=xsf)
            print(f'{self.real_lattice[0,0]} {self.real_lattice[0,1]} {self.real_lattice[0,2]}', file=xsf)
            print(f'{self.real_lattice[1,0]} {self.real_lattice[1,1]} {self.real_lattice[1,2]}', file=xsf)
            print(f'{self.real_lattice[2,0]} {self.real_lattice[2,1]} {self.real_lattice[2,2]}', file=xsf)
            print('PRIMCOORD', file=xsf)
            print(f'{self.natom} 1', file=xsf)
            for i, coord in enumerate(self.xred):
                atomic_num = int(self.znucltypat[self.typat[i] - 1])
                cart_coord = np.dot(coord, self.real_lattice)
                print(f'{atomic_num} {cart_coord[0]} {cart_coord[1]} {cart_coord[2]}', file=xsf)
            print('ATOMS', file=xsf)
            for i, coord in enumerate(self.xred):
                atomic_num = int(self.znucltypat[self.typat[i] - 1])
                cart_coord = np.dot(coord, self.real_lattice)
                print(f'{atomic_num} {cart_coord[0]} {cart_coord[1]} {cart_coord[2]}', file=xsf)
            print('BEGIN_BLOCK_DATAGRID3D', file=xsf)
            print('datagrids', file=xsf)
            print('DATAGRID_3D_DENSITY', file=xsf)
            print(f'{self.ngfftx+1} {self.ngffty+1} {self.ngfftz+1}', file=xsf)
            print('0.0 0.0 0.0', file=xsf)
            print(f'{self.real_lattice[0,0]} {self.real_lattice[0,1]} {self.real_lattice[0,2]}', file=xsf)
            print(f'{self.real_lattice[1,0]} {self.real_lattice[1,1]} {self.real_lattice[1,2]}', file=xsf)
            print(f'{self.real_lattice[2,0]} {self.real_lattice[2,1]} {self.real_lattice[2,2]}', file=xsf)
            count = 0
            for z in range(self.ngfftz+1):
                for x in range(self.ngfftx+1):
                    for y in range(self.ngffty+1):
                        count += 1
                        if component:
                            print(state[z][x][y].real, file=xsf, end=' ')
                        else:
                            print(state[z][x][y].imag, file=xsf, end=' ')
                        if count == 6:
                            count = 0
                            print('\n', file=xsf, end='')
            print('END_DATAGRID_3D', file=xsf)
            print('END_BLOCK_DATAGRID3D', file=xsf)
        # rerun method to write out imaginary part
        if component:
            xsf_file = xsf_file.split('_real')[0]
            self.WriteXSF(xsf_file, state, component=False)
    #-----------------------------------------------------------------------------------------------------------------#
    # a wrapper around BandU and WriteXSF for calling both in one line
    def MakeXSF(
            self, xsf_name:str, energy_level:float=None, states:int=None, width:float=0.005
    )->None:
        '''
        A wrapper around the BandU and WriteXSF methods for making XSF files in one line

        Parameters
        ----------
        xsf_name : str
            The name of the output XSF file
        energy_level : float
            This defines what energy level to look for states at, default is the Fermi energy
        states : int
            This defines how many grids are generated, default is all states found
        width : float
            This defines the upper and lower bounds on the energy_level, default is 0.005 Hartree
        '''
        for i, state in enumerate(self.BandU(energy_level=energy_level, states=states, width=width)):
            print('Writing XSF')
            self.WriteXSF(f'{xsf_name}_{i+1}', state)
            print('XSF complete')