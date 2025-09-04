import struct
import numpy as np
from typing import Generator, Union
import sys
from . import wfk_class as wc
np.set_printoptions(threshold=sys.maxsize)

def bytes2float(bin_data)->float:
    return struct.unpack('<d', bin_data)[0]

def bytes2int(bin_data)->int:
    return int.from_bytes(bin_data, 'little', signed=True)

class Abinit7WFK():
    def __init__(
        self, filename:str
    ):
        self.filename = filename
        self._ReadHeader()
#---------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ METHODS ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#
    # method to read wavefunction header for abinit version 7
    def _ReadHeader(
        self
    )->None:
        wfk = open(self.filename, 'rb')
        print('Reading WFK header')
        #---------------#
        # unpack integers
        self.header = bytes2int(wfk.read(4))
        self.version = wfk.read(6).decode()
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
        if self.usepaw == 1:
            raise NotImplementedError(
                f'''PAW potentials are not supported by this program for Abinit v{self.version}.
                This program supports PAW potentials in Abinit v10, consider upgrading to the latest version.
                '''
            )
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
        self.stmbias = bytes2float(wfk.read(8))
        self.tphysel = bytes2float(wfk.read(8))
        self.tsmear = bytes2float(wfk.read(8))
        #---------------#
        # unpack integers
        self.usewvl = bytes2int(wfk.read(4))
        wfk.read(8)
        self.istwfk:list[int] = []
        for i in range(self.nkpt):
            val = bytes2int(wfk.read(4))
            self.istwfk.append(val)    
        self.bands:list[int] = []
        for i in range(self.nkpt*self.nsppol):
            val = bytes2int(wfk.read(4))
            self.bands.append(val)
        self.npwarr:list[int] = []
        for i in range(self.nkpt):
            val = bytes2int(wfk.read(4))
            self.npwarr.append(val)
        self.so_psp:list[int] = []
        for i in range(self.npsp):
            val = bytes2int(wfk.read(4))
            self.so_psp.append(val)
        self.symafm:list[int] = []
        for i in range(self.nsym):
            val = bytes2int(wfk.read(4))
            self.symafm.append(val)
        self.symrel:list[np.ndarray] = []
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
        self.typat:list[int] = []
        for i in range(self.natom):
            val = bytes2int(wfk.read(4))
            self.typat.append(val)
        #--------------#
        # unpack doubles
        self.kpts:list[np.ndarray] = []
        for i in range(self.nkpt):
            vec = np.zeros(3)
            vec[0] = bytes2float(wfk.read(8))
            vec[1] = bytes2float(wfk.read(8))
            vec[2] = bytes2float(wfk.read(8))
            self.kpts.append(vec)
        self.occ:list[float] = []
        for i in range(self.bandtot):
            val = bytes2float(wfk.read(8))
            self.occ.append(val)
        self.tnons:list[np.ndarray] = []
        for i in range(self.nsym):
            vec = np.zeros(3)
            vec[0] = bytes2float(wfk.read(8))
            vec[1] = bytes2float(wfk.read(8))
            vec[2] = bytes2float(wfk.read(8))
            self.tnons.append(vec)
        self.znucltypat:list[float] = []
        for i in range(self.ntypat):
            val = bytes2float(wfk.read(8))
            self.znucltypat.append(val)
        self.wtk:list[float] = []           
        for i in range(self.nkpt):
            val = bytes2float(wfk.read(8))
            self.wtk.append(val)
        #-----------------------#
        # unpack pseudopotentials
        wfk.read(4)
        self.title:list[str] = []
        self.znuclpsp:list[float] = []
        self.zionpsp:list[float] = []
        self.pspso:list[int] = []
        self.pspdat:list[int] = []
        self.pspcod:list[int] = []
        self.pspxc:list[int] = []
        self.lmn_size:list[int] = []
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
        self.xred:list[np.ndarray] = []
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
    # check for time reversal symmetry
    def _CheckTimeRev(
        self
    ):
        # if system is centrosymmetric, do not double reciprocal symmetry operations
        if True in [np.array_equal(-np.identity(3),mat) for mat in self.symrel]:
            self.time_reversal = False
        else:
            self.time_reversal = True
            print((
            'Noncentrosymmetric system identified, assuming time reversal symmetry\n'
            'To change this, set "check_time_rev" keyword to False when calling ReadWFK() or ReadEigenvalues()'
            ))
    #-----------------------------------------------------------------------------------------------------------------#
    # method to read entire body of abinit version 7 wavefunction file
    def ReadWFK(
        self, energy_level=np.nan, width=np.nan, check_time_rev:bool=True
    )->Generator[wc.WFK, None, None]:
        '''
        Method that constructs WFK objects from ABINIT v7 WFK file

        Parameters
        ----------
        energy_level : float
        The energy level to pull plane wave coefficients from
        This is relative to the Fermi energy
        If not defined but width is defined, default is 0 Hartree

        width : float
        Defines range about energy_level to search for plane wave coefficients
        Total range is equal to width, so look width/2 above and below energy_level
        If not defined but energy_level is defined, default is 0.005 Hartree
        '''
        # check for time reversal symmetry
        if check_time_rev:
            self._CheckTimeRev()
        else:
            self.time_reversal = False
        # read wfk file
        wfk = open(self.filename, 'rb')
        skip = True
        if energy_level is np.nan:
            if width is np.nan:
                skip = False
            else:
                energy_level = 0
        else:
            if width is np.nan:
                width = 0.005
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
                pw_indices:list[tuple] = []
                eigenvalues:list[float] = []
                wfk.read(4)
                npw = bytes2int(wfk.read(4))
                self.nspinor = bytes2int(wfk.read(4))
                nband_temp = bytes2int(wfk.read(4))
                wfk.read(8)
                for pw in range(npw):
                    kx = bytes2int(wfk.read(4))
                    ky = bytes2int(wfk.read(4))
                    kz = bytes2int(wfk.read(4))
                    pw_indices.append((kx, ky, kz))
                wfk.read(8)
                for nband in range(nband_temp):
                    eigenval = bytes2float(wfk.read(8))
                    eigenvalues.append(eigenval)
                # skip reading coefficients if they energy level of interest is not in band
                if skip:
                    min_val = self.fermi + energy_level - width/2
                    max_val = self.fermi + energy_level + width/2
                    eig_found = False
                    for eigval in eigenvalues:
                        if min_val <= eigval <= max_val:
                            eig_found = True
                            break
                    if eig_found:
                        yield self._ReadCoeffs(
                            wfk,
                            nband_temp,
                            npw,
                            eigenvalues,
                            pw_indices,
                            ind=j
                        )
                    else:
                        wfk.read(nband_temp*8)
                        wfk.read(4)
                        wfk.read(nband_temp*(8 + npw*16))
                else:
                    yield self._ReadCoeffs(
                        wfk,
                        nband_temp,
                        npw,
                        eigenvalues,
                        pw_indices,
                        ind=j
                    )
        print('WFK body read')
        wfk.close()
    #-----------------------------------------------------------------------------------------------------------------#
    # method to read in plane wave coefficients
    def _ReadCoeffs(
        self, wfk_file, nbands, npws, eigs, pw_inds, ind
    )->wc.WFK:
        occupancies = np.zeros((1,nbands), dtype=float)
        coeffs = np.zeros((nbands,npws), dtype=complex)
        for nband in range(nbands):
            occ = bytes2float(wfk_file.read(8))
            occupancies[:,nband] = occ
        wfk_file.read(4)
        for nband in range(nbands):
            wfk_file.read(4)
            cg:np.ndarray = np.zeros((1,npws), dtype=complex)
            for pw in range(npws):
                cg1 = bytes2float(wfk_file.read(8))
                cg2 = bytes2float(wfk_file.read(8))
                cg[:,pw] = cg1 + 1j*cg2
            coeffs[nband,:] = cg
            wfk_file.read(4) 
        return wc.WFK(
            eigenvalues=np.array(eigs), 
            wfk_coeffs=np.array(coeffs),
            pw_indices=np.array(pw_inds),
            kpoints=np.array(self.kpts[ind]),
            nkpt=self.nkpt,
            nbands=self.bands[0],
            ngfftx=self.ngfftx,
            ngffty=self.ngffty,
            ngfftz=self.ngfftz,
            symrel=np.array(self.symrel),
            nsym=self.nsym,
            lattice=self.real_lattice,
            natom=self.natom,
            xred=np.array(self.xred),
            typat=self.typat,
            znucltypat=self.znucltypat,
            fermi_energy=self.fermi,
            non_symm_vecs=np.array(self.tnons),
            time_reversal=self.time_reversal
        )
    #-----------------------------------------------------------------------------------------------------------------#
    # method to read only eigenvalues from body of abinit version 7 wavefunction file
    def ReadEigenvalues(
        self, check_time_rev:bool=True
    )->Generator[wc.WFK, None, None]:
        '''
        Method that constructs WFK objects from ABINIT v7 WFK file.
        '''
        # check for time reversal symmetry
        if check_time_rev:
            self._CheckTimeRev()
        else:
            self.time_reversal = False
        # read wfk
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
                eigenvalues:list[float] = []
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
                yield wc.WFK(
                    eigenvalues=np.array(eigenvalues), 
                    kpoints=np.array(self.kpts[j]),
                    nkpt=self.nkpt,
                    nbands=self.bands[0],
                    ngfftx=self.ngfftx,
                    ngffty=self.ngffty,
                    ngfftz=self.ngfftz,
                    symrel=np.array(self.symrel),
                    nsym=self.nsym,
                    lattice=self.real_lattice,
                    natom=self.natom,
                    xred=np.array(self.xred),
                    typat=self.typat,
                    znucltypat=self.znucltypat,
                    fermi_energy=self.fermi,
                    non_symm_vecs=np.array(self.tnons),
                    time_reversal=self.time_reversal
                )
        print('WFK body read')
        wfk.close()

#---------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------- END CLASS -----------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#

class Abinit10WFK():
    def __init__(
        self, filename:str
    ):
        self.filename = filename
        self._ReadHeader()
#---------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ METHODS ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#
    # method to read wavefunction header for abinit version 7
    def _ReadHeader(
        self
    )->None:
        wfk = open(self.filename, 'rb')
        print('Reading WFK header')
        #---------------#
        # unpack integers
        self.header = bytes2int(wfk.read(4))
        self.version = wfk.read(6).decode()
        wfk.read(2)
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
        self.real_lattice = 0.529177*np.array([
            [rprimd_ax, rprimd_ay, rprimd_az],
            [rprimd_bx, rprimd_by, rprimd_bz], 
            [rprimd_cx, rprimd_cy, rprimd_cz]
        ]).reshape((3,3))
        self.stmbias = bytes2float(wfk.read(8))
        self.tphysel = bytes2float(wfk.read(8))
        self.tsmear = bytes2float(wfk.read(8))
        #---------------#
        # unpack integers
        self.usewvl = bytes2int(wfk.read(4))
        self.nshiftk_orig = bytes2int(wfk.read(4))
        self.nshiftk = bytes2int(wfk.read(4))
        self.mband = bytes2int(wfk.read(4))
        wfk.read(8)
        self.istwfk:list[int] = []
        for _ in range(self.nkpt):
            val = bytes2int(wfk.read(4))
            self.istwfk.append(val)      
        self.bands:list[int] = []
        for _ in range(self.nkpt*self.nsppol):
            val = bytes2int(wfk.read(4))
            self.bands.append(val)
        self.npwarr:list[int] = []
        for _ in range(self.nkpt):
            val = bytes2int(wfk.read(4))
            self.npwarr.append(val)
        self.so_psp:list[int] = []
        for _ in range(self.npsp):
            val = bytes2int(wfk.read(4))
            self.so_psp.append(val)
        self.symafm:list[int] = []
        for _ in range(self.nsym):
            val = bytes2int(wfk.read(4))
            self.symafm.append(val)
        self.symrel:list[np.ndarray] = []
        for _ in range(self.nsym):
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
        self.typat:list[int] = []
        for _ in range(self.natom):
            val = bytes2int(wfk.read(4))
            self.typat.append(val)
        #--------------#
        # unpack doubles
        self.kpts:list[np.ndarray] = []
        for _ in range(self.nkpt):
            vec = np.zeros(3)
            vec[0] = bytes2float(wfk.read(8))
            vec[1] = bytes2float(wfk.read(8))
            vec[2] = bytes2float(wfk.read(8))
            self.kpts.append(vec)
        self.occ:list[float] = []
        for _ in range(self.bandtot):
            val = bytes2float(wfk.read(8))
            self.occ.append(val)
        self.tnons:list[np.ndarray] = []
        for _ in range(self.nsym):
            vec = np.zeros(3)
            vec[0] = bytes2float(wfk.read(8))
            vec[1] = bytes2float(wfk.read(8))
            vec[2] = bytes2float(wfk.read(8))
            self.tnons.append(vec)
        self.znucltypat:list[float] = []
        for _ in range(self.ntypat):
            val = bytes2float(wfk.read(8))
            self.znucltypat.append(val)
        self.wtk:list[float] = []           
        for _ in range(self.nkpt):
            val = bytes2float(wfk.read(8))
            self.wtk.append(val)
        #------------------#
        # unpack coordinates
        wfk.read(8)
        self.residm = bytes2float(wfk.read(8))
        self.xred:list[np.ndarray] = []
        for _ in range(self.natom):
            vec = np.zeros(3)
            vec[0] = bytes2float(wfk.read(8))
            vec[1] = bytes2float(wfk.read(8))
            vec[2] = bytes2float(wfk.read(8))
            self.xred.append(vec)
        #------------------------------------#
        # unpack total energy and fermi energy
        self.etotal = bytes2float(wfk.read(8))
        self.fermi = bytes2float(wfk.read(8))
        #-------------#
        # unpack floats
        self.amu:list[float] = []
        for _ in range(self.ntypat):
            val = bytes2float(wfk.read(8))
            self.amu.append(val)
        #----------------------------#
        # unpack reciprocal space info
        wfk.read(8)
        self.kptopt = bytes2int(wfk.read(4))
        self.pawcpxocc = bytes2int(wfk.read(4))
        self.nelect = bytes2float(wfk.read(8))
        self.cellcharge = bytes2float(wfk.read(8))
        self.icoulomb = bytes2int(wfk.read(4))
        rec_latt_ax = bytes2int(wfk.read(4))
        rec_latt_ay = bytes2int(wfk.read(4))
        rec_latt_az = bytes2int(wfk.read(4))
        rec_latt_bx = bytes2int(wfk.read(4))
        rec_latt_by = bytes2int(wfk.read(4))
        rec_latt_bz = bytes2int(wfk.read(4))
        rec_latt_cx = bytes2int(wfk.read(4))
        rec_latt_cy = bytes2int(wfk.read(4))
        rec_latt_cz = bytes2int(wfk.read(4))
        self.kptr_latt = np.array([
            [rec_latt_ax, rec_latt_ay, rec_latt_az],
            [rec_latt_bx, rec_latt_by, rec_latt_bz],
            [rec_latt_cx, rec_latt_cy, rec_latt_cz]
        ]).reshape((3,3))
        rec_latt_ax = bytes2int(wfk.read(4))
        rec_latt_ay = bytes2int(wfk.read(4))
        rec_latt_az = bytes2int(wfk.read(4))
        rec_latt_bx = bytes2int(wfk.read(4))
        rec_latt_by = bytes2int(wfk.read(4))
        rec_latt_bz = bytes2int(wfk.read(4))
        rec_latt_cx = bytes2int(wfk.read(4))
        rec_latt_cy = bytes2int(wfk.read(4))
        rec_latt_cz = bytes2int(wfk.read(4))
        self.kptr_latt_orig = np.array([
            [rec_latt_ax, rec_latt_ay, rec_latt_az],
            [rec_latt_bx, rec_latt_by, rec_latt_bz],
            [rec_latt_cx, rec_latt_cy, rec_latt_cz]
        ]).reshape((3,3))
        shift_orig_x = bytes2float(wfk.read(8))
        shift_orig_y = bytes2float(wfk.read(8))
        shift_orig_z = bytes2float(wfk.read(8))
        self.origin_shift = [shift_orig_x, shift_orig_y, shift_orig_z]
        shift_kx = bytes2float(wfk.read(8))
        shift_ky = bytes2float(wfk.read(8))
        shift_kz = bytes2float(wfk.read(8))
        self.kpt_shift = [shift_kx, shift_ky, shift_kz]
        #-----------------------#
        # unpack pseudopotentials
        wfk.read(4)
        self.title:list[str] = []
        self.znuclpsp:list[float] = []
        self.zionpsp:list[float] = []
        self.pspso:list[int] = []
        self.pspdat:list[int] = []
        self.pspcod:list[int] = []
        self.pspxc:list[int] = []
        self.lmn_size:list[int] = []
        self.md5_pseudos:list[str] = []
        for _ in range(self.npsp):
            wfk.read(4)
            self.title.append(wfk.read(132).decode())
            self.znuclpsp.append(bytes2float(wfk.read(8)))
            self.zionpsp.append(bytes2float(wfk.read(8)))
            self.pspso.append(bytes2int(wfk.read(4)))
            self.pspdat.append(bytes2int(wfk.read(4)))
            self.pspcod.append(bytes2int(wfk.read(4)))
            self.pspxc.append(bytes2int(wfk.read(4)))
            self.lmn_size.append(bytes2int(wfk.read(4)))
            self.md5_pseudos.append(wfk.read(32).decode())
            wfk.read(4)
        #---------------#
        # unpack PAW data
        if self.usepaw == 1:
            wfk.read(4)
            self.nrho:list[int] = []
            for _ in range(self.natom):
                for _ in range(self.nspden):
                    nrhoij = bytes2int(wfk.read(4))
                    self.nrho.append(nrhoij)
            self.cplex = bytes2int(wfk.read(4))
            self.nspden_paw = bytes2int(wfk.read(4))
            wfk.read(8)
            self.irho:list[int] = []
            for i in range(self.natom):
                for _ in range(self.nspden_paw):
                    nrhoij = self.nrho[i]
                    for _ in range(nrhoij):
                        rhoij = bytes2int(wfk.read(4))
                        self.irho.append(rhoij)
            wfk.read(4)
            self.rho:list[float] = []
            for i in range(self.natom):
                    for _ in range(self.nspden_paw):
                        nrhoij = self.nrho[i]
                        for _ in range(nrhoij):
                            rhoij = bytes2float(wfk.read(8))
                            self.rho.append(rhoij)
            wfk.read(4)
        print('WFK header read')
        wfk.close()
    #-----------------------------------------------------------------------------------------------------------------#
    # check for time reversal symmetry
    def _CheckTimeRev(
        self
    ):
        # if system is centrosymmetric, do not double reciprocal symmetry operations
        if True in [np.array_equal(-np.identity(3),mat) for mat in self.symrel]:
            self.time_reversal = False
        else:
            self.time_reversal = True
            print((
            'Noncentrosymmetric system identified, assuming time reversal symmetry\n'
            'To change this, set "check_time_rev" keyword to False when calling ReadWFK() or ReadEigenvalues()'
            ))
    #-----------------------------------------------------------------------------------------------------------------#
    # method to read entire body of abinit version 7 wavefunction file
    def ReadWFK(
        self, energy_level=np.nan, width=np.nan, check_time_rev:bool=True
    )->Generator[wc.WFK, None, None]:
        '''
        Method that constructs WFK objects from ABINIT v10 WFK file.

        Parameters
        ----------
        energy_level : float
        The energy level to pull plane wave coefficients from
        This is relative to the Fermi energy
        If not defined but width is defined, default is 0 Hartree

        width : float
        Defines range about energy_level to search for plane wave coefficients
        Total range is equal to width, so look width/2 above and below energy_level
        If not defined but energy_level is defined, default is 0.005 Hartree
        '''
        # check for time reversal symmetry
        if check_time_rev:
            self._CheckTimeRev()
        else:
            self.time_reversal = False
        # read wfk
        wfk = open(self.filename, 'rb')
        skip = True
        if energy_level is np.nan:
            if width is np.nan:
                skip = False
            else:
                energy_level = 0
        else:
            if width is np.nan:
                width = 0.005
        #-----------#
        # skip header
        wfk.read(468)
        wfk.read(4*(2*self.nkpt + self.nkpt*self.nsppol + self.npsp + 10*self.nsym + self.natom))
        wfk.read(8*(4*self.nkpt + self.bandtot + 3*self.nsym + 2*self.ntypat + 3*self.natom))
        wfk.read(self.npsp*(208))
        if self.usepaw == 1:
            wfk.read(4)
            for _ in range(self.natom):
                for _ in range(self.nspden):
                    _ = wfk.read(4)
            wfk.read(4)
            wfk.read(4)
            wfk.read(8)
            for i in range(self.natom):
                for _ in range(self.nspden_paw):
                    nrhoij = self.nrho[i]
                    for _ in range(nrhoij):
                        _ = wfk.read(4)
            wfk.read(4)
            for i in range(self.natom):
                for _ in range(self.nspden_paw):
                    nrhoij = self.nrho[i]
                    for _ in range(nrhoij):
                        _ = wfk.read(8)
            wfk.read(4)
        #-------------------------------#
        # begin reading wavefunction body
        for i in range(self.nsppol):
            for j in range(self.nkpt):
                print(f'Reading kpoint {j+1} of {self.nkpt}', end='\r')
                if j+1 == self.nkpt:
                    print('\n', end='')
                wfk.read(4)
                npw = bytes2int(wfk.read(4))
                self.nspinor = bytes2int(wfk.read(4))
                nband_temp = bytes2int(wfk.read(4))
                pw_indices = np.zeros((npw,3), dtype=int)
                eigenvalues = np.zeros(nband_temp, dtype=float)
                wfk.read(8)
                for pw in range(npw):
                    kx = bytes2int(wfk.read(4))
                    ky = bytes2int(wfk.read(4))
                    kz = bytes2int(wfk.read(4))
                    pw_indices[pw,:] = [kx, ky, kz]
                wfk.read(8)
                for nband in range(nband_temp):
                    eigenval = bytes2float(wfk.read(8))
                    eigenvalues[nband] = eigenval
                # skip reading coefficients if they energy level of interest is not in band
                if skip:
                    min_val = self.fermi + energy_level - width/2
                    max_val = self.fermi + energy_level + width/2
                    eig_found = False
                    for eigval in eigenvalues:
                        if min_val <= eigval <= max_val:
                            eig_found = True
                            break
                    if eig_found:
                        yield self._ReadCoeffs(
                            wfk,
                            nband_temp,
                            npw,
                            eigenvalues,
                            pw_indices,
                            ind=j
                        )
                    else:
                        wfk.read(nband_temp*8)
                        wfk.read(4)
                        wfk.read(nband_temp*(8 + npw*16))
                # if not skip, read full wavefunction
                else:
                    yield self._ReadCoeffs(
                        wfk,
                        nband_temp,
                        npw,
                        eigenvalues,
                        pw_indices,
                        ind=j
                    )
        print('WFK body read')
        wfk.close()
    #-----------------------------------------------------------------------------------------------------------------#
    # method to read in plane wave coefficients
    def _ReadCoeffs(
        self, wfk_file, nbands, npws, eigs, pw_inds, ind
    )->wc.WFK:
        occupancies = np.zeros((1,nbands), dtype=float)
        coeffs = np.zeros((nbands,npws), dtype=complex)
        for nband in range(nbands):
            occ = bytes2float(wfk_file.read(8))
            occupancies[:,nband] = occ
        wfk_file.read(4)
        for nband in range(nbands):
            wfk_file.read(4)
            cg:np.ndarray = np.zeros((1,npws), dtype=complex)
            for pw in range(npws):
                cg1 = bytes2float(wfk_file.read(8))
                cg2 = bytes2float(wfk_file.read(8))
                cg[:,pw] = cg1 + 1j*cg2
            coeffs[nband,:] = cg
            wfk_file.read(4) 
        return wc.WFK(
            eigenvalues=np.array(eigs), 
            wfk_coeffs=np.array(coeffs),
            pw_indices=np.array(pw_inds),
            kpoints=np.array(self.kpts[ind]),
            nkpt=self.nkpt,
            nbands=self.bands[0],
            ngfftx=self.ngfftx,
            ngffty=self.ngffty,
            ngfftz=self.ngfftz,
            symrel=np.array(self.symrel),
            nsym=self.nsym,
            lattice=self.real_lattice,
            natom=self.natom,
            xred=np.array(self.xred),
            typat=self.typat,
            znucltypat=self.znucltypat,
            fermi_energy=self.fermi,
            non_symm_vecs=np.array(self.tnons),
            time_reversal=self.time_reversal
        )
    #-----------------------------------------------------------------------------------------------------------------#
    # method to read only eigenvalues from body of abinit version 10 wavefunction file
    def ReadEigenvalues(
        self, check_time_rev:bool=True
    )->Generator[wc.WFK, None, None]:
        '''
        Method that constructs WFK objects from ABINIT v10 WFK file.
        '''
        # check for time reversal symmetry
        if check_time_rev:
            self._CheckTimeRev()
        else:
            self.time_reversal = False
        # read wfk
        wfk = open(self.filename, 'rb')
        #-----------#
        # skip header
        wfk.read(468)
        wfk.read(4*(2*self.nkpt + self.nkpt*self.nsppol + self.npsp + 10*self.nsym + self.natom))
        wfk.read(8*(4*self.nkpt + self.bandtot + 3*self.nsym + 2*self.ntypat + 3*self.natom))
        wfk.read(self.npsp*(208))
        if self.usepaw == 1:
            wfk.read(4)
            for _ in range(self.natom):
                for _ in range(self.nspden):
                    _ = wfk.read(4)
            wfk.read(4)
            wfk.read(4)
            wfk.read(8)
            for i in range(self.natom):
                for _ in range(self.nspden_paw):
                    nrhoij = self.nrho[i]
                    for _ in range(nrhoij):
                        _ = wfk.read(4)
            wfk.read(4)
            for i in range(self.natom):
                for _ in range(self.nspden_paw):
                    nrhoij = self.nrho[i]
                    for _ in range(nrhoij):
                        _ = wfk.read(8)
            wfk.read(4)
        #-------------------------------#
        # begin reading wavefunction body
        for i in range(self.nsppol):
            for j in range(self.nkpt):
                print(f'Reading kpoint {j+1} of {self.nkpt}', end='\r')
                if j+1 == self.nkpt:
                    print('\n', end='')
                wfk.read(4)
                npw = bytes2int(wfk.read(4))
                self.nspinor = bytes2int(wfk.read(4))
                nband_temp = bytes2int(wfk.read(4))
                eigenvalues = np.zeros((1,nband_temp), dtype=float)
                wfk.read(8)
                wfk.read(12*npw)
                wfk.read(8)
                #------------------------------------------------------------------#
                # only need eigenvalues for Fermi surface, skip over everything else
                for nband in range(nband_temp):
                    eigenval = bytes2float(wfk.read(8))
                    eigenvalues[:,nband] = eigenval
                wfk.read(nband_temp*8)
                wfk.read(4)
                wfk.read(nband_temp*(8 + npw*16))
                yield wc.WFK(
                    eigenvalues=np.array(eigenvalues), 
                    kpoints=np.array(self.kpts[j]),
                    nkpt=self.nkpt,
                    nbands=self.bands[0],
                    ngfftx=self.ngfftx,
                    ngffty=self.ngffty,
                    ngfftz=self.ngfftz,
                    symrel=np.array(self.symrel),
                    nsym=self.nsym,
                    lattice=self.real_lattice,
                    natom=self.natom,
                    xred=np.array(self.xred),
                    typat=self.typat,
                    znucltypat=self.znucltypat,
                    fermi_energy=self.fermi,
                    non_symm_vecs=np.array(self.tnons),
                    time_reversal=self.time_reversal
                )
        print('WFK body read')
        wfk.close()

#---------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------- END CLASS -----------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#

class AbinitNetCDF():
    def __init__(
        self, filename:str
    ):
        print('WFK found with netCDF4 format')
        self.filename = filename
        self.dataset = self._GetDataset()
        self.kpts:np.ndarray = self.dataset.variables['reduced_coordinates_of_kpoints'][:]
        self.nkpt:int = self.dataset.dimensions['number_of_kpoints'].size
        self.xred:np.ndarray = self.dataset.variables['reduced_atom_positions'][:]
        self.natom:int = self.dataset.dimensions['number_of_atoms'].size
        self.tnons:np.ndarray = self.dataset.variables['reduced_symmetry_translations'][:]
        self.symrel:np.ndarray = self.dataset.variables['reduced_symmetry_matrices'][:]
        self.symrel = np.array([op.T for op in self.symrel])
        self.nsym:int = self.dataset.dimensions['number_of_symmetry_operations'].size
        self.fermi:float = self.dataset['fermi_energy'][:]
        self.real_lattice:np.ndarray = self.dataset['primitive_vectors'][:]
        self.real_lattice *= 0.529177
        self.ngfftx:int = self.dataset.dimensions['number_of_grid_points_vector1'].size
        self.ngffty:int = self.dataset.dimensions['number_of_grid_points_vector2'].size
        self.ngfftz:int = self.dataset.dimensions['number_of_grid_points_vector3'].size
        self.nband:int = int(self.dataset.dimensions['bantot'].size / self.nkpt)
        self.typat:list = self.dataset.variables['atom_species'][:].tolist()
        self.zion:np.ndarray = self.dataset.variables['atomic_numbers'][:]
        _all_ions = [int(self.zion[i-1]) for i in self.typat]
        self.znucltypat:list = []
        for i in _all_ions:
            if i not in self.znucltypat:
                self.znucltypat.append(i)
        self.bands = [self.nband]
#---------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ METHODS ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#
    # create netCDF dataset
    def _GetDataset(
        self
    ):
        from netCDF4 import Dataset
        return Dataset(self.filename,format='NETCDF4')
    #-----------------------------------------------------------------------------------------------------------------#
    # check for time reversal symmetry
    def _CheckTimeRev(
        self
    ):
        # if system is centrosymmetric, do not double reciprocal symmetry operations
        if True in [np.array_equal(-np.identity(3),mat) for mat in self.symrel]:
            self.time_reversal = False
        else:
            self.time_reversal = True
            print((
            'Noncentrosymmetric system identified, assuming time reversal symmetry\n'
            'To change this, set "check_time_rev" keyword to False when calling ReadWFK() or ReadEigenvalues()'
            ))
    #-----------------------------------------------------------------------------------------------------------------#
    # fetch wavefunction coefficients from netcdf dataset
    def ReadWFK(
        self, energy_level:float=np.nan, width:float=np.nan, check_time_rev:bool=True
    )->Generator[wc.WFK, None, None]:
        # check for time reversal symmetry
        if check_time_rev:
            self._CheckTimeRev()
        else:
            self.time_reversal = False
        eigs:np.ndarray = self.dataset.variables['eigenvalues'][:][0]
        pw_inds:np.ndarray = self.dataset.variables['reduced_coordinates_of_plane_waves'][:]
        # restructure wavefunction coefficients
        for kpt in range(self.nkpt):
            print(f'Reading kpoint {kpt+1} of {self.nkpt}', end='\r')
            kpt_coeffs = []
            # format plane wave indices
            kpt_pw_inds = pw_inds[kpt]
            kpt_pw_inds = kpt_pw_inds[~kpt_pw_inds.mask]
            kpt_pw_inds = kpt_pw_inds.reshape((-1,3))
            kpt_pw_inds = np.ma.getdata(kpt_pw_inds)
            coeffs = self.dataset.variables['coefficients_of_wavefunctions'][0,kpt]
            for band in coeffs:
                band = band[0]
                band = band[~band.mask]
                band = band.reshape((-1,2))
                band = np.ma.getdata(band)
                kpt_coeffs.append(band[:,0] + 1j*band[:,1])
            yield wc.WFK(
                eigenvalues=eigs[kpt,:], 
                wfk_coeffs=np.array(kpt_coeffs),
                pw_indices=kpt_pw_inds,
                kpoints=self.kpts[kpt],
                nkpt=self.nkpt,
                nbands=self.nband,
                ngfftx=self.ngfftx,
                ngffty=self.ngffty,
                ngfftz=self.ngfftz,
                symrel=self.symrel,
                nsym=self.nsym,
                lattice=self.real_lattice,
                natom=self.natom,
                xred=self.xred,
                typat=self.typat,
                znucltypat=self.znucltypat,
                fermi_energy=self.fermi,
                non_symm_vecs=self.tnons,
                time_reversal=self.time_reversal
            )
    #-----------------------------------------------------------------------------------------------------------------#
    # fetch eigenvalues from netcdf dataset
    def ReadEigenvalues(
        self, energy_level:float=np.nan, width:float=np.nan, check_time_rev:bool=True
    )->Generator[wc.WFK, None, None]:
        # check for time reversal symmetry
        if check_time_rev:
            self._CheckTimeRev()
        else:
            self.time_reversal = False
        eigs:np.ndarray = self.dataset.variables['eigenvalues'][:][0]
        for i, kpt in enumerate(self.kpts):
            print(f'Reading kpoint {i+1} of {self.nkpt}', end='\r')
            yield wc.WFK(
                eigenvalues=eigs[i,:], 
                kpoints=kpt,
                nkpt=self.nkpt,
                nbands=self.nband,
                ngfftx=self.ngfftx,
                ngffty=self.ngffty,
                ngfftz=self.ngfftz,
                symrel=self.symrel,
                nsym=self.nsym,
                lattice=self.real_lattice,
                natom=self.natom,
                xred=self.xred,
                typat=self.typat,
                znucltypat=self.znucltypat,
                fermi_energy=self.fermi,
                non_symm_vecs=self.tnons,
                time_reversal=self.time_reversal
            )        

#---------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------- END CLASS -----------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#

def AbinitWFK(
    filename:str
)->Union[Abinit10WFK, Abinit7WFK, AbinitNetCDF]:
    '''
    A function for identifying ABINIT version and constructing respective AbinitWFK object.
    '''
    if filename.endswith('.nc'):
        return AbinitNetCDF(filename)
    wfk = open(filename, 'rb')
    wfk.read(4)
    version = wfk.read(6).decode()
    wfk.close()
    if version.strip().split('.')[0] == '7':
        return Abinit7WFK(filename)
    elif version.strip().split('.')[0] == '10':
        return Abinit10WFK(filename)
    else:
        print(f'Version {version} may not be supported, attempting to analyze with v10 procedure.')
        return Abinit10WFK(filename)