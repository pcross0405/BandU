import struct
import numpy as np
from typing import Generator
import sys
from wfk_class import WFK
np.set_printoptions(threshold=sys.maxsize)

def bytes2float(bin_data):
    return struct.unpack('<d', bin_data)[0]

def bytes2int(bin_data):
    return int.from_bytes(bin_data, 'little', signed=True)

class AbinitWFK():
    '''
    A class for storing all variables from ABINIT WFK file
    '''
    def __init__(self, filename):
        self.filename = filename
        self._ReadHeader(self.filename)
#---------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ METHODS ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#
    # method to read wavefunction header
    def _ReadHeader(
        self, filename
    )->None:
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
    def ReadWFK(
        self
    )->Generator:
        '''
        Method that constructs WFK objects from ABINIT WFK file.
        '''
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
                yield WFK(
                            eigenvalues=eigenvalues, 
                            wfk_coeffs=coeffs,
                            kpoints=kpoints,
                            nbands=self.bands[0],
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
                            fermi_energy=self.fermi
                )
        print('WFK body read')
        wfk.close()
    #-----------------------------------------------------------------------------------------------------------------#
    # method to read only eigenvalues from body of wavefunction file
    def _ReadEigenvalues(
        self
    )->Generator:
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