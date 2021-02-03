import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

from cosmosis.dataset import *

from abc import ABC, abstractmethod
import os, re, random, h5py, pickle

import numpy as np

from scipy import spatial as sp
from scipy.io import loadmat

from rdkit import Chem

from torch.utils.data import Dataset, IterableDataset, ConcatDataset
from torch import as_tensor, cat


class Molecule(ABC):
    """A class for creating a rdmol obj and coulomb matrix from a smile.  
    Subclass and implement load_data()"""
    
    atomic_n = {'C': 6, 'H': 1, 'N': 7, 'O': 8, 'F': 9}
    
    def __init__(self, in_dir):
        self.load_data(in_dir)
        self.rdmol_from_smile(self.smile)
        self.create_adjacency(self.rdmol)
        self.create_distance(self.xyz)
        self.create_coulomb(self.distance, self.xyz)
        
    @abstractmethod
    def __repr__(self):
        return self.mol_id
        
    @abstractmethod
    def load_data(self):
        self.smile = '' 
        self.n_atoms = 0    
        self.xyz = []  # [['atom_type',x,y,z],...]
        
    def open_file(self, in_file):
        with open(in_file) as f:
            data = []
            for line in f.readlines():
                data.append(line)
            return data
        
    def rdmol_from_smile(self, smile):
        self.rdmol = Chem.AddHs(Chem.MolFromSmiles(smile))
    
    def create_adjacency(self, rdmol):
        """use the rdmol mol block adjacency list to create a nxn symetric matrix with 0, 1, 2 or
        3 for bond type where n is the indexed atom list for the molecule"""
        block = Chem.MolToMolBlock(rdmol)
        self.adjacency = np.zeros((self.n_atoms, self.n_atoms), dtype='float32')
        block = block.split('\n')
        for b in block[:-2]:
            line = ''.join(b.split())
            if len(line) == 4:
                # shift -1 to index from zero
                self.adjacency[(int(line[0])-1),(int(line[1])-1)] = int(line[2]) 
                # create bi-directional connection
                self.adjacency[(int(line[1])-1),(int(line[0])-1)] = int(line[2]) 
             
    def create_distance(self, xyz):
        m = np.zeros((len(xyz), 3))
        # fix the scientific notation
        for i, atom in enumerate(xyz):
            m[i,:] = [float(np.char.replace(x, '*^', 'e')) for x in atom[1:4]] 
        self.distance = sp.distance.squareform(sp.distance.pdist(m)).astype('float32')
      
    def create_coulomb(self, distance, xyz, sigma=1):
        """creates coulomb matrix obj attr.  set sigma to False to turn off random sorting.  
        sigma = stddev of gaussian noise.
        https://papers.nips.cc/paper/4830-learning-invariant-representations-of-\
        molecules-for-atomization-energy-prediction"""
        atoms = []
        for atom in xyz:
            atoms.append(Molecule.atomic_n[atom[0]]) 
        atoms = np.asarray(atoms, dtype='float32')
        qmat = atoms[None, :]*atoms[:, None]
        idmat = np.linalg.inv(distance)
        np.fill_diagonal(idmat, 0)
        coulomb = qmat@idmat
        np.fill_diagonal(coulomb, 0.5 * atoms ** 2.4)
        if sigma:  
            self.coulomb = self.sort_permute(coulomb, sigma)
        else:  
            self.coulomb = coulomb
    
    def sort_permute(self, matrix, sigma):
        norm = np.linalg.norm(matrix, axis=1)
        noised = np.random.normal(norm, sigma)
        indexlist = np.argsort(noised)
        indexlist = indexlist[::-1]  # invert
        return matrix[indexlist][:,indexlist]
    
class QM9Mol(Molecule):
    
    properties = ['A','B','C','mu','alpha','homo','lumo', 
                  'gap','r2','zpve','U0','U','H','G','Cv']
    
    def __repr__(self):
        return self.in_file[:-4]
    
    def load_data(self, in_file):
        """load from the .xyz files of the qm9 dataset
        (http://quantum-machine.org/datasets/)
        
        """
        self.in_file = in_file
        xyz = self.open_file(in_file)
        self.smile = xyz[-2]
        self.n_atoms = int(xyz[0])
        
        properties = xyz[1].strip().split('\t')[1:] # [float,...]
        for i, p in enumerate(properties):
            setattr(self, QM9Mol.properties[i], p)
            
        self.xyz = []
        for atom in xyz[2:self.n_atoms+2]:
            self.xyz.append(atom.strip().split('\t')) # [['atom_type',x,y,z,mulliken],...]
            
        self.mulliken = []
        for atom in self.xyz:
            m = np.reshape(np.asarray(np.char.replace(atom[4], '*^', 'e'), 
                                                          dtype=np.float32), -1)
            self.mulliken.append(m)
        self.mulliken = np.concatenate(self.mulliken, axis=0)
       
    
class QM9(CDataset):
    """http://quantum-machine.org/datasets/
   
    dsgdb9nsd.xyz.tar.bz2    - 133885 molecules with properties in XYZ-like format
    dsC7O2H10nsd.xyz.tar.bz2 - 6095 isomers of C7O2H10 with properties in XYZ-like format
    validation.txt           - 100 randomly drawn molecules from the 133885 set with 
                               enthalpies of formation
    uncharacterized.txt      - 3054 molecules from the 133885 set that failed a consistency check
    atomref.txt              - Atomic reference data
    readme.txt               - Documentation

    1          Number of atoms na
    2          Properties 1-17 (see below)
    3,...,na+2 Element type, coordinate (x,y,z) (Angstrom), and Mulliken partial charge (e) of atom
    na+3       Frequencies (3na-5 or 3na-6)
    na+4       SMILES from GDB9 and for relaxed geometry
    na+5       InChI for GDB9 and for relaxed geometry

    The properties stored in the second line of each file:

    I.  Property  Unit         Description
    --  --------  -----------  --------------
     1  tag       -            "gdb9"; string constant to ease extraction via grep
     2  index     -            Consecutive, 1-based integer identifier of molecule
     3  A         GHz          Rotational constant A
     4  B         GHz          Rotational constant B
     5  C         GHz          Rotational constant C
     6  mu        Debye        Dipole moment
     7  alpha     Bohr^3       Isotropic polarizability
     8  homo      Hartree      Energy of Highest occupied molecular orbital (HOMO)
     9  lumo      Hartree      Energy of Lowest occupied molecular orbital (LUMO)
    10  gap       Hartree      Gap, difference between LUMO and HOMO
    11  r2        Bohr^2       Electronic spatial extent
    12  zpve      Hartree      Zero point vibrational energy
    13  U0        Hartree      Internal energy at 0 K
    14  U         Hartree      Internal energy at 298.15 K
    15  H         Hartree      Enthalpy at 298.15 K
    16  G         Hartree      Free energy at 298.15 K
    17  Cv        cal/(mol K)  Heat capacity at 298.15 K
    
    https://www.nature.com/articles/sdata201422
    Quantum chemistry structures and properties of 134 kilo molecules
    
    https://arxiv.org/abs/1809.02723
    Deep Neural Network Computes Electron Densities and Energies of a Large Set of 
    Organic Molecules Faster than Density Functional Theory (DFT)
    
    https://arxiv.org/abs/1908.00971
    Physical machine learning outperforms "human learning" in Quantum Chemistry
    
    """
    LOW_CONVERGENCE = [21725,87037,59827,117523,128113,129053,129152, 
                       129158,130535,6620,59818]
    
    properties = ['A','B','C','mu','alpha','homo','lumo', 
                  'gap','r2','zpve','U0','U','H','G','Cv']
    
    def __init__(self, 
                 in_dir='./data/qm9/qm9.xyz/', 
                 n=133885, 
                 features=[],
                 embed=[],
                 targets=[], 
                 pad=29,
                 filter_on=False,
                 use_pickle='qm9_datadic.p'):
        """pad = length of longest molecule that all molecules will be padded to
        features/target = QM9.properties,'coulomb','mulliken','adjacency',QM9Mol.attr
        filter_on = ('attr', 'test', 'value')
        n = non random subset selection (for testing)
        embed=[('feature',voc,vec,padding_idx,param.requires_grad)] 
        use_pickle = False/'qm9_datadic.p' (if file exists loads, if not creates and saves)
        """
        self.features, self.targets, self.pad = features, targets, pad
        self.datadic = self.load_data(in_dir, n, filter_on, use_pickle)
        self.ds_idx = list(self.datadic.keys())
        self.embed = embed 
    
    def __getitem__(self, i):
        
        mol = self.datadic[i]
       
        x_con = self.get_features(mol, self.features, np.float32)
        targets = self.get_features(mol, self.targets, np.float64)
        x_cat = self.get_features(mol, [self.embed[0][0]], np.int64)
        
        return as_tensor(x_con), [as_tensor(x_cat)], as_tensor(targets)
        
    def get_features(self, mol, features, dtype):
        data = []
        for feature in features:
            flat = np.reshape(np.asarray(getattr(mol, feature), dtype=dtype), -1)
            if self.pad:
                if feature in ['coulomb','adjacency','distance']: 
                    out = np.pad(flat, (0, self.pad**2-len(getattr(mol, feature))**2))
                elif feature == 'mulliken':
                    out = np.pad(getattr(mol, feature), (0, self.pad-len(getattr(mol, feature))))
                elif feature in QM9.properties: 
                    out = flat
                else: 
                    out = flat
            data.append(out)
        if len(data) == 0:
            return data
        else:
            return np.concatenate(data)
       
    def __len__(self):
        return len(self.ds_idx)
    
    def open_file(self, in_file):
        with open(in_file) as f:
            data = []
            for line in f.readlines():
                data.append(line)
            return data
        
    def load_data(self, in_dir, n, filter_on, use_pickle): 
        
        if use_pickle and os.path.exists('./data/qm9/'+use_pickle):
            print('loading QM9 datadic from a pickled copy...')
            datadic = pickle.load(open('./data/qm9/'+use_pickle, 'rb'))
        else:
            print('creating QM9 dataset...')
            i = 0
            datadic = {}
            for filename in sorted(os.listdir(in_dir)):
                if filename.endswith('.xyz'):
                    i += 1
                    datadic[int(filename[-10:-4])] = QM9Mol(in_dir+filename)
                    
                    if filter_on:
                        val = self.get_features(datadic[int(filename[-10:-4])], 
                                                     [filter_on[0]], np.float32)
                        val = np.array2string(val, precision=4, floatmode='maxprec')[1:-1]
                     
                        if not eval(val+filter_on[1]+filter_on[2]):
                            del datadic[int(filename[-10:-4])]
                        
                    if i % 10000 == 1: 
                        print('QM9 molecules scanned: ', i)
                        print('QM9 molecules created: ', len(datadic))
                    if len(datadic) > n - 1:
                        break
                       
            unchar = self.get_uncharacterized()
            for mol in unchar: 
                try: del datadic[mol]
                except: continue
            print('total QM9 molecules created:', len(datadic))
            
            if use_pickle:
                print('pickling a copy of the QM9 datadic...')        
                pickle.dump(datadic, open('./data/qm9/'+use_pickle, 'wb'))
                
        return datadic
    
    def get_uncharacterized(self, in_file='./data/qm9/uncharacterized.txt'):
        """uncharacterized.txt - 3054 molecules from the 133885 set that failed a 
        consistency check.  Returns a list of ints of the 3054 molecules (datadic keys)"""
        data = self.open_file(in_file)
        unchar = []
        for mol in data[8:]:
            for m in mol.strip().split():
                if m.isdigit():
                    unchar.append(int(m))
        return unchar
        
        
class ANI1x(CDataset):
    """https://www.nature.com/articles/s41597-020-0473-z#Sec11
    https://github.com/aiqm/ANI1x_datasets
    https://springernature.figshare.com/articles/dataset/ANI-1x_Dataset_Release/10047041
    
    The source dataset is organized:
    [molecular formula][conformation index][feature,feature,...]
    
    It is indexed by a molecular formula and conformation index
    
    This dataset is a pytorch and cosmosis dataset:
    Returns [feature,feature,...,padding], [target,target,...]
    
    Longest molecule is 63 atoms
    
    select:
        molecular formula
        conformation
        features
        target
        padding
        data file location
        embedding options
    
    criterion = the feature used to select the conformation
    conformation = logic used on the criterion feature
        'min' - choose the index with the lowest value
        'max' - choose the index with the highest value
        'random' - choose the index randomly 
        int - choose the index int
    features = ['feature','feature',...]
    targets = ['feature',...]
    pad = int/False
    embed = [('feature',voc,vec,padding_idx,train)] 
    
    Na = number of atoms, Nc = number of conformations
    Atomic Positions ‘coordinates’ Å float32 (Nc, Na, 3)
    Atomic Numbers   ‘atomic_numbers’ — uint8 (Na)
    Total Energy     ‘wb97x_dz.energy’ Ha float64 (Nc)
                     ‘wb97x_tz.energy’ Ha float64 (Nc)  
                     ‘ccsd(t)_cbs.energy’ Ha float64 (Nc)
    HF Energy        ‘hf_dz.energy’ Ha float64 (Nc)
                     ‘hf_tz.energy’ Ha float64 (Nc)
                     ‘hf_qz.energy’ Ha float64 (Nc)
    NPNO-CCSD(T)     ‘npno_ccsd(t)_dz.corr_energy’ Ha float64 (Nc)
    Correlation      ‘npno_ccsd(t)_tz.corr_energy’ Ha float64 (Nc)
    Energy           ‘npno_ccsd(t)_qz.corr_energy’ Ha float64 (Nc)
    MP2              ‘mp2_dz.corr_energy’ Ha float64 (Nc)
    Correlation      ‘mp2_tz.corr_energy’ Ha float64 (Nc)
    Energy           ‘mp2_qz.corr_energy’ Ha float64 (Nc)
    Atomic Forces    ‘wb97x_dz.forces’ Ha/Å float32 (Nc, Na, 3)
                     ‘wb97x_tz.forces’ Ha/Å float32 (Nc, Na, 3)
    Molecular        ‘wb97x_dz.dipole’ e Å float32 (Nc, 3)
    Electric         ‘wb97x_tz.dipole’ e Å float32 (Nc, 3)
    Moments          ‘wb97x_tz.quadrupole’ e AA2 (Nc, 6)
    Atomic           ‘wb97x_dz.cm5_charges’ e float32 (Nc, Na)
    Charges          ‘wb97x_dz.hirshfeld_charges’ e float32 (Nc, Na)
                     ‘wb97x_tz.mbis_charges’ e float32 (Nc, Na)
    Atomic           ‘wb97x_tz.mbis_dipoles’ a.u. float32 (Nc, Na)
    Electric         ‘wb97x_tz.mbis_quadrupoles’ a.u. float32 (Nc, Na)
    Moments          ‘wb97x_tz.mbis_octupoles’ a.u. float32 (Nc, Na)
    Atomic Volumes   ‘wb97x_tz.mbis_volumes’ a.u. float32 (Nc, Na)
    
    distance = (Na, Na) distance matrix constructed from 'coordinates' feature
    """
    features = ['atomic_numbers', 'ccsd(t)_cbs.energy', 'coordinates', 'hf_dz.energy',
                'hf_qz.energy', 'hf_tz.energy', 'mp2_dz.corr_energy', 'mp2_qz.corr_energy',
                'mp2_tz.corr_energy', 'npno_ccsd(t)_dz.corr_energy', 'npno_ccsd(t)_tz.corr_energy',
                'tpno_ccsd(t)_dz.corr_energy', 'wb97x_dz.cm5_charges', 'wb97x_dz.dipole', 
                'wb97x_dz.energy', 'wb97x_dz.forces', 'wb97x_dz.hirshfeld_charges', 
                'wb97x_dz.quadrupole', 'wb97x_tz.dipole', 'wb97x_tz.energy', 'wb97x_tz.forces',
                'wb97x_tz.mbis_charges', 'wb97x_tz.mbis_dipoles', 'wb97x_tz.mbis_octupoles',
                'wb97x_tz.mbis_quadrupoles', 'wb97x_tz.mbis_volumes']
    
    atomic_n = {0:0, 1:1, 6:2, 7:3, 8:4, 9:5}
    
    def __init__(self, features=['distance'], targets=[], pad=63,
                 embed=[('atomic_numbers',9,16,0,True)], criterion=None, 
                 conformation='random', in_file='./data/ani1x/ani1x-release.h5'):
        
        self.features, self.targets = features, targets
        self.pad, self.embed, self.in_file  = pad, embed, in_file
        self.conformation, self.criterion = conformation, criterion

        self.datadic = self.load_data()
        self.ds_idx = list(self.datadic.keys())
    
    def __getitem__(self, i):
        ci = self.get_conformation_index(self.datadic[i])
        
        def get_features(features, dtype):
            data = []
            for f in features:
                #(Na)
                if f in ['atomic_numbers']:
                    out = np.reshape(self.datadic[i][f], -1).astype(dtype)
                    if self.pad:
                        out = np.pad(out, (0, (self.pad - out.shape[0])))          
                #(Nc, Na)    
                elif f in ['wb97x_dz.cm5_charges','wb97x_dz.hirshfeld_charges',
                           'wb97x_tz.mbis_charges','wb97x_tz.mbis_dipoles',
                           'wb97x_tz.mbis_quadrupoles','wb97x_tz.mbis_octupoles',
                           'wb97x_tz.mbis_volumes']:
                    out = np.reshape(self.datadic[i][f][ci], -1).astype(dtype)
                    if self.pad:
                        out = np.pad(out, (0, (self.pad - out.shape[0])))        
                #(Nc, Na, 3)   
                elif f in ['coordinates','wb97x_dz.forces','wb97x_dz.forces']:
                    out = np.reshape(self.datadic[i][f][ci], -1).astype(dtype)
                    if self.pad:
                        out = np.pad(out, (0, (self.pad*3 - out.shape[0])))
                #(Na, Na)
                elif f in ['distance']:
                    out = self.datadic[i]['coordinates'][ci]
                    out = sp.distance.squareform(sp.distance.pdist(out))
                    out = np.reshape(out, -1).astype(dtype)
                    if self.pad:
                        out = np.pad(out, (0, (self.pad*self.pad - out.shape[0])))
                #(Nc, 6), (Nc, 3), (Nc)
                else:
                    out = np.reshape(self.datadic[i][f][ci], -1).astype(dtype)   
                data.append(out)
            if len(data) == 0:
                return data
            else: 
                return np.concatenate(data)
            
        x_cat = []
        if len(self.embed) > 0:
            molecule = get_features([self.embed[0][0]], 'int64')
            idx = []
            for atom in molecule:
                idx.append(ANI1x.atomic_n[atom])
            x_cat.append(as_tensor(np.asarray(idx, 'int64')))
            
        x_con = get_features(self.features, 'float32')
        
        targets = get_features(self.targets, 'float64')
            
        return as_tensor(x_con), x_cat, as_tensor(targets)
    
    def __len__(self):
        return len(self.ds_idx)
    
    def load_data(self):
        """data_keys = ['wb97x_dz.energy','wb97x_dz.forces'] 
        # Original ANI-1x data (https://doi.org/10.1063/1.5023802)
        data_keys = ['wb97x_tz.energy','wb97x_tz.forces'] 
        # CHNO portion of the data set used in AIM-Net (https://doi.org/10.1126/sciadv.aav6490)
        data_keys = ['ccsd(t)_cbs.energy'] 
        # The coupled cluster ANI-1ccx data set (https://doi.org/10.1038/s41467-019-10827-4)
        data_keys = ['wb97x_dz.dipoles'] 
        # A subset of this data was used for training the ACA charge model 
        (https://doi.org/10.1021/acs.jpclett.8b01939)
        
        ragged dataset each mol has all keys and nan for missing values
        throws out the mol if any of the feature values or criterion feature values are missing
        """
        structure_count = 0
        if len(self.embed) > 0:
            attributes = self.features+self.targets+[self.embed[0][0]]
        else:
            attributes = self.features+self.targets
        if self.criterion != None and self.criterion not in attributes:
            attributes.append(self.criterion)
        datadic = {}
        with h5py.File(self.in_file, 'r') as f:
            for mol in f.keys():
                nan = False
                while not nan:  # if empty values break out and del mol
                    data = {}
                    for attr in attributes:
                        if attr == 'distance':
                            attr = 'coordinates'
                        if np.isnan(f[mol][attr][()]).any():
                            nan = True
                        else:
                            data[attr] = f[mol][attr][()]
                            datadic[mol] = data
                            structure_count += 1
                    break
                if nan: 
                    try: del datadic[mol]
                    except: pass
        print('structures loaded: ', structure_count)               
        return datadic
    
    def get_conformation_index(self, mol):
        """each molecular formula (mol) may have many different isomers
        select the conformation based on some criterion (attribute value)
        """
        if self.criterion == None:
            criterion = self.targets[0]
        else:
            criterion = self.criterion
            
        ci = 0        
        if isinstance(self.conformation, int):
            ci = self.conformation
        elif self.conformation == 'random':
            ci = random.randrange(mol[criterion].shape[0])
        elif self.conformation == 'max':
            ci = np.argmax(mol[criterion], axis=0)
        elif self.conformation == 'min':
            ci = np.argmin(mol[criterion], axis=0)
        
        return ci
                
class QM7X(CDataset):
    """QM7-X: A comprehensive dataset of quantum-mechanical properties spanning 
    the chemical space of small organic molecules
    https://arxiv.org/abs/2006.15139
    https://zenodo.org/record/3905361
    
    decompress the .xz files in qchem/data/qm7x/
    tar xvf *000.xz
    
    1000.hdf5 6.5 GB
    2000.hdf5 8.8 GB
    3000.hdf5 16.9 GB
    4000.hdf5 12.4 GB
    5000.hdf5 9.8 GB
    6000.hdf5 17.2 GB
    7000.hdf5 9.8 GB
    8000.hdf5 0.8 GB
    
    A description of the structure generation procedure is available in the paper 
    related to this dataset.  Each HDF5 file contains information about the molecular 
    properties of equilibrium and non-equilibrium   conformations of small molecules
    composed of up to seven heavy atoms (C, N, O, S, Cl). For instance, you can access
    to the information saved in the 1000.hdf5 file as,

    fDFT = h5py.File('1000.hdf5', 'r')
    fDFT[idmol]: idmol, ID number of molecule (e.g., '1', '100', '94')
    fDFT[idmol][idconf]: idconf, ID configuration (e.g., 'Geom-m1-i1-c1-opt', 'Geom-m1-i1-c1-50')

    The idconf label has the general form "Geom-mr-is-ct-u", were r enumerated the
    SMILES strings, s the stereoisomers excluding conformers, t the considered
    (meta)stable conformers, and u the optimized/displaced structures; u = opt
    indicates the DFTB3+MBD optimized structures and u = 1,...,100 enumerates
    the displaced non-equilibrium structures. Note that these indices are not
    sorted according to their PBE0+MBD relative energies.

    Then, for each structure (i.e., idconf), you will find the following properties:

    -'atNUM': Atomic numbers (N)
    -'atXYZ': Atoms coordinates [Ang] (Nx3)
    -'sRMSD': RMSD to optimized structure [Ang] (1)
    -'sMIT': Momente of inertia tensor [amu.Ang^2] (9)

    -'ePBE0+MBD': Total PBE0+MBD energy [eV] (1)
    -'eDFTB+MBD': Total DFTB+MBD energy [eV] (1)
    -'eAT': PBE0 atomization energy [eV] (1)
    -'ePBE0': PBE0 energy [eV] (1)
    -'eMBD': MBD energy [eV] (1)
    -'eTS': TS dispersion energy [eV] (1)
    -'eNN': Nuclear-nuclear repulsion energy [eV] (1)
    -'eKIN': Kinetic energy [eV] (1)
    -'eNE': Nuclear-electron attracttion [eV] (1)
    -'eEE': Classical coulomb energy (el-el) [eV] (1)
    -'eXC': Exchange-correlation energy [eV] (1)
    -'eX': Exchange energy [eV] (1)
    -'eC': Correlation energy [eV] (1)
    -'eXX': Exact exchange energy [eV] (1)
    -'eKSE': Sum of Kohn-Sham eigenvalues [eV] (1)
    -'KSE': Kohn-Sham eigenvalues [eV] (depends on the molecule)
    -'eH': HOMO energy [eV] (1)
    -'eL': LUMO energy [eV] (1)
    -'HLgap': HOMO-LUMO gap [eV] (1)
    -'DIP': Total dipole moment [e.Ang] (1)
    -'vDIP': Dipole moment components [e.Ang] (3)
    -'vTQ': Total quadrupole moment components [e.Ang^2] (3)
    -'vIQ': Ionic quadrupole moment components [e.Ang^2] (3)
    -'vEQ': Electronic quadrupole moment components [eAng^2] (3)
    -'mC6': Molecular C6 coefficient [hartree.bohr^6] (computed using SCS) (1)
    -'mPOL': Molecular polarizability [bohr^3] (computed using SCS) (1)
    -'mTPOL': Molecular polarizability tensor [bohr^3] (9)

    -'totFOR': Total PBE0+MBD atomic forces (unitary forces cleaned) [eV/Ang] (Nx3)
    -'vdwFOR': MBD atomic forces [eV/Ang] (Nx3)
    -'pbe0FOR': PBE0 atomic forces [eV/Ang] (Nx3)
    -'hVOL': Hirshfeld volumes [bohr^3] (N)
    -'hRAT': Hirshfeld ratios (N)
    -'hCHG': Hirshfeld charges [e] (N)
    -'hDIP': Hirshfeld dipole moments [e.bohr] (N)
    -'hVDIP': Components of Hirshfeld dipole moments [e.bohr] (Nx3)
    -'atC6': Atomic C6 coefficients [hartree.bohr^6] (N)
    -'atPOL': Atomic polarizabilities [bohr^3] (N)
    -'vdwR': van der Waals radii [bohr] (N)
    
    seletor = list of regular expression strings (attr) for searching 
        and selecting idconf keys.  
        returns datadic[idmol] = [idconf,idconf,...]
        idconf, ID configuration (e.g., 'Geom-m1-i1-c1-opt', 'Geom-m1-i1-c1-50')
         
    """
    set_ids = ['1000','2000','3000','4000','5000','6000','7000','8000']
    
    features = ['DIP','HLgap','KSE','atC6','atNUM','atPOL','atXYZ','eAT', 
                'eC','eDFTB+MBD','eEE','eH','eKIN','eKSE','eL','eMBD','eNE', 
                'eNN','ePBE0','ePBE0+MBD','eTS','eX','eXC','eXX','hCHG', 
                'hDIP','hRAT','hVDIP','hVOL','mC6','mPOL','mTPOL','pbe0FOR', 
                'sMIT','sRMSD','totFOR','vDIP','vEQ','vIQ','vTQ','vdwFOR','vdwR']
    
    atomic_n = {0:0, 1:1, 6:2, 7:3, 8:4, 9:5, 15:6, 16:7, 17:8}
    
    def __init__(self, features=['atXYZ'], targets=['eAT'], pad=None, 
                         in_dir='./data/qm7x/', selector=['i1-c1-opt'],
                            embed=[('atNUM',6,16,0,True)], use_h5=True):
        
        self.features, self.targets = features, targets 
        self.pad, self.embed, self.in_dir  = pad, embed, in_dir
        self.selector, self.use_h5 = selector, use_h5
        self.datadic = self.load_data()
        self.ds_idx = list(self.datadic.keys())
        if use_h5:
            self.h5_handles = self.load_h5()
         
    def __getitem__(self, i):
        #if multiple conformations one is randomly selected
        conformations = self.datadic[i]['idconf']
        idconf = random.choice(conformations)

        if self.use_h5:
            #select the correct h5 handle
            if i == 1: j = 1
            else: j = i-1
            k = j // 1000  
            handle = self.h5_handles[k]
            mol = handle[str(i)][idconf]
            x_con = self.get_features(mol, self.features, 'float32')
            targets = self.get_features(mol, self.targets, 'float64')             
            molecule = self.get_features(mol, [self.embed[0][0]], 'int64')   
        else:
            x_con = self.datadic[i][idconf]['x_con']
            targets = self.datadic[i][idconf]['targets']
            molecule = self.datadic[i][idconf]['x_cat']
            
        x_cat = []
        if len(molecule) > 0:
            idx = []
            for atom in molecule:
                idx.append(QM7X.atomic_n[atom])
            x_cat.append(as_tensor(np.asarray(idx, 'int64')))
            
        return as_tensor(x_con), x_cat, as_tensor(targets)
         
    def __len__(self):
        return len(self.ds_idx)
        
    def get_features(self, mol, features, dtype):
        #datadic[idmol][idconf][feature]
        data = []
        for f in features:
            #(Nc, Na)    
            if f in ['atNUM','hVOL','hRAT','cCHG','atC6','atPOL','vdwR']:
                out = np.reshape(mol[f], -1).astype(dtype)
                if self.pad:
                    out = np.pad(out, (0, (self.pad - out.shape[0])))        
            #(Nc, Na, 3)   
            elif f in ['atXYZ','totFOR','vdwFOR','pbe0FOR','hVDIP']:
                out = np.reshape(mol[f], -1).astype(dtype)
                if self.pad:
                    out = np.pad(out, (0, (self.pad*3 - out.shape[0])))
            #(Nc, Na, Na)
            elif f in ['distance']:
                out = mol['atXYZ']
                out = sp.distance.squareform(sp.distance.pdist(out))
                out = np.reshape(out, -1).astype(dtype)
                if self.pad:
                    out = np.pad(out, (0, (self.pad*self.pad - out.shape[0])))
            #(Nc, 9), (Nc, 3), (Nc)
            else:
                out = np.reshape(mol[f], -1).astype(dtype)
                
        data.append(out)
        if len(data) == 0:
            return data
        else: 
            return np.concatenate(data)     
    
    def load_h5(self):
        h5_handles = []
        for set_id in QM7X.set_ids:
            handle = h5py.File(self.in_dir+set_id+'.hdf5', 'r')
            h5_handles.append(handle)
        return h5_handles
    
    def load_data(self):
        """seletor = list of regular expression strings (attr) for searching 
        and selecting idconf keys.  
        returns datadic[idmol] = [idconf,idconf,...]
        idconf, ID configuration (e.g., 'Geom-m1-i1-c1-opt', 'Geom-m1-i1-c1-50')
        """
        datadic = {}
        structure_count = 0
        for set_id in QM7X.set_ids:
            with h5py.File(self.in_dir+set_id+'.hdf5', 'r') as f:
                print('mapping... ', f)
                for idmol in f:
                    datadic[int(idmol)] = {'idconf': []}
                    for idconf in f[idmol]:
                        for attr in self.selector:
                            if re.search(attr, idconf):
                                datadic[int(idmol)]['idconf'].append(idconf)
                                structure_count += 1
                                if not self.use_h5:
                                    datadic[int(idmol)][idconf] = {}
                                    x_con = self.get_features(f[idmol][idconf], 
                                                              self.features, 'float32')
                                    datadic[int(idmol)][idconf]['x_con'] = x_con

                                    targets = self.get_features(f[idmol][idconf], 
                                                                self.targets, 'float64')
                                    datadic[int(idmol)][idconf]['targets'] = targets  
                                    
                                    x_cat = self.get_features(f[idmol][idconf], 
                                                              [self.embed[0][0]], 'int64')
                                    datadic[int(idmol)][idconf]['x_cat'] = x_cat
                                                          
                    if len(datadic[int(idmol)]['idconf']) == 0: del datadic[int(idmol)]
                    
        print('molecular formula (idmol) mapped: ', len(datadic))
        print('total molecular structures (idconf) mapped: ', structure_count)
        return datadic                                        
        
class QM7(CDataset):
    """http://quantum-machine.org/datasets/
    This dataset is a subset of GDB-13 (a database of nearly 1 billion stable 
    and synthetically accessible organic molecules) composed of all molecules of 
    up to 23 atoms (including 7 heavy atoms C, N, O, and S), totalling 7165 molecules. 
    We provide the Coulomb matrix representation of these molecules and their atomization 
    energies computed similarly to the FHI-AIMS implementation of the Perdew-Burke-Ernzerhof 
    hybrid functional (PBE0). This dataset features a large variety of molecular structures 
    such as double and triple bonds, cycles, carboxy, cyanide, amide, alcohol and epoxy.
    
    Prediction of the Atomization Energy of Molecules Using Coulomb Matrix and Atomic
    Composition in a Bayesian Regularized Neural Networks
    https://arxiv.org/abs/1904.10321
    """
    atomic_n = {0:0, 1:1, 6:2, 7:3, 8:4, 16:5}
    
    def __init__(self, features=[], targets=[], embed=[], 
                     in_file = './data/qm7/qm7.mat'):
        
        self.features, self.targets = features, targets
        self.embed = embed
        self.load_data(in_file)
        
    def __getitem__(self, i):
        
        def get_features(features, dtype):
            data = []
            for f in features:
                if f in ['coulomb']:
                    attr = getattr(self, f)
                    out = attr[i,:,:]
                elif f in ['xyz','atoms']:
                    attr = getattr(self, f)
                    out = attr[i,:]
                elif f in ['ae']:
                    attr = getattr(self, f)
                    out = attr[:,i]
                elif f in ['distance']:
                    attr = getattr(self, 'xyz')
                    padded = attr[i,:]
                    try:
                        j = np.where(padded == 0)[0][0] #molecule length
                        xyz = padded[:j,:] #padding removed
                    except:
                        xyz = padded #unpadded(longest molecule)

                    out = sp.distance.squareform(sp.distance.pdist(xyz))
                    out = np.reshape(out, -1).astype(dtype)
                    out = np.pad(out, (0, 23*23 - out.shape[0]))
                    
                data.append(np.reshape(out, -1).astype(dtype))
                
            if len(data) == 0:
                return data
            else: 
                return np.concatenate(data)
        
        x_con = get_features(self.features, np.float32)
        targets = get_features(self.targets, np.float64)
        molecule = get_features([self.embed[0][0]], np.int64)
        
        x_cat = []
        if len(molecule) > 0:
            idx = []
            for atom in molecule:
                idx.append(QM7.atomic_n[atom])
            x_cat.append(as_tensor(np.asarray(idx, 'int64')))

        return as_tensor(x_con), x_cat, as_tensor(targets)
    
    def __len__(self):
        return len(self.ds_idx)
    
    def load_data(self, in_file):
        qm7 = loadmat(in_file)
        self.coulomb = qm7['X'] # (7165, 23, 23)
        self.xyz = qm7['R'] # (7165, 3)
        self.atoms = qm7['Z'] # (7165, 23)
        self.ae = qm7['T'] # (1, 7165) atomization energy
        self.ds_idx = list(range(7165))
        
        
class QM7b(CDataset):
    """http://quantum-machine.org/datasets/
    This dataset is an extension of the QM7 dataset for multitask learning where 13 
    additional properties (e.g. polarizability, HOMO and LUMO eigenvalues, excitation 
    energies) have to be predicted at different levels of theory (ZINDO, SCS, PBE0, GW). 
    Additional molecules comprising chlorine atoms are also included, totalling 7211 molecules.
    
    properties: atomization energies, static polarizabilities (trace of tensor) α, frontier 
    orbital eigenvalues HOMO and LUMO, ionization potential, electron affinity, optical 
    spectrum simulations (10nm-700nm) first excitation energy, optimal absorption maximum, 
    intensity maximum.
    
    Machine Learning of Molecular Electronic Properties in Chemical Compound Space
    https://th.fhi-berlin.mpg.de/site/uploads/Publications/QM-NJP_20130315.pdf
    """
    properties = ['E','alpha_p','alpha_s','HOMO_g','HOMO_p','HOMO_z',
                  'LUMO_g','LUMO_p','LUMO_z','IP','EA','E1','Emax','Imax']
   
    def __init__(self, features=[], targets=[], embed=[], 
                     in_file='./data/qm7b/qm7b.mat'):
        self.features, self.targets = features, targets
        self.embed = embed
        self.load_data(in_file)
        
    def __getitem__(self, i):
        
        def get_features(features, dtype):
            data = []
            for f in features:
                if f in ['coulomb']:
                    out = getattr(self, f)[i,:,:]
                elif f in ['E','alpha_p','alpha_s','HOMO_g','HOMO_p','HOMO_z',
                           'LUMO_g','LUMO_p','LUMO_z','IP','EA','E1','Emax','Imax']:
                    out = getattr(self, 'properties')[i,QM7b.properties.index(f)]
                else:
                    NotImplemented('feature not found...')
                data.append(np.reshape(out, -1).astype(dtype))
                
            if len(data) == 0:
                return data
            else: 
                return np.concatenate(data)
        
        x_con = get_features(self.features, np.float32)
        targets = get_features(self.targets, np.float64)

        return as_tensor(x_con), [], as_tensor(targets)
    
    def __len__(self):
        return len(self.ds_idx)  
    
    def load_data(self, in_file):
        qm7b = loadmat(in_file)
        self.coulomb = qm7b['X'] # (7211, 23, 23)
        self.properties = qm7b['T'] # (7211, 14)
        self.ds_idx = list(range(7211))
          