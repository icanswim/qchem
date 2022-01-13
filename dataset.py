import sys # required for relative imports in jupyter lab
sys.path.insert(0, '../')

from cosmosis.dataset import CDataset

from abc import ABC, abstractmethod
import os, re, random, h5py, pickle

import numpy as np

from scipy import spatial as sp
from scipy.io import loadmat

from rdkit import Chem

from torch import as_tensor, cat


class Molecule(ABC):
    """a class for creating rdmol, mol_block, adjacency, distance and coulomb.  
    subclass and impliment load_data() and __repr__()."""
    atomic_n = {'C': 6, 'H': 1, 'N': 7, 'O': 8, 'F': 9}
    properties = ['rdmol','mol_block','adjacency','distance','coulomb']
    
    def __init__(self, *args):
        self.load_data(*args)
        if hasattr(self, 'smile'):
            self.rdmol_from_smile(self.smile)
        if hasattr(self, 'mol_block'):
            self.adjacency = self.create_adjacency(self.mol_block)
        if hasattr(self, 'xyz'):
            self.distance = self.create_distance(self.xyz)
        if hasattr(self, 'distance') and hasattr(self, 'atom_types'):
            self.coulomb = self.create_coulomb(self.distance, self.atom_types) 
        
    @abstractmethod
    def __repr__(self):
        return self.mol_id
        
    @abstractmethod
    def load_data(self, *args):
        self.smile = None
        self.mol_block = None
        self.xyz = None
        self.distance = None
        self.atom_types = None
        self.n_atoms = None
        self.atomic_numbers = None
        
    def open_file(self, in_file):
        with open(in_file) as f:
            data = []
            for line in f.readlines():
                data.append(line)
            return data
        
    def rdmol_from_smile(self, smile):
        self.rdmol = Chem.AddHs(Chem.MolFromSmiles(smile))

        self.atom_types = []
        self.atomic_numbers = []
        self.aromatic = []
        self.hybrid_types = []
        
        for atom in self.rdmol.GetAtoms():
            self.atom_types.append(atom.GetSymbol())
            self.atomic_numbers.append(atom.GetAtomicNum()) 
            self.aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybrid = atom.GetHybridization()
            if hybrid == Chem.HybridizationType.SP: self.hybrid_types.append('sp')
            elif hybrid == Chem.HybridizationType.SP2: self.hybrid_types.append('sp2')
            elif hybrid == Chem.HybridizationType.SP3: self.hybrid_types.append('sp3')
            else: self.hybrid_types.append('na')
            
        self.mol_block = Chem.MolToMolBlock(self.rdmol)
        self.n_atoms = self.rdmol.GetNumAtoms()

    def create_adjacency(self, mol_block):
        """use the V2000 chemical table's (rdmol MolBlock) adjacency list to create a 
        nxn symetric matrix with 0, 1, 2 or 3 for bond type where n is the indexed 
        atom"""
        adjacency = np.zeros((self.n_atoms, self.n_atoms), dtype='int64')
        block = mol_block.split('\n')
        for b in block[:-2]:
            line = ''.join(b.split())
            if len(line) == 4:
                # shift -1 to index from zero
                adjacency[(int(line[0])-1),(int(line[1])-1)] = int(line[2]) 
                # create bi-directional connection
                adjacency[(int(line[1])-1),(int(line[0])-1)] = int(line[2]) 
        return adjacency
            
    def create_distance(self, xyz):
        m = np.zeros((len(xyz), 3))
        for i, atom in enumerate(xyz):
            m[i,:] = atom 
        distance = sp.distance.squareform(sp.distance.pdist(m)).astype('float32')
        return distance
      
    def create_coulomb(self, distance, atom_types, sigma=1):
        """creates coulomb matrix obj attr.  set sigma to False to turn off random sorting.  
        sigma = stddev of gaussian noise.
        https://papers.nips.cc/paper/4830-learning-invariant-representations-of-\
        molecules-for-atomization-energy-prediction"""
        atoms = []
        for atom in atom_types:
            atoms.append(Molecule.atomic_n[atom]) 
        atoms = np.asarray(atoms, dtype='float32')
        qmat = atoms[None, :]*atoms[:, None]
        idmat = np.linalg.inv(distance)
        np.fill_diagonal(idmat, 0)
        coul = qmat@idmat
        np.fill_diagonal(coul, 0.5 * atoms ** 2.4)
        if sigma:  
            coulomb = self.sort_permute(coul, sigma)
        else:  
            coulomb = coul
        return coulomb
    
    def sort_permute(self, matrix, sigma):
        norm = np.linalg.norm(matrix, axis=1)
        noised = np.random.normal(norm, sigma)
        indexlist = np.argsort(noised)
        indexlist = indexlist[::-1]  #invert
        return matrix[indexlist][:,indexlist]
    
class QM9Mol(Molecule):
    
    properties = ['A','B','C','mu','alpha','homo','lumo', 
                  'gap','r2','zpve','U0','U','H','G','Cv',
                  'smile','n_atoms','xyz','mulliken']
       
    def __repr__(self):
        return self.in_file[-20:-4]
    
    def load_data(self, in_file):
        """load from the .xyz files of the qm9 dataset
        (http://quantum-machine.org/datasets/)
        """
        self.in_file = in_file
        self.qm9_block = self.open_file(in_file)
        self.smile = self.qm9_block[-2]    
        self.n_atoms = int(self.qm9_block[0])
        
        properties = self.qm9_block[1].strip().split('\t')[1:] #[float,...]
        for i, p in enumerate(properties):
            setattr(self, QM9Mol.properties[i], np.reshape(np.asarray(p, 'float32'), -1))
            
        atom_types = []
        xyz = []
        mulliken = []
        for atom in self.qm9_block[2:self.n_atoms+2]:
            stripped = atom.strip().split('\t') #[['atom_type',x,y,z,mulliken],...] 
            atom_types.append(stripped[0])
            xyz.append(np.reshape(np.asarray( #fix scientific notation
                np.char.replace(stripped[1:4], '*^', 'e'), dtype=np.float32), -1))
            mulliken.append(np.reshape(np.asarray( #fix scientific notation
                np.char.replace(stripped[4], '*^', 'e'), dtype=np.float32), -1))

        self.atom_types = atom_types
        self.xyz = np.reshape(np.concatenate(xyz), (-1, 3))
        self.mulliken = np.concatenate(mulliken, axis=0)
       
    
class QM9(CDataset):
    """http://quantum-machine.org
    
    Dataset source/download:
    https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
    
    Decompress dsgdb9nsd.xyz.tar.bz2 in the 'in_dir' folder (qchem/data/qm9/qm9.xyz)
    
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
    
    pad = length of longest molecule that all molecules will be padded to
    features/target = QM9.properties
    filter_on = ('property', 'test', 'value')
    n = non random subset selection (for testing)
    use_pickle = False/'qm9_datadic.p' (if file exists loads, if not creates and saves)
    """
    LOW_CONVERGENCE = [21725,87037,59827,117523,128113,129053,129152, 
                       129158,130535,6620,59818]
    
    properties = ['A','B','C','mu','alpha','homo','lumo','gap','r2',
                  'zpve','U0','U','H','G','Cv','n_atoms','smile','coulomb',
                  'adjacency','distance','mulliken']
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
         
    def _get_features(self, mol, features):
        data = []
        for f in features:
            out = getattr(mol, f)
            if self.pad is not None:
                if f in ['coulomb','adjacency','distance','mulliken']: 
                    out = np.pad(out, (0, (self.pad - out.shape[0])))
            if self.flatten:
                out = np.reshape(out, -1)
            data.append(out)
        if len(data) == 0:
            return data
        else:
            return np.concatenate(data)
        
    def _get_embed_idx(self, mol, embeds, embed_lookup):
        embed_idx = []
        for e in embeds:
            out = getattr(mol, e)
            idx = []
            if self.pad is not None:
                out = np.pad(out, (0, (self.pad - out.shape[0])))
            for i in np.reshape(out, -1).tolist():
                idx.append(np.reshape(np.asarray(embed_lookup[i]), -1).astype('int64'))
            embed_idx.append(np.concatenate(idx))
        return embed_idx
       
    def open_file(self, in_file):
        with open(in_file) as f:
            data = []
            for line in f.readlines():
                data.append(line)
            return data
        
    def load_data(self, in_dir='./data/qm9/qm9.xyz/', n=133885,  
                 filter_on=None, use_pickle='qm9_datadic.p', dtype='float32'): 
        
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
                    
                    if filter_on is not None:
                        val = self._get_features(datadic[int(filename[-10:-4])], 
                                                     [filter_on[0]])
                        val = np.array2string(val, precision=4, floatmode='maxprec')[1:-1]
                     
                        if not eval(val+filter_on[1]+filter_on[2]):
                            del datadic[int(filename[-10:-4])]
                        
                    if i % 10000 == 1: 
                        print('QM9 molecules scanned: ', i)
                        print('QM9 molecules created: ', len(datadic))
                    if len(datadic) > n - 1:
                        break
            
            unchar = 0
            uncharacterized = self.get_uncharacterized()
            for mol in uncharacterized: 
                try: 
                    del datadic[mol]
                    unchar += 1
                except: continue
            print('total uncharacterized molecules removed: ', unchar)       
            print('total QM9 molecules created: ', len(datadic))
            
            if use_pickle:
                print('pickling a copy of the QM9 datadic...')        
                pickle.dump(datadic, open('./data/qm9/'+use_pickle, 'wb'))
                
        return datadic
    
    def get_uncharacterized(self, in_file='./data/qm9/uncharacterized.txt'):
        """uncharacterized.txt - 3054 molecules from the 133885 set that failed a 
        consistency check.  Returns a list of ints of the 3054 molecules (datadic keys)"""
        unchar = []
        try:
            data = self.open_file(in_file)
            for mol in data[8:]:
                for m in mol.strip().split():
                    if m.isdigit():
                        unchar.append(int(m))
        except:
            print('uncharaterized file missing...')
   
        return unchar
        
        
class ANI1x(CDataset):
    """https://www.nature.com/articles/s41597-020-0473-z#Sec11
    https://github.com/aiqm/ANI1x_datasets
    
    Dataset source/download:
    https://springernature.figshare.com/articles/dataset/ANI-1x_Dataset_Release/10047041
    
    Place the downloaded h5 file in the 'in_dir' folder (qchem/data/ani1x)
    
    The source dataset is organized:
    [molecular formula][conformation index][feature,feature,...]
    
    It is indexed by a molecular formula and conformation index
    
    This dataset is a pytorch and cosmosis dataset:
    Returns [feature1,feature2,...,padding], [target1,target2,...]
    
    Longest molecule is 63 atoms
    
    select:
        molecular formula
        conformation
        properties
        target
        padding
        data file location
        embedding options
    
    criterion = the property used to select the conformation
    conformation = logic used on the criterion property
        'min' - choose the index with the lowest value
        'max' - choose the index with the highest value
        'random' - choose the index randomly 
        int - choose the index int
    properties = ['feature','feature',...]
    targets = ['feature',...]
    pad = int/False
    embeds = [('feature',voc,vec,padding_idx,train)] 
    
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
    properties = ['atomic_numbers', 'ccsd(t)_cbs.energy', 'coordinates', 'hf_dz.energy',
                  'hf_qz.energy', 'hf_tz.energy', 'mp2_dz.corr_energy', 'mp2_qz.corr_energy',
                  'mp2_tz.corr_energy', 'npno_ccsd(t)_dz.corr_energy', 'npno_ccsd(t)_tz.corr_energy',
                  'tpno_ccsd(t)_dz.corr_energy', 'wb97x_dz.cm5_charges', 'wb97x_dz.dipole', 
                  'wb97x_dz.energy', 'wb97x_dz.forces', 'wb97x_dz.hirshfeld_charges', 
                  'wb97x_dz.quadrupole', 'wb97x_tz.dipole', 'wb97x_tz.energy', 'wb97x_tz.forces',
                  'wb97x_tz.mbis_charges', 'wb97x_tz.mbis_dipoles', 'wb97x_tz.mbis_octupoles',
                  'wb97x_tz.mbis_quadrupoles', 'wb97x_tz.mbis_volumes','distance']
    
    
    def __init__(self, criterion=[], conformation='random', **kwargs):
        self.criterion = criterion
        self.conformation = conformation
        super().__init__(**kwargs)        

        
    def __getitem__(self, i):
        X, embed_idx, y = [], [], []
        ci = self._get_conformation_index(self.ds[i])
        
        if len(self.features) > 0:
            X = self._get_features(self.ds[i], self.features, ci)
        for transform in self.transform:
                X = transform(X)
        
        if len(self.embeds) > 0:
            embed_idx = self._get_embed_idx(self.ds[i], self.embeds, self.embed_lookup)
        
        if len(self.targets) > 0:
            y = self._get_features(self.ds[i], self.targets, ci)
        for transform in self.target_transform:
            y = transform(y)
            
        return X, embed_idx, y    
        
    def _get_features(self, datadic, features, ci):
        data = []
        for f in features:
            if f == 'atomic_numbers':
                out = datadic[f]
            elif f == 'distance':
                out = sp.distance.squareform(sp.distance.pdist(datadic[f][ci]))
                out = out.astype('float32')
                if self.pad is not None:
                    out = np.pad(out, (0, (self.pad - out.shape[0])))
                    if self.flatten:
                        out = np.reshape(out, -1)
            else:
                out = datadic[f][ci]
            if out.ndim == 0:
                out = np.reshape(out, -1)
            data.append(out)
            
        return np.concatenate(data)
        
    def _get_conformation_index(self, datadic):
        """each molecular formula (mol) may have many different isomers
        select the conformation based on some criterion (attribute value)
        """
        if len(self.criterion) == 0:
            self.criterion = list(self.targets[0])
        
        ci = 0        
        if isinstance(self.conformation, int):
            ci = self.conformation
        elif self.conformation == 'random':
            ci = random.randrange(datadic[self.criterion[0]].shape[0])
        elif self.conformation == 'max':
            ci = np.argmax(datadic[self.criterion[0]], axis=0)
        elif self.conformation == 'min':
            ci = np.argmin(datadic[self.criterion[0]], axis=0)
            
        return ci
    
    def _load_features(self, mol, features, dtype='float32'):
        datadic = {}
        nan = False
        while not nan:
            for f in features:
                if f in ['distance']: 
                    out = mol['coordinates'][()]
                else: 
                    out = mol[f][()]
                    
                if np.isnan(out).any(): 
                    nan = True
                    
                if self.pad is not None:
                    #(Na)
                    if f in ['atomic_numbers']:
                        out = np.pad(out, (0, (self.pad - out.shape[0])))
                    #(Na, Nc)    
                    elif f in ['wb97x_dz.cm5_charges','wb97x_dz.hirshfeld_charges',
                               'wb97x_tz.mbis_charges','wb97x_tz.mbis_dipoles',
                               'wb97x_tz.mbis_quadrupoles','wb97x_tz.mbis_octupoles',
                               'wb97x_tz.mbis_volumes']:
                        out = np.pad(out, ((0, (self.pad - out.shape[0])), (0, 0)))
                    #(Na, Nc, 3)   
                    elif f in ['coordinates','wb97x_dz.forces','wb97x_dz.forces']:
                        out = np.pad(out, ((0, (self.pad - out.shape[0])), (0, 0), (0, 0)))
                    #(Nc, 6), (Nc, 3), (Nc) no padding
                    
                if self.flatten and f != 'distance': 
                    out = np.reshape(out, -1)
                out = out.astype(dtype)
                datadic[f] = out
            return datadic, nan #returns datadic, nan=False
        return datadic, nan #breaks out returns datadic, nan=True
                    
               
    def load_data(self, in_file='./data/ani1x/ani1x-release.h5'):
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
        datadic = {}
        with h5py.File(in_file, 'r') as f:
            for mol in f.keys():
                features, nan = self._load_features(f[mol], 
                                                    self.criterion+self.features+\
                                                    self.embeds+self.targets)
                if nan:
                    continue
                else:
                    datadic[mol] = features
                if len(datadic) % 1000 == 0:
                    print('molecules loaded: ', len(datadic))
                                    
        print('molecules loaded: ', len(datadic))
        self.embed_lookup = {0:0, 1:1, 6:2, 7:3, 8:4, 9:5, 15:6, 16:7, 17:8}
        return datadic    

                
class QM7X(CDataset):
    """QM7-X: A comprehensive dataset of quantum-mechanical properties spanning 
    the chemical space of small organic molecules
    https://arxiv.org/abs/2006.15139
    
    Dataset source/download:
    https://zenodo.org/record/3905361
    
    Decompress the .xz files in the 'in_dir' folder (qchem/data/qm7x/)
    
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
    
    'distance': N x N distance matrix created from atXYZ 
    
    seletor = list of regular expression strings (attr) for searching 
        and selecting idconf keys.
        idconf, ID configuration (e.g., 'Geom-m1-i1-c1-opt', 'Geom-m1-i1-c1-50')
    flatten = True/False 
    pad = None/int (pad length int in the Na (number of atoms) dimension)
    
    returns datadic[idmol][idconf][properties]
    """
    set_ids = ['1000','2000','3000','4000','5000','6000','7000','8000']
    
    properties = ['DIP','HLgap','KSE','atC6','atNUM','atPOL','atXYZ','eAT', 
                'eC','eDFTB+MBD','eEE','eH','eKIN','eKSE','eL','eMBD','eNE', 
                'eNN','ePBE0','ePBE0+MBD','eTS','eX','eXC','eXX','hCHG', 
                'hDIP','hRAT','hVDIP','hVOL','mC6','mPOL','mTPOL','pbe0FOR', 
                'sMIT','sRMSD','totFOR','vDIP','vEQ','vIQ','vTQ','vdwFOR','vdwR',
                'distance']
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
         
    def __getitem__(self, i):
        X, embed_idx, y = [], [], []
        #if multiple conformations one is randomly selected
        conformations = list(self.ds[i].keys())
        idconf = random.choice(conformations)
        
        if len(self.features) > 0:
            X = self._get_features(self.ds[i][idconf], self.features)
        for transform in self.transform:
                X = transform(X)
        
        if len(self.embeds) > 0:
            embed_idx = self._get_embed_idx(self.ds[i][idconf], self.embeds, self.embed_lookup)
        
        if len(self.targets) > 0:
            y = self._get_features(self.ds[i][idconf], self.targets)
        for transform in self.target_transform:
            y = transform(y)
            
        return X, embed_idx, y
    
    def _get_features(self, datadic, features):
        data = []
        for f in features:
            out = datadic[f]
            if self.pad is not None:
                #(Nc, Na), (Nc, Na, Na)
                if f in ['atNUM','hVOL','hRAT','cCHG','atC6','atPOL','vdwR','distance']:
                    out = np.pad(out, (0, (self.pad - out.shape[0])))
                #(Nc, Na, 3)
                elif f in ['atXYZ','totFOR','vdwFOR','pbe0FOR','hVDIP']:
                    out = np.pad(out, ((0, (self.pad - out.shape[0])), (0, 0)))
                #(Nc, 9), (Nc, 3), (Nc) no padding
            if self.flatten:
                out = np.reshape(out, -1)
                
            data.append(out)
            
        return np.concatenate(data)
        
    def _load_features(self, mol, features, dtype='float32'):
        datadic = {}
        for f in features:
            if f == 'distance': 
                out = mol['atXYZ'][()]
                out = sp.distance.squareform(sp.distance.pdist(out))
            else: 
                out = mol[f][()]
            datadic[f] = out.astype(dtype)
        return datadic
        
    def load_data(self, selector='opt', in_dir='./data/qm7x/'):
        """seletor = list of regular expression strings (attr) for searching 
        and selecting idconf keys.  
        returns datadic[idmol] = {'idconf': {'feature': val}}
        idconf = ID configuration (e.g., 'Geom-m1-i1-c1-opt', 'Geom-m1-i1-c1-50')
        datadic[idmol][idconf][feature]
        """
        datadic = {}
        structure_count = 0
        for set_id in QM7X.set_ids:
            with h5py.File(in_dir+set_id+'.hdf5', 'r') as f:
                print('mapping... ', f)
                for idmol in f:
                    datadic[int(idmol)] = {}
                    for idconf in f[idmol]:
                        for attr in selector:
                            if re.search(attr, idconf):
                                structure_count += 1
                                features = self._load_features(f[idmol][idconf], 
                                                               self.features+self.targets+self.embeds)
                                datadic[int(idmol)][idconf] = features
                                                                           
        print('molecular formula (idmol) mapped: ', len(datadic))
        print('total molecular structures (idconf) mapped: ', structure_count)
        self.embed_lookup = {0:0, 1:1, 6:2, 7:3, 8:4, 9:5, 15:6, 16:7, 17:8}
        return datadic                                        
        
class QM7(CDataset):
    """http://quantum-machine.org
    
    Dataset source/download:
    http://quantum-machine.org/data/qm7.mat
    
    Place the downloaded file in the 'in_dir' folder (qchem/data/qm7)
    
    This dataset is a subset of GDB-13 (a database of nearly 1 billion stable 
    and synthetically accessible organic molecules) composed of all molecules of 
    up to 23 atoms (including 7 heavy atoms C, N, O, and S), totalling 7165 molecules. 
    We provide the Coulomb matrix representation of these molecules and their atomization 
    energies computed similarly to the FHI-AIMS implementation of the Perdew-Burke-Ernzerhof 
    hybrid functional (PBE0). This dataset features a large variety of molecular structures 
    such as double and triple bonds, cycles, carboxy, cyanide, amide, alcohol and epoxy.
    
    coulomb = qm7['X'] # (7165, 23, 23)
    xyz = qm7['R'] # (7165, 3)
    atoms = qm7['Z'] # (7165, 23)
    ae = qm7['T'] # (1, 7165) atomization energy
    
    Prediction of the Atomization Energy of Molecules Using Coulomb Matrix and Atomic
    Composition in a Bayesian Regularized Neural Networks
    https://arxiv.org/abs/1904.10321
    """
    properties = ['coulomb','xyz','atoms','ae']
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def load_data(self, in_file = './data/qm7/qm7.mat'):
        ds = loadmat(in_file)
        datadic = {}
        for i in range(7165):
            datadic[i] = {}
            for f in self.features+self.embeds+self.targets:
                if f in ['coulomb']:
                    out = ds['X'][i,:,:]
                elif f in ['xyz']:
                    out = ds['R'][i,:]
                elif f in ['atoms']: 
                    out = list(map(int, ds['Z'][i,:]))  
                elif f in ['ae']:
                    out = ds['T'][:,i]
                elif f in ['distance']:
                    padded = ds['R'][i,:]
                    try:
                        j = np.where(padded == 0)[0][0] #molecule length
                        xyz = padded[:j,:] #padding removed
                    except:
                        xyz = padded #no padding (longest molecule)
                    out = sp.distance.squareform(sp.distance.pdist(xyz))
                    out = np.pad(out, ((0, 23-out.shape[0]), (0, 23-out.shape[1])))
                    
                if self.flatten: out = np.reshape(out, -1)
                datadic[i].update({f: out})
                    
        self.embed_lookup = {0:0, 1:1, 6:2, 7:3, 8:4, 16:5} #atomic numbers
        return datadic
        
class QM7b(CDataset):
    """http://quantum-machine.org
    
    Dataset source/download:
    http://quantum-machine.org/data/qm7b.mat
    
    Place the downloaded file in the 'in_dir' folder (qchem/data/qm7b)
    coulomb = ds['X'] # (7211, 23, 23)
    properties = ds['T'] # (7211, 14)
    
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
                  'LUMO_g','LUMO_p','LUMO_z','IP','EA','E1','Emax','Imax',
                  'coulomb']
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
          
    def load_data(self, in_file):
        ds = loadmat(in_file)
        datadic = {}
        for i in range(7211):
            datadic[i] = {}
            for f in self.features+self.embeds+self.targets:
                if f in ['coulomb']:
                    datadic[i].update({f: ds['X'][i,:,:].astype('float32')})
                elif f in ['E','alpha_p','alpha_s','HOMO_g','HOMO_p','HOMO_z',
                           'LUMO_g','LUMO_p','LUMO_z','IP','EA','E1','Emax','Imax']:
                    datadic[i].update({f: np.reshape(ds['T'][i,QM7b.properties.index(f)], 
                                                                     -1).astype('float32')})
                else:
                    NotImplemented("feature not implemented")    
        return datadic
                
          