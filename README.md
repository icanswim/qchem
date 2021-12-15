# qchem 
An exploration of the state of the art in the application of datascience to molecular quantum mechanics. 

From Machine Learning for Molecular Simulation  
>In 1929 Paul Dirac stated that: “The underlying physical laws necessary for the mathematical theory of a large part of physics and the whole of chemistry are thus completely known, and the difficulty is only that the exact application of these laws leads to equations much too complicated to be soluble.  It therefore becomes desirable that approximate practical methods of applying quantum mechanics should be developed, which can lead to an explanation of the main features of complex atomic systems without too much computation.”  Ninety years later,  this quote is still state of the art.  However, in the last decade, new tools from the rapidly developing field of machine learning (ML) have started to make significant impact on the development of approximate methods for complex atomic systems, bypassing the direct solution of “equations much too complicated to be soluble”.

## Datasets, models and learners implemented in pytorch 
* light weight, modular, extentable
* pytorch=1.8, pytorch_geometric, cuda=10.2, python=3.8
* uses the icanswim/cosmosis datascience repo for rapid prototyping
* see setup_notes.txt for implementation details
* see experiment.ipynb for examples


## References 
GeoMol: Torsional Geometric Generation of Molecular 3D Conformer Ensembles  
https://arxiv.org/abs/2106.07802  
https://github.com/PattanaikL/GeoMol  

Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges  
https://arxiv.org/abs/2104.13478  

Learning Neural Generative Dynamics for Molecular Conformation Generation  
https://arxiv.org/abs/2102.10240  

Equivariant message passing for the prediction of tensorial properties and molecular spectra  
https://arxiv.org/abs/2102.03150  

Graph Neural Networks with Learnable Structural and Positional Representations  
https://arxiv.org/abs/2110.07875  
https://github.com/vijaydwivedi75/gnn-lspe  

Physics-based Deep Learning  
https://arxiv.org/abs/2109.05237  

AlphaFold  
https://www.nature.com/articles/s41586-021-03819-2  
https://github.com/deepmind/alphafold 

Deep learning for molecular design - a review of the state of the art  
https://arxiv.org/abs/1903.04388 

Machine learning for molecular simulation  
https://arxiv.org/abs/1911.02792 

Machine learning for molecular and materials science  
https://www.researchgate.net/publication/326608140_Machine_learning_for_molecular_and_materials_science 

Deep learning methods in protein structure prediction  
https://www.sciencedirect.com/science/article/pii/S2001037019304441 

Machine learning for protein folding and dynamics  
https://arxiv.org/abs/1911.09811 

Graph Attention Networks  
https://arxiv.org/abs/1710.10903  
https://youtu.be/8owQBFAHw7E  

Learning to Simulate Complex Physics with Graph Networks  
https://arxiv.org/abs/2002.09405  
https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate 

Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks  
https://arxiv.org/abs/1810.00825  
https://github.com/juho-lee/set_transformer 

Learning continuous and data-driven molecular descriptors by translating equivalent chemical representations  
https://pubs.rsc.org/en/content/articlehtml/2019/sc/c8sc04175j 

Machine Learning Force Fields  
https://arxiv.org/abs/2010.07067 

Unifying machine learning and quantum chemistry -- a deep neural network for molecular wavefunctions  
https://arxiv.org/abs/1906.10033 

Integrating Machine Learning with Physics-Based Modeling  
https://arxiv.org/abs/2006.02619 

Bypassing the Kohn-Sham equations with machine learning  
https://www.nature.com/articles/s41467-017-00839-3 

### Chemistry Machine Learning Libraries 
RDKit  
https://www.rdkit.org/ 

PyG  
https://pytorch-geometric.readthedocs.io/en/latest/  

SchNetPack: A Deep Learning Toolbox For Atomistic Systems  
https://arxiv.org/abs/1809.01072 
https://github.com/atomistic-machine-learning/schnetpack 

Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals  
https://arxiv.org/abs/1812.05055  
https://github.com/materialsvirtuallab/megnet 

MoleculeNet: A Benchmark for Molecular Machine Learning  
https://arxiv.org/abs/1703.00564  
https://github.com/deepchem/deepchem 

### Datasets 
Quantum-Machine.org  
http://quantum-machine.org/datasets/ 

GDB Databases  
https://gdb.unibe.ch/downloads/ 

Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models  
https://arxiv.org/abs/1906.09427  
https://github.com/tencent-alchemy/Alchemy 

QM7-X: A comprehensive dataset of quantum-mechanical properties spanning the chemical space of small organic molecules  
https://arxiv.org/abs/2006.15139  
https://zenodo.org/record/3905361 

The ANI-1ccx and ANI-1x data sets, coupled-cluster and density functional theory properties for molecules  
https://www.nature.com/articles/s41597-020-0473-z#Sec11 

PyG Datasets  
https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html  

Kaggle  
https://www.kaggle.com/c/champs-scalar-coupling 




