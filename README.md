# qchem

A framework for molecular modeling. This framework combines the most powerful open-source libraries for machine learning, geometric deep learning, and chemoinformatics with full-featured molecular datasets while remaining lightweight, modular, and extendable.

From *Machine Learning for Molecular Simulation*:
> In 1929 Paul Dirac stated that: “The underlying physical laws necessary for the mathematical theory of a large part of physics and the whole of chemistry are thus completely known, and the difficulty is only that the exact application of these laws leads to equations much too complicated to be soluble. It therefore becomes desirable that approximate practical methods of applying quantum mechanics should be developed, which can lead to an explanation of the main features of complex atomic systems without too much computation.” Ninety years later, this quote is still state of the art. However, in the last decade, new tools from the rapidly developing field of machine learning (ML) have started to make a significant impact on the development of approximate methods for complex atomic systems, bypassing the direct solution of “equations much too complicated to be soluble”.

## Core Features
* **Lightweight, modular, and extendable** design.
* Implements datasets, models, and learners directly in **PyTorch**.
* Built-in environment stack: `pytorch=2.12`, `torch-geometric=2.8.1`, `cuda=12.6`, `python=3.12`, `rdkit=2026.9.1`, `scikit-learn=1.9.1`, `torcheval=0.0.7`.
* Integrates the `icanswim/cosmosis` data science repository for rapid prototyping.
* Setup instructions can be found in `setup_notes.txt`.
* Code implementation examples are provided in `experiment.ipynb`.

---

## Repository Structure

```text
.
├── cosmosis
│   └── src
│       ├── cosmosis
│       │   └── __pycache__
│       └── cosmosis.egg-info
├── data
│   ├── ani1x
│   ├── pyg
│   │   └── qm9
│   │       ├── processed
│   │       └── raw
│   ├── qm7
│   ├── qm7b
│   ├── qm7x
│   └── qm9
│       └── dsgdb9nsd.xyz
├── log
├── model
└── src
    └── qchem
        └── __pycache__

22 directories
```

---

## Installed Packages

| Package | Version | Editable Project Location |
| :--- | :--- | :--- |
| aiohappyeyeballs | 2.7.1 | |
| aiohttp | 3.14.5 | |
| aiosignal | 1.4.0 | |
| anyio | 4.15.1 | |
| argon2-cffi | 25.1.0 | |
| argon2-cffi-bindings | 26.1.0 | |
| arrow | 1.4.0 | |
| asttokens | 3.0.2 | |
| async-lru | 2.4.0 | |
| attrs | 26.1.0 | |
| babel | 2.18.0 | |
| beautifulsoup4 | 4.15.0 | |
| bleach | 6.4.0 | |
| certifi | 2026.10.19 | |
| cffi | 2.1.1 | |
| charset-normalizer | 3.5.2 | |
| cloudpickle | 3.1.2 | |
| comm | 0.2.3 | |
| contourpy | 1.4.0 | |
| cosmosis | 0.1.0 | `/home/fltr/qchem/cosmosis` |
| cuda-bindings | 12.9.10 | |
| cuda-pathfinder | 1.8.3 | |
| cuda-toolkit | 12.6.3 | |
| cycler | 0.12.1 | |
| debugpy | 1.8.22 | |
| defusedxml | 0.7.1 | |
| executing | 2.3.0 | |
| fastjsonschema | 2.22.2 | |
| filelock | 4.2.4 | |
| fonttools | 4.67.0 | |
| fqdn | 1.6.0 | |
| frozenlist | 1.8.0 | |
| fsspec | 2026.9.0 | |
| h11 | 0.16.0 | |
| h5py | 3.16.0 | |
| httpcore | 1.0.9 | |
| httpx | 0.28.1 | |
| idna | 3.20 | |
| ipykernel | 7.4.0 | |
| ipython | 9.17.1 | |
| ipython-pygments-lexers | 1.1.1 | |
| isoduration | 20.11.0 | |
| jedi | 0.20.1 | |
| jinja2 | 3.1.6 | |
| joblib | 1.6.0 | |
| json5 | 0.17.3 | |
| jsonpointer | 3.4 | |
| jsonschema | 4.26.0 | |
| jsonschema-specifications | 2025.9.1 | |
| jupyter-builder | 1.2.3 | |
| jupyter-client | 8.10.0 | |
| jupyter-core | 5.9.1 | |
| jupyter-events | 0.12.1 | |
| jupyter-lsp | 2.3.1 | |
| jupyter-server | 2.21.1 | |
| jupyter-server-terminals | 0.5.4 | |
| jupyterlab | 4.6.4 | |
| jupyterlab-pygments | 0.3.0 | |
| jupyterlab-server | 2.28.1 | |
| kiwisolver | 1.5.1 | |
| lark | 1.3.1 | |
| markupsafe | 3.0.4 | |
| matplotlib | 3.11.2 | |
| matplotlib-inline | 0.2.2 | |
| mistune | 3.4.0 | |
| mpmath | 1.3.0 | |
| multidict | 7.1.0 | |
| narwhals | 2.27.1 | |
| nbclient | 0.11.0 | |
| nbconvert | 7.17.2 | |
| nbformat | 5.11.1 | |
| nest-asyncio2 | 1.7.4 | |
| networkx | 3.7 | |
| notebook-shim | 0.2.4 | |
| numpy | 2.5.4 | |
| nvidia-cublas-cu12 | 12.6.4.1 | |
| nvidia-cuda-cupti-cu12 | 12.6.80 | |
| nvidia-cuda-nvrtc-cu12 | 12.6.85 | |
| nvidia-cuda-runtime-cu12 | 12.6.77 | |
| nvidia-cudnn-cu12 | 9.10.2.21 | |
| nvidia-cufft-cu12 | 11.3.0.4 | |
| nvidia-cufile-cu12 | 1.11.1.6 | |
| nvidia-curand-cu12 | 10.3.7.77 | |
| nvidia-cusolver-cu12 | 11.7.1.2 | |
| nvidia-cusparse-cu12 | 12.5.4.2 | |
| nvidia-cusparselt-cu12 | 0.7.1 | |
| nvidia-nccl-cu12 | 2.29.3 | |
| nvidia-nvjitlink-cu12 | 12.6.85 | |
| nvidia-nvshmem-cu12 | 3.4.5 | |
| nvidia-nvtx-cu12 | 12.6.77 | |
| packaging | 26.3 | |
| pandas | 3.0.6 | |
| pandocfilters | 1.5.1 | |
| parso | 0.8.7 | |
| pexpect | 4.9.0 | |
| pillow | 13.0.0 | |
| platformdirs | 4.13.5 | |
| prometheus-client | 0.26.0 | |
| prompt-toolkit | 3.0.53 | |
| propcache | 0.5.4 | |
| psutil | 7.2.2 | |
| ptyprocess | 0.7.0 | |
| pure-eval | 0.2.4 | |
| pycparser | 3.11 | |
| pygments | 2.21.0 | |
| pyparsing | 3.3.3 | |
| python-dateutil | 2.9.0.post0 | |
| python-json-logger | 4.2.0 | |
| pyyaml | 6.0.3 | |
| pyzmq | 27.2.0 | |
| qchem | 0.1.0 | `/home/fltr/qchem` |
| rdkit | 2026.9.1 | |
| referencing | 0.37.0 | |
| requests | 2.34.2 | |
| rfc3339-validator | 0.1.4 | |
| rfc3986-validator | 0.1.1 | |
| rfc3987-syntax | 1.1.0 | |
| rpds-py | 2026.9.1 | |
| scikit-learn | 1.9.1 | |
| scipy | 1.18.1 | |
| send2trash | 2.1.0 | |
| setuptools | 81.0.0 | |
| six | 1.17.0 | |
| soupsieve | 3.0.3 | |
| stack-data | 0.6.3 | |
| sympy | 1.14.0 | |
| terminado | 0.18.1 | |
| threadpoolctl | 3.7.0 | |
| tinycss2 | 1.5.1 | |
| torch | 2.12.1+cu126 | |
| torch-geometric | 2.8.1 | |
| torchaudio | 2.11.0+cu126 | |
| torcheval | 0.0.7 | |
| torchvision | 0.27.1+cu126 | |
| tornado | 6.5.10 | |
| tqdm | 4.70.1 | |
| traitlets | 5.16.1 | |
| triton | 3.7.1 | |
| typing-extensions | 4.16.0 | |
| tzdata | 2026.5 | |
| uri-template | 1.3.0 | |
| urllib3 | 2.8.0 | |
| wcwidth | 0.9.3 | |
| webcolors | 25.10.0 | |
| webencodings | 0.6.1 | |
| websocket-client | 1.9.2 | |
| xxhash | 4.0.2 | |
| yarl | 1.25.1 | |

---

## References

* **ICLR 2021 Keynote: Geometric Deep Learning**  
  [Video Presentation](https://youtu.be/w6Pw4MOzMuo)
* **Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges**  
  [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
* **GeoMol: Torsional Geometric Generation of Molecular 3D Conformer Ensembles**  
  [arXiv:2106.07802](https://arxiv.org/abs/2106.07802) | [GitHub Repository](https://github.com/PattanaikL/GeoMol)
* **Learning Neural Generative Dynamics for Molecular Conformation Generation**  
  [arXiv:2102.10240](https://arxiv.org/abs/2102.10240)
* **Equivariant message passing for the prediction of tensorial properties and molecular spectra**  
  [arXiv:2102.03150](https://arxiv.org/abs/2102.03150)
* **Graph Neural Networks with Learnable Structural and Positional Representations**  
  [arXiv:2110.07875](https://arxiv.org/abs/2110.07875) | [GitHub Repository](https://github.com/vijaydwivedi75/gnn-lspe)
* **Physics-based Deep Learning**  
  [arXiv:2109.05237](https://arxiv.org/abs/2109.05237)
* **Everything is Connected: Graph Neural Networks**  
  [arXiv:2301.08210](https://arxiv.org/abs/2301.08210)
* **AlphaFold**  
  [Nature Article](https://www.nature.com/articles/s41586-021-03819-2) | [GitHub Repository](https://github.com/deepmind/alphafold)
* **Deep learning for molecular design - a review of the state of the art**  
  [arXiv:1903.04388](https://arxiv.org/abs/1903.04388)
* **Machine learning for molecular simulation**  
  [arXiv:1911.02792](https://arxiv.org/abs/1911.02792)
* **Machine learning for molecular and materials science**  
  [ResearchGate Publication](https://www.researchgate.net/publication/326608140_Machine_learning_for_molecular_and_materials_science)
* **Deep learning methods in protein structure prediction**  
  [ScienceDirect Article](https://www.sciencedirect.com/science/article/pii/S2001037019304441)
* **Machine learning for protein folding and dynamics**  
  [arXiv:1911.09811](https://arxiv.org/abs/1911.09811)
* **Graph Attention Networks**  
  [arXiv:1710.10903](https://arxiv.org/abs/1710.10903) | [Video Explanation](https://youtu.be/8owQBFAHw7E)
* **Learning to Simulate Complex Physics with Graph Networks**  
  [arXiv:2002.09405](https://arxiv.org/abs/2002.09405) | [GitHub Repository](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate)
* **Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks**  
  [arXiv:1810.00825](https://arxiv.org/abs/1810.00825) | [GitHub Repository](https://github.com/juho-lee/set_transformer)
* **Learning continuous and data-driven molecular descriptors by translating equivalent chemical representations**  
  [RSC Publication](https://pubs.rsc.org/en/content/articlehtml/2019/sc/c8sc04175j)
* **Machine Learning Force Fields**  
  [arXiv:2010.07067](https://arxiv.org/abs/2010.07067)
* **Unifying machine learning and quantum chemistry -- a deep neural network for molecular wavefunctions**  
  [arXiv:1906.10033](https://arxiv.org/abs/1906.10033)
* **Integrating Machine Learning with Physics-Based Modeling**  
  [arXiv:2006.02619](https://arxiv.org/abs/2006.02619)
* **Bypassing the Kohn-Sham equations with machine learning**  
  [Nature Communications Article](https://www.nature.com/articles/s41467-017-00839-3)

### Chemistry Machine Learning Libraries

* **RDKit**  
  [Official Documentation](https://www.rdkit.org/)
* **PyG (PyTorch Geometric)**  
  [Official Documentation](https://pytorch-geometric.readthedocs.io/en/latest/)
* **SchNetPack: A Deep Learning Toolbox For Atomistic Systems**  
  [arXiv:1809.01072](https://arxiv.org/abs/1809.01072) | [GitHub Repository](https://github.com/atomistic-machine-learning/schnetpack)
* **Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals**  
  [arXiv:1812.05055](https://arxiv.org/abs/1812.05055) | [GitHub Repository](https://github.com/materialsvirtuallab/megnet)
* **MoleculeNet: A Benchmark for Molecular Machine Learning**  
  [arXiv:1703.00564](https://arxiv.org/abs/1703.00564) | [GitHub Repository](https://github.com/deepchem/deepchem)

### Datasets

* **Quantum-Machine.org**  
  [Dataset Portal](http://quantum-machine.org/datasets/)
* **GDB Databases**  
  [Download Page](https://gdb.unibe.ch/downloads/)
* **Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models**  
  [arXiv:1906.09427](https://arxiv.org/abs/1906.09427) | [GitHub Repository](https://github.com/tencent-alchemy/Alchemy)
* **QM7-X: A comprehensive dataset of quantum-mechanical properties spanning the chemical space of small organic molecules**  
  [arXiv:2006.15139](https://arxiv.org/abs/2006.15139) | [Zenodo Repository](https://zenodo.org/record/3905361)
* **The ANI-1ccx and ANI-1x data sets, coupled-cluster and density functional theory properties for molecules**  
  [Nature Scientific Data Article](https://www.nature.com/articles/s41597-020-0473-z#Sec11)
* **PyG Datasets**  
  [Available Datasets Documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)
* **Kaggle: Predict Molecular Properties**  
  [CHAMPS Scalar Coupling Challenge](https://www.kaggle.com/c/champs-scalar-coupling)
* **QMugs: Quantum Mechanical Properties of Drug-like Molecules**  
  [arXiv:2107.00367](https://arxiv.org/abs/2107.00367) | [ETH Zürich Repository (DOI)](https://doi.org/10.3929/ethz-b-000482129)






