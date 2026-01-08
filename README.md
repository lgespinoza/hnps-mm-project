## Repository Structure and Contents

This repository contains all input files and analysis scripts used for the construction and characterization of hybrid nanoparticles and membrane systems.

### `hnps_files/`
Contains all **PDB structures of the nanoparticles (hNPs)** used throughout the simulations.  
These files define the full nanoparticle architecture employed in system construction and analysis.

### `membrane_files/`
Includes all **PDB structures of the membrane systems**, corresponding to the different membrane compositions used in the study.

### `itp/`
Collects **all ITP topology files** used in the simulations, including nanoparticle components, lipids, and auxiliary parameters required for system setup in GROMACS.

### `masterscript_analysis.py`
Main analysis script used to:
- Compute **nanoparticle–membrane contacts**
- Quantify **membrane deformation**
- Process and aggregate data across replicas

This script constitutes the core of the post-processing and quantitative analysis workflow.

### `README.md`
Updated documentation reflecting the current repository structure and contents.

---

For questions or further information, please contact:  
**luis.espinoza-@utalca.cl**
