# AMR_prediction [![DOI](https://zenodo.org/badge/814745431.svg)](https://doi.org/10.5281/zenodo.17398581)
Repo for study "Biased sampling confounds machine learning prediction of antimicrobial resistance"

## Contents
1. **script**: python scripts used in this study
2. **metadata**: metadata for each genome of each species, including phenotypic information from previous studies (https://github.com/BV-BRC/AMRMetadataReview_2021, https://doi.org/10.1371/journal.pcbi.1006258), sequence types, and number of contigs. Genomes that failed CheckM quality check have been removed.
3. **itol_visualization**: files used in iTOL to visualize the tree and define clades for machine learning training
4. **clade_split**: files to classify genomes in each clade for each antibiotic
5. **doc**: result summary files used for plotting and supplement tables


## Genome FASTA downloads
* The majority of genomes used for this study are downloaded using the instructions from: https://github.com/BV-BRC/AMRMetadataReview_2021
* Additional *E. coli* genomes were downloaded based on ENA accession numbers provided in S1 from previous study (https://doi.org/10.1371/journal.pcbi.1006258), followed by assembling using velvet.

## Python Environment
This project uses **Conda** to manage dependencies and ensure reproducibility across macOS, Linux, and Windows.
### 1. Install Conda or Mamba

If you don’t have Conda installed, install one of the following:

- **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)** – the official minimal installer for Conda  

After installation, verify it works:

```bash
conda --version
```
### 2. Clone the repository
git clone https://github.com/BarquistLab/AMR_prediction.git
cd AMR_prediction

```bash
conda env create -f script/arm_spec.yml
```

