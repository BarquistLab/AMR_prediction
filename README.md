# AMR_prediction
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

```
$ conda create --name <env> --file script/arm_spec.txt
```

