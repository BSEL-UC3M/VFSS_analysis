# VFSS_analysis

This repository provides tools to preprocess Videofluoroscopic Swallowing Studies (VFSS), label them using a deep learning-based framework, and extract several dynamic parameters characteristic of dysphagia.

## Citation
If you use this code in your work, please cite:

*To be updated with citation details.*

## Installation Steps
Follow these steps to set up the repository:

```bash
git clone https://github.com/BSEL-UC3M/VFSS_analysis.git
cd VFSS
conda env create -f environment.yml
conda activate VFSS_env
pip install -e .
```

## Important Considerations
- This repository is under active development.
- The current version of nnU-Net used for training and inference is **version 1**. An update to **version 2** is planned soon.

For any issues or questions, please feel free to contact us:
lcubero@ing.uc3m.es ; jpascau@ing.uc3m.es ; acosta@univ-rennes.fr
