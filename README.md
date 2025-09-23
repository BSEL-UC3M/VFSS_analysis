# VFSS_analysis

This repository provides tools to preprocess Videofluoroscopic Swallowing Studies (VFSS), automatically label them using a deep learning–based framework, and extract dynamic parameters characteristic of dysphagia.

## Citation
If you use this repository in your research, please cite our publication:  
> **Cubero L, Tessier C, Castelli J, Robert K, de Crevoisier R, Jégoux F, Pascau J, Acosta O.**  
> *Automated dysphagia characterization in head and neck cancer patients using videofluoroscopic swallowing studies.*  
> **Comput Biol Med.** 2025 Mar;187:109759. doi: 10.1016/j.compbiomed.2025.109759. Epub 2025 Feb 6. PMID: 39914196.  
> (https://doi.org/10.1016/j.compbiomed.2025.109759)


## Installation
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
- The current version of nnU-Net used for training and inference is **version 1**. An update to **version 2** is planned.
- A sample VFSS file is provided at:  
  `VFSS_analysis/data/raw_VFSS/test/healthy_001`  
   - The file is in AVI format, compressed to preserve quality. It may not be compatible with all video players. 


For any issues or questions, please feel free to contact us:
lcubero@ing.uc3m.es ; jpascau@ing.uc3m.es ; acosta@univ-rennes.fr
