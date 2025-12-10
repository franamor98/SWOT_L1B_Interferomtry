# SWOT L1B Interferometric Height Processing

This repository contains a processing chain for
deriving interferometric heights from SWOT L1B data.

The code implements an end-to-end workflow including:
- SLC handling and nadir removal
- Interferogram formation and detrending
- Gamma-weighted multilooking
- Optional Goldstein phase filtering
- Interferometric height inversion
- Geolocation. Simplified. Spherical Earth toy model.

The processing is exposed through a single class and accompanied by
explanatory Jupyter notebooks that document each processing step.

---

## Repository structure
- src/ Intf_processor.py  # SWOT interferometric processing class
- notebooks/ Step-by-step explanatory notebooks
- environment.yml # Conda environment specification



--- 


## Installation

The recommended way to run this code is using Conda.

1. Create the environment:

```
conda env create -f environment.yml
```


2. Activate the environment:
```

conda activate swot-intf
```


## Usage

Open the notebooks in the notebooks/ directory and select the
Python (swot-intf) kernel.
The notebooks demonstrate how to run the full processing chain and inspect
intermediate interferometric products (phase, coherence, height, σ⁰).

## Notes

The code is intended for research and diagnostic purposes.


## Author

Fran Amor, UiT
