![BinMod1D Banner](https://raw.githubusercontent.com/NOAA-National-Severe-Storms-Laboratory/BinMod1D-PARS/main/assets/PyPi_Banner.png)
=======
# BinMod1D-PARS

[![Documentation Status](https://readthedocs.org/projects/binmod1d-pars/badge/?version=latest)](https://binmod1d-pars.readthedocs.io/en/latest/?badge=latest)

## 📖 Documentation

The full documentation for **BinMod1D-PARS** is available at [Read the Docs](https://binmod1d-pars.readthedocs.io/).

### Key Resources
* [**Introduction**](https://binmod1d-pars.readthedocs.io/en/latest/introduction.html) – Overview of the python package.
* [**Installation**](https://binmod1d-pars.readthedocs.io/en/latest/installation.html) – Quick start guide for local setup.
* [**Tutorial**](https://binmod1d-pars.readthedocs.io/en/latest/tutorials.html) – Quick start guide for using the **BinMod1D**
* [**Example Gallery**](https://binmod1d-pars.readthedocs.io/en/latest/examples.html) – Pre-configured notebooks that highlight the capabilities of **BinMod1D**.

---

This repository contains the BinMod1D python code. This is a python-based 1D bin (spectral) microphysical model designed to explicitly simulate collision-coalescence and collisional breakup. In order to use the code:

1.) Initialize the spectral model using the spectral_1d class: 

```python
from binmod1d.spectral_model import spectral_1d

s1 = spectral_1D()
```

The inputs to this class will specify the initial gamma particle size distribution (PSD), the bin resolution (sbin) and number of bins used (bins), as well as other parameters that determine how the model will be run.

2.) Use the spectral_1D.run() method to run the model with the inputs specified in spectral_1D()

```python
s1.run()
```

Various methods are included in the spectral_1d.py class that allow for easy plotting of bin model results.

