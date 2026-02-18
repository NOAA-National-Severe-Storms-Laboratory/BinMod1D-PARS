Tutorial
=========

## Getting Started

Getting started with binmod1d is incredibly simple. Users will first initialize 
the spectral model using the spectral_1d class: 

```python
from binmod1d.spectral_model import spectral_1d

s1 = spectral_1D()
```

The inputs to this class will specify the initial gamma particle size distribution (PSD), 
the bin resolution (sbin) and number of bins used (bins), as well as other parameters 
that determine how the model will be run.

After the model is initialized, users can then use the run method 
to run the model with the inputs specified in spectral_1D:

```python
s1.run()
```

Various methods are included in the spectral_1d.py class that allow for easy 
plotting of bin model results including:

```python
s1.plot_init()
```
This method plots the initial particle size distribution.
   
   
## A Quick Example


## Handling Multiple Distributions

## Running BinMod1D in Different Modes

### The Box Model mode

### The Steady-State Mode

### The Full 1D Time/Height Mode