```{image} _static/GPT_Banner_new.png
:width: 800px
:align: center
:target: [https://github.com/NOAA-National-Severe-Storms-Laboratory/BinMod1D-PARS/](https://github.com/NOAA-National-Severe-Storms-Laboratory/BinMod1D-PARS/)
```
# Introduction

{bdg-primary}`v1.0` {bdg-secondary}`Python 3.9+` {bdg-success}`NSSL`

**BinMod1D** is a versatile and convenient 1D spectral bin microphysics model that
explicitly simulates the evolution of rain or snow particle size distributions 
(PSDs) and the resulting polarimetric radar profiles due to collisional 
coalescence and breakup processes. These calculations represent a notoriously 
difficult numerical problem and requires special consideration. BinMod1D however 
makes it easy to both simulate PSD evolution in time and in height and visualize
results from a variety of different formats.


### 🛠 Model Capabilities
BinMod1D can run in three different modes:

::::{grid} 3
:gutter: 2

:::{grid-item-card} 📦 Box Model
**Time-only** simulation. Perfect for testing microphysical kernels and coalescence rates.
:::

:::{grid-item-card} 🏔 Steady-State
**Height-only** profile. Calculates the equilibrium state where $t \to \infty$.
:::

:::{grid-item-card} 🏗 1D Column
**Time & Height**. The full vertical evolution of the particle size distribution.
:::

::::


:::{sidebar} Key Highlights
* **Explicit Bin Physics:** No assumed Gamma distributions.
* **Radar Ready:** Includes polarimetric operators for Z, ZDR, and KDP.
* **Fast Solver:** Optimized for research-grade performance.
:::



