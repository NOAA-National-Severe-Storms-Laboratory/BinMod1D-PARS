```{image} _static/GPT_Banner_new.png
:width: 800px
:align: center
:target: https://github.com/NOAA-National-Severe-Storms-Laboratory/BinMod1D-PARS/
```
# Introduction

{bdg-primary}`v1.0` {bdg-secondary}`Python 3.9+` {bdg-success}`NSSL`

**BinMod1D** is a versatile and convenient 1D spectral bin microphysics model that
explicitly simulates the evolution of rain or snow particle size distributions 
(PSDs) and the resulting polarimetric radar profiles due to collisional 
coalescence and breakup processes. These calculations represent a notoriously 
difficult numerical problem and requires special consideration. **BinMod1D** however 
makes it easy to both simulate PSD evolution in time and in height and visualize
results from a variety of different formats.

```{image} _static/tutorial_time_height_Z_full.svg
:width: 100%
:align: center
```

## 🛠 Model Capabilities
BinMod1D can run in three different modes:

::::{grid} 3
:gutter: 3

:::{grid-item-card} 📦 Box Model
:class-card: pop-card
**Time-only** simulation. Perfect for testing microphysical kernels and coalescence and breakup rates.
:::

:::{grid-item-card} 🏔 Steady-State
:class-card: pop-card
**Height-only** profiles. Calculates the equilibrium state where $t \to \infty$.
:::

:::{grid-item-card} 🏗 1D Column
:class-card: pop-card
**Time & Height**. The full vertical evolution of the particle size distribution.
:::
::::

+++

::::{grid} 2
:gutter: 3
:class-container: sd-items-start
:class-row: sd-items-start

:::{grid-item-card}
:columns: 5
:class-card: sd-bg-light sd-shadow-sm pop-card
:class-body: sd-d-flex sd-align-items-center sd-justify-content-center
```{image} _static/tutorial_dists_height_breakup_SS_2cat.svg
:width: 100%
:align: center
```
:::

<br>

:::{grid-item-card}
:columns: 7
:class-card: sd-shadow-sm
:class-header: sd-bg-light sd-text-black sd-font-weight-bold sd-text-center
:class-body: sd-bg-white
🌟 Key Highlights
^^^

<br>
<br>

<i class="fa fa-flask sd-text-primary" aria-hidden="true" style="margin-right: 8px;"></i> 
**Explicit Bin Physics:** No assumed Gamma distributions.

<br>

<i class="fa fa-rss sd-text-warning" aria-hidden="true" style="margin-right: 8px;"></i> 
**Radar Ready:** Includes polarimetric radar operators for $Z$, $Z_{\mathrm{DR}}$, $K_{\mathrm{dp}}$, and $\rho_{\mathrm{hv}}$.

<br>

<i class="fa fa-bolt sd-text-success" aria-hidden="true" style="margin-right: 8px;"></i> 
**Fast Solver:** Optimized for research-grade performance.
:::
::::
