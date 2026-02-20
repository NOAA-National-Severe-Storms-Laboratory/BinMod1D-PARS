Tutorial
=========

## Getting Started

Getting started with binmod1d is simple. Users will first initialize 
the spectral model using the **`spectral_1d()`** class: 

```python
from binmod1d.spectral_model import spectral_1d

s1 = spectral_1D()
```

The inputs to this class will specify the initial gamma particle size (or mass) distribution (PSD), 
the bin resolution (sbin) and number of bins used (bins), as well as other parameters 
that determine how the model will be run.

In general, the most relevant inputs are:

:::{grid-item-card} 
:class-header: sd-bg-light
:class-card: sd-mb-4
:class-body: sd-pb-0

* **`sbin`** {bdg-secondary}`int`  
  Resolution of geometric mass bins (default: 8).
* **`bins`** {bdg-secondary}`int`  
  Number of mass bins for the distribution (default: 140).
* **`dt`** {bdg-secondary}`float`  
  Model time step in seconds. (default:2)
* **`tmax`** {bdg-secondary}`float`  
  Maximum time in seconds the model is run (default: 800).
* **`output_freq`** {bdg-secondary}`int`  
  Frequency in with which the model is output. (default: 1)
* **`dz`** {bdg-secondary}`float`  
  Height grid spacing in meters. (default: 20.)
* **``ztop``** {bdg-secondary}`float`  
  Top height of steady-state/1D model domain. (default: 0.)
* **``zbot``** {bdg-secondary}`float`  
  Bottom height of steady-state/1D model domain. (default: 0.)
* **``D1``** {bdg-secondary}`float`  
  Minimum equivolume diameter bin size in mm when the **``dist_var``** parameter is 'size'. (default: 0.25)
* **``x0``** {bdg-secondary}`float`  
  Minimum bin mass in grams when the 'dist_var' parameter is 'mass'. (default: 0.01)
* **``moments``** {bdg-secondary}`int`  
  Number of moments to use for spectral bin model (either 1 (mass) or 2 (mass and number)). (default: 2)
* **``dist_var``** {bdg-secondary}`str`  
  Whether to use mass or size to specify initial gamma distribution. (default: 'mass')
* **``kernel``** {bdg-secondary}`str`  
  Type of collision kernel in collection_kernels.py to use for coalescence/breakup. (default: 'Golovin')
* **``frag_dist``** {bdg-secondary}`str`  
  Type of fragment distribution. (default: 'exp')
* **``ptype``** {bdg-secondary}`str`  
  Whether particles are rain or snow. (default: 'rain')
:::


After the model is initialized, users can then use the **`run()`** method 
to run the model with the inputs specified in spectral_1D:

```python
s1.run()
```

Various methods are included in the spectral_1d.py class that allow for easy 
plotting of bin model setup and results. For example, after the ```spectral_1d()```
object is initialized, users can view the initial number and mass distributions
using the ```plot_init()``` method. 
```python
s1.plot_init()
```

The following plotting methods are include:

:::{grid-item-card} 
:class-header: sd-bg-light
:class-card: sd-mb-4
:class-body: sd-pb-0

**`spectral_1d.plot_init()`**  

Plots the starting number and mass binned distribution.

**`spectral_1d.plot_dists()`**  

Plots the resulting number and mass distributions at a particular time and height.

**`spectral_1d.plot_dists_height()`**  

(steady-state and full 1D model only) Plots the resulting number distributions at a particular time for a range of heights.

**`spectral_1d.plot_moments_radar()`**  

(steady-state/Full 1D modes) Plots height profiles of number concentration, mean volume diameter, water contents, precipitation rates, and radar variables. Box model mode instead plots timeseries of each variable. 

**`spectral_1d.plot_time_height()`**  

(full 1D model only) Plots the time/height pcolor profiles for a particular microphysical or radar variable.

:::


::::{grid} 1
:class-container: sd-mb-4

:::{grid-item-card} 📏 Units & Conventions
:class-header: sd-bg-light sd-font-weight-bold
:class-card: sd-shadow-sm

**BinMod1D** primarily uses **CGS units** for internal calculations (cm, g, s). 
However, for user convenience, inputs follow these conventions:

| Quantity | Unit | Parameter Examples |
| :--- | :--- | :--- |
| **Size** | millimeters (mm) | `Dm0`, `D1` |
| **Mass** | grams (g) | `x0` |
| **Number** | per liter ($L^{-1}$) | `Nt0` |

{octicon}`info;1em;sd-text-info` *Note: This prevents the need for very large or small scientific notation for typical rain/snow cases.*
:::
::::

## Illustrative Examples

Here, I'll use the same type of simulation (i.e., continental rain) with the same 
initial gamma distribution but run in different ways. Let's assume also that users have 
a particular set of predetermined gamma size distribution parameters that govern
and initial size distribution. Often, researchers tend to (unwisely) use the 
$N_{0}$, $\Lambda$, $\mu$ convention where the gamma distribution is defined as

$$
n(D) = N_{0} D^{\mu} \exp(-\Lambda D).
$$

Of course, when parameterized in this way, it is generally difficult at a glance 
to get a sense for how these distribution parameters correlate to key properties like 
the total number and mean sizes of the initial population of particles. Therefore,
the initial gamma distribution input is instead parameterized as 

$$
n(D) = \frac{N_{t}}{\Gamma(\mu+1)} \frac{1}{D_{m}} \left( \mu+4 \right)^{\mu+1} \left( \frac{D}{D_{m}}\right)^{\mu} \exp\left[ - (\mu+4) \frac{D}{D_{m}}\right],
$$

where

$$
N_{t} = N_{0} \Lambda^{-(\mu+1)}\, \Gamma(\mu_{0}+1)

D_{m} = \frac{\mu+4}{\Lambda}

\mu = \mu.

$$
Here, $N_{t}$ is the total number concentration of the distribution whereas $D_{m}$ is
the mean volume diameter which is defined as 

$$
D_{m} \equiv \frac{M_{4}}{M_{3}} = \frac{\int\limits_{D=0}^{\infty} D^{4} n(D) dD}{\int\limits_{D=0}^{\infty} D^{3} n(D) dD}.
$$

In this tutorial, let's start with a rain distribution with the following parameters:
$N_{t,0}=10\, \mathrm{L}^{-1}$, $D_{m,0}=1.25\, \mathrm{mm}$, and $\mu_{0}=0$ (i.e., an
exponential or Marshall-Palmer distribution) and then evolve these distributions in each 
mode.

### Box model mode

First, let's set up a box model run. To do this, we need to first determine the grid
we'll be using by using the **`D1`** size parameter which specifies the initial 
left edge of the first size distribution bin. Because we are determining the grid
parameters based on size, we also need to set the **`dist_var`** parameter to 'size'
rather than 'mass'. Finally, we use the **`dt`** and **`tmax`** parameter to specify
the timestep and the maximum time we want to go out to. Finally, we can set the
**`output_freq`** parameter to 1 to let the model know that we want the final 
distribution arrays to be for all timesteps (for more involved and computionally
heavy model runs, users can set this parameter to a higher value). Let's also 
specify that we want a collision efficiency $E_{\mathrm{col}}$ of unity and 
a coalescence efficiency of $0.2$ and that we want the hydrodynamic kernel. 
We'll also run the model out to 30 minutes to get a nice timeseries when we 
plot some results.

```python
rain_box = spectral_1d(sbin=1,bins=40,D1=0.01,dt=2.,tmax=1800.,Nt0=10.,Dm0=1.25,mu0=0.,habit_list=['rain'],ptype='rain',kernel='Hydro',Ecol=1.0,Es=0.2,radar=True,dist_var='size')
rain_box.run()
```

This run should be really quick for an **`sbin`** value of 1 and only one distribution.
Now let's look at the results. First, let's use the **`plot_dists()`** method to plot 
the initial and final number and mass distribution functions

```python
rain_box.plot_dists(x_axis='size')
```
```{image} _static/tutorial_dist_log_box.svg
:width: 600px
:align: center
```

Notice that with collision-coalescence on, the number and mass distributions shift 
toward larger sizes from the initial time to $30$ minutes. The default plotting 
for **`plot_dists()`** is in terms of $dN/d\log(D)$ and $dM/d\log(D)$ (i.e., `distscale='log'`) 
because this reliably produces nice bell-shaped distributions. However, we can use some of the 
optional input parameters to customize the plots a bit more in the style of what 
are typically shown in journal articles. Often, researchers will plot these 
distributions in a semi-log fashion where the $x$-axis is linearly spaced
but the $y$-axis is log-spaced. Exponential distributions (like our initial
Marshall-Palmer one) will appear as straight lines in the number distribution 
plots when plotted in this way. We can reproduce the number and mass distribution
plots in a semi-log fashion by using the `xscale` and `yscale` input parameters along 
with `distscale='linear'`

```python
rain_box.plot_dists(-1,x_axis='size',xscale='linear',yscale='log',distscale='linear')
```

```{image} _static/tutorial_dist_lin_log_box.svg
:width: 600px
:align: center
```

We can also plot these distributions in a simple linear-linear style plot as well

```python
rain_box.plot_dists(-1,x_axis='size',xscale='linear',yscale='linear',distscale='linear')
```

```{image} _static/tutorial_dist_lin_box.svg
:width: 600px
:align: center
```

The number distribution function becomes a lot lower at $30$ minutes
due to collision-coalescence which rapidly decreases the number of raindrops with time. 
We can better compare the initial and final number distribution functions by using
`normbin=True` which normalizes the number (mass) distribution by the total number (mass)

```python
rain_box.plot_dists(-1,x_axis='size',xscale='linear',yscale='linear',distscale='linear',normbin=True)
```

```{image} _static/tutorial_dist_lin_norm_box.svg
:width: 600px
:align: center
```

We can also look at the timeseries of the moments by using the **`plot_moments_radar()`** method:

```python
rain_box.plot_moments_radar()
```
Notice that the liquid water content is constant throughout whereas the other variables
increase or decrease approximately with generalized power-law behaviors.

```{image} _static/tutorial_moments_box.svg
:width: 100%
:align: center
```


### Steady-state mode

Now we'll replicate the box model run in the steady-state mode. To run the model
as a steady-state model we can simply change a few of the inputs to the **`spectral_1d()`**
initialization. In particular, we'll use `tmax=0` to indicate that we aren't using
the time domain at all (this indicates to **BinMod1D** that the steady-state mode
is what we want) and we'll set up the height domain by specifying the height grid
spacing as `dz=20.` and the top and bottom heights as `ztop=3000.` and `zbot=0.` 
where each variable is in meters. 

```python
rain_SS = spectral_1d(sbin=1,bins=40,D1=0.01,tmax=0.,dz=20.,ztop=3000.,zbot=0.,Nt0=10.,Dm0=1.25,mu0=0.,habit_list=['rain'],ptype='rain',kernel='Hydro',Ecol=1.0,Es=0.2,radar=True,dist_var='size')
rain_SS.run()
```
Now we can plot the distributions as before to get a sense for what the final distributions look like. 
We can use the **`plot_dist()`** like before. **`plot_dists()`** can plot any time or 
height index using the `tind` or `hind` parameters. By default, the function plots the final
distribution (i.e., `tind=hind=-1`) so we can just call the method like before without any arguments.

```python
rain_SS.plot_dists()
```

```{image} _static/tutorial_dist_log_SS.svg
:width: 600px
:align: center
```
We can also reuse the **`plot_moments_radar()`** method to plot the steady-state
height profiles for all variables

```python
rain_SS.plot_moments_radar()
```

```{image} _static/tutorial_moments_SS.svg
:width: 100%
:align: center
```
Notice that the mass flux, instead of the total mass, is conserved for all heights. 
Other variables decrease or increase in a generalized power-law-like way.

We can see the evolution of the particle size distributions with height by using
the **`plot_dists_height()`** method. We'll create a figure with 3 subpanels where
we'll plot the size distribution at $3.0$ km, $1.5$ km, and $0$ km. To do this,
we'll use the `dz` input which specifies the dz spacing for each subplots between
`ztop` and `zbot`.

```python
rain_SS.plot_dists_height(dz=1.5)
```
```{image} _static/tutorial_dists_height_SS.svg
:width: 600px
:align: center
```

Now let's try to incorporate breakup. We'll set the breakup efficiency to $0.035$ and we'll
raise the coalescence efficiency to $0.8$ so that we can better see the effects of 
combined coalescence and breakup. We'll also choose a lognormal distribution to represent
fragments.

```python
rain_breakup_SS = spectral_1d(sbin=1,bins=40,D1=0.01,tmax=0.,dz=20.,ztop=3000.,zbot=0.,Nt0=10.,Dm0=1.25,mu0=0.,habit_list=['rain'],frag_dist='LGN',ptype='rain',kernel='Hydro',Ecol=1.0,Es=0.8,Eb=0.035,radar=True,dist_var='size')
rain_breakup_SS.run()
```

Now let's plot the resulting moments and radar variables as well as the same 
distribution plot as before with height

```{image} _static/tutorial_moments_breakup_SS.svg
:width: 100%
:align: center
```


```{image} _static/tutorial_dists_height_breakup_SS.svg
:width: 600px
:align: center
```


### Full time/height mode (1D Column Model)

Finally, let's do a full 1D model run with both coalescence and breakup turned on. To
initialize a full 1D column model simulation, users just need to modify both the
time (`tmax`) and top and bottom (`ztop` and `zbot`) input values. Let's use the same `tmax` 
parameter (i.e., 30 minute simulation) along with the same height grid as in the
steady-state example. We'll also fix the top boundary conditon with `boundary='fixed'` 
in order for the model approach a steady-state solution after a sufficiently long
period of time. Note that **BinMod1D** uses numba functionality which will parallelize
calculations if users have multiple CPUs. Therefore, even this simulation will not take
very long on most modern PCs.

```python
rain_breakup_full = spectral_1d(sbin=1,bins=40,D1=0.01,dt=2,tmax=1800.,dz=20.,ztop=3000.,zbot=0.,Nt0=10.,Dm0=1.25,mu0=0.,habit_list=['rain'],frag_dist='LGN',ptype='rain',kernel='Hydro',boundary='fixed',Ecol=1.0,Es=0.8,Eb=0.035,radar=True,dist_var='size')
rain_breakup_full.run()
```

Now we can plot the full time/height reflectivity profile by using the **`plot_time_height()`**
method. Reflectivity is plotted by default so we just need to call the function with no arguments.

```python
rain_breakup_full.plot_time_height()
```

```{image} _static/tutorial_time_height_Z_full.svg
:width: 100%
:align: center
```

This simulation shows a rapid increase in simulated radar reflectivity toward the
ground during the initial rain evolution due to rapid size sorting of large drops 
sedimenting and then a leveling off of reflectivity afterward in a quasi-steady state. 
We can use **`plot_moments_radar()`** to directly overlay the height profiles of the 
distribution and radar variables at $30$ minutes with the steady-state solution from before.
To do this, we first return the pyplot figure and axis when plotting the full 1D results.
Then, we can use this axis handle as an input to the steady-state plotting. Let's set `lstyle='--`
when we plot the steady-state solution. This will plot the steady-state profiles with a dashed line
whereas the full 1D column model will be plotted as a solid line.

```python
fig, ax = rain_breakup_full.plot_moments_radar()
rain_breakup_SS.plot_moments_radar(ax=ax,lstyle='--')
``` 

```{image} _static/tutorial_moments_full_SS.svg
:width: 100%
:align: center
```

Finally we can plot the time/height profiles of the other radar variables using
the `var` input parameter. If `radar=True` in the **`spectral_1d()`** object then
users can plot any of the following variables: `var=Z`, `var=ZDR`, `var=KDP`, or
`var=RHOHV`. Users can also plot any of the bulk microphysical properties as well
(i.e., `var='Nt`, `var=Dm`, `var=WC`, `var=R`). Let's plot differential reflecivity 
$Z_{\mathrm{DR}}$ and specific differential phase $K_{\mathrm{dp}}$

```python
rain_breakup_full.plot_time_height(var='ZDR')
rain_breakup_full.plot_time_height(var='KDP')
```

```{image} _static/tutorial_time_height_ZDR_full.svg
:width: 100%
:align: center
```
```{image} _static/tutorial_time_height_KDP_full.svg
:width: 100%
:align: center
```


## Customizing the model

### Handling multiple distributions

**BinMod1d** can easily incorporate as many distributions as users want. To specify multiple 
distributions, users can use the `habit_list` input parameter. The `habit_list` parameter
is a list of strings where each string dictates the type of habit that represents each
distribution (e.g., `['snow','fragments']`). Therefore, the length of the `habit_list` indicates the number of 
requested distributions. The individual strings represent the dictionary that's used to determine
the distributions properties (see the following section). Users can then specify which distribution receives coalesced
particles by using the `cc_dest` parameter (value of 1 signifies the first distribution). 
Similarly, the `br_dest` determines the location of breakup (fragmented) particles. 
Here, let's keep the coalesced particles with the same distribution as the initial
distribution but we'll put the fragmented particles in a second distribution. Let's
use the steady-state coalescence/breakup example from before.

```python
rain_breakup_SS_2cat = spectral_1d(sbin=1,bins=40,D1=0.01,tmax=0.,dz=20.,ztop=3000.,zbot=0.,Nt0=10.,Dm0=1.25,mu0=0.,habit_list=['rain','rain'],frag_dist='LGN',cc_dest=1,br_dest=2,ptype='rain',kernel='Hydro',Ecol=1.0,Es=0.8,Eb=0.035,radar=True,dist_var='size')
rain_breakup_SS_2cat.run()
```
We'll use the `plot_habits` input parameter in the plots to specify that we want to see the individual
distributions as well as the combined distribution variables.

```python
rain_breakup_SS_2cat.plot_moments_radar(plot_habits=True)
rain_breakup_SS_2cat.plot_dists_height(dz=1.5,plot_habits=True)
```

```{image} _static/tutorial_moments_breakup_SS_2cat.svg
:width: 100%
:align: center
```


```{image} _static/tutorial_dists_height_breakup_SS_2cat.svg
:width: 600px
:align: center
```

### Using custom distribution and habit parameters

By default, **BinMod1D** has three example habit distributions with predetermined
values: `'rain'`, `'snow'`, and `'fragments'`. These dictionaries are available from the `binmod1d.habits` module.

```python
from binmod1d.habits import habits, fragments
habit_dict = habits()
```
The keys of these dictionaries represent the specified habits that users
can use to describe the properties of the distribution for each category. 
For example, the 'snow' key shows
```python
  {'arho': 0.2,
  'brho': 1.0,
  'av': 0.8,
  'bv': 0.14,
  'ar': 0.6,
  'br': 0.0,
  'sig': 0.0,
  'am': 0.00010471975511965977,
  'bm': 2.0}
```
Here, `arho` and `brho` represent the density-size power-law relation 
($\rho (D) = \alpha_{\rho} D^{-\beta_{\rho}}$), `av` and `bv` represent the fallspeed-size power-law relation 
($v_{t}(D) = a_{v} D^{b_{v}}$), `ar` and `br` represent the spheroidal aspect ratio-size power-law relation 
($\varphi (D)= a_{r} D^{br}$), `sig` is the two-dimensional Gaussian orientation standard deviation
parameter in degrees (see Ryzhkov et al. (2011)) where `sig=0` is horizontally oriented and `sig=40` is chaotically oriented,
and `am` and `bm` are the mass-dimensional power-law parameters ($m(D) = \alpha_{m} D^{\beta_{m}}$). Users only
need to modify the `habit_list` input parameter to **`spectral_1d()`**. 

For example, we can create a new habit dictionary for ice aggregates using the 
"Aggregates of densely rimed radiating assemblages of dendrites or dendrites" category
from table 1 of Locatelli and Hobbs (1974)

```python
 agg_dict = {'av': 0.79,
   'bv': 0.27,
   'ar': 0.6,
   'br': 0.0,
   'sig': 10.0,
   'am': 3.7e-05,
   'bm': 1.9}
```

Then the dictionary can be passed in a list in the **`spectral_1d()`** initialization call

```python
s_agg = spectral_1d(habit_list=[agg_dict])
```

Note that users only need to specify either `am` and `bm` or `arho` and `brho`;
the mass or density parameters will be determined by either pair.


### Reading and writing netcdf outputs

After users run their model with **`run()`**, they can easily save their model 
results using the **`write_netcdf()`** method. 

Let's use the first steady-state Jupyter notebook test to demonstrate this. 

```python
from binmod1d.spectral_model import spectral_1d
s3_SS = spectral_1d(sbin=3,bins=160,D1=0.001,tmax=0.,Nt0=15.,Dm0=0.8,mu0=0.,dz=10.0,ztop=1000.,zbot=0.,habit_list=['rain'],ptype='rain',kernel='Hydro',Ecol=1.0,Es=0.8,radar=True,dist_var='size')
s3_SS.run()
```

Now users can write a netcdf file with all the simulated distribution function parameters
to a file by simply running **`write_netcdf(filename)`** where ```filename``` is the full
path to the directory and filename where they would like to save the netcdf file

```python
s3_SS.write_netcdf('C://Users/username/Documents/BinMod1D/s3_SS.nc')
```

Loading previous model run netcdf files is trivial. To do so, just use the ```load``` 
input parameter in the spectral_1d() class without any other inputs.

```python
s3_SS_load = spectral_1d(load='C://Users/username/Documents/BinMod1D/s3_SS.nc')
```

**`spectral_1d()`** will read the 4D binned mass (or mass and number if ```moments=2```)
distribution for all output times and heights, any input parameters that were used 
during the initialization, and will recalculate the linear subgrid distribution parameters,
and bulk microphysical and radar (if ```radar=True``` was specified
during initialization) variables.

## References

Locatelli, J. D. and P. V. Hobbs: Fall speeds and masses of solid precipitation particles,
*J. Geophys. Res.*, **79**, 2185--2197, 1974, https://doi.org/10.1029/JC079i015p02185.  

Ryzhkov, A., Pinsky,M., Pokrovsky, A., and Khain, A.: Polarimetric radar observation operator for a cloud model with spectral microphysics,
*J. Appl. Meteor. Climatol.*, **50**, 873–894, 2011, https://doi.org/10.1175/2010JAMC2363.1. 