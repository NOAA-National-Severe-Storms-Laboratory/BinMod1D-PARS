# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 09:06:50 2025

@author: edwin.dunnavan
"""

import numpy as np

from .bin_integrals import gam_int, LGN_int, In_int, GAU_int
from .collection_kernels import Straub_params

from scipy.special import erfinv
from scipy.stats import gamma

# from .distribution import extended_brandes, rain_terminal_velocity


def habits():
    """
    Sample habit dictionary parameters that can be used in 'habit_params' input
    for spectral_1d class.

    Returns
    -------
    habits : Dict
        Dictionary of habit parameters.

    """

    habits = {}

    habits["rain"] = {
        "arho": 1.0,  # g/cm^3
        "brho": 0.0,
        "av": 3.78,  # Atlas and Ulbrich (1977)
        "bv": 0.67,  # Atlas and Ulbrich (1977)
        "ar": 1.0,
        "br": 0.0,
        "sig": 10.0,
    }
    #'aspect_ratio':lambda d: extended_brandes(d)}
    #'vt':lambda d: rain_terminal_velocity(d)}

    habits["rain"]["am"] = (
        0.001 * (np.pi / 6.0) * habits["rain"]["arho"]
    )  # units of g/mm^3
    habits["rain"]["bm"] = 3.0 - habits["rain"]["brho"]

    habits["snow"] = {
        "arho": 0.2,  # g/cm^3 * 1/mm^brho
        "brho": 1.0,
        "av": 0.8,
        "bv": 0.14,
        "ar": 0.6,
        "br": 0.0,
        "sig": 0.0,
    }

    habits["snow"]["am"] = 0.001 * (np.pi / 6.0) * habits["snow"]["arho"]
    habits["snow"]["bm"] = 3.0 - habits["snow"]["brho"]

    habits["fragments"] = {
        "arho": 0.6,
        "brho": 0.0,
        "av": 0.8,
        "bv": 0.14,
        "ar": 0.8,
        "br": 0.0,
        "sig": 20.0,
    }

    habits["fragments"]["am"] = 0.001 * (np.pi / 6.0) * habits["fragments"]["arho"]
    habits["fragments"]["bm"] = 3.0 - habits["fragments"]["brho"]

    return habits


def fragments(dist="exp", **kwargs):
    """
    This function can be used to generate example fragment distribution functions
    that spectral_1d() can use based on the desired dist input parameter string.
    The format of each output dictionary has the same structure. For example,
    the lognormal fragment distribution has the following default dictionary:

        {'dist': 'LGN',
         'var': 'size',
         'Df_med': 0.25,
         'Df_mode': 0.2,
         'cdf_bounds': (0.5, 0.95),
         'd_start': np.float64(0.25),
         'd_end': np.float64(0.5437328654539426),
         'func': <function binmod1d.habits.fragments.<locals>.<lambda>(n, c, pi, pj)>}

        Here, 'dist' is the name of the distribution, 'var' is whether the fragment
        distribution is defined in terms of mass ('mass') or size ('size'),
        'Df_med' and 'Df_mode' are the lognormal distribution parameters,
        cdf_bounds are the requested percentile bounds of the lognormal distribution
        where the distribution will become active. The d_start and d_end parameters
        correspond to the 50% and 95% percentiles where d<d_start will not be active
        d_start<d<d_end will ramp from 0 to Eb and d>d_end will have Eb.
        The lambda function 'func' is the output conditional fragment distribution
        BIN-WISE MOMENT FUNCTION:

                            (mass) int_xi1^xi2 m^n P(m|x,y) dm

                                            or

                            (size) int_d1^d2 d_m^n P(d_m|d_x,d_y) dd_m


        that Interaction() will use. Users who want to create their own fragment
        distributions simply need to structure their lambda function as follows:

                                FOR 'var' = 'size'
            func = lambda n, c, pi, pj: dist_int(*params,n,c.d1,c.d2)

                                FOR 'var' = 'mass'
            func = lambda n, c, pi, pj: dist_int(*params,n,c.xi1,c.xi2),

        where c corresponds to the fragment distribution (i.e., dists[indb]) whereas
        pi and pj correspond to the i and j distributions (i.e., dists[d1_indices] and dists[d2_indices]).

        Note, for marginal distributions which use a static (yet truncated)
        distribution (as with the examples shown here), pi and pj do not need
        to be used in the lambda function but still need to be inputs. For
        conditional fragment distributions of the form p(m|x,y), users can use
        the pi and pj object inputs to structure the conditional nature of the
        multivariate distribution function. For example, if the conditional
        distribution depends upon the i and j bin midpoint fall speeds, then
        the lambda function structure might be something like:

            func = lambda n, c, pi, pj: dist_int(n,c.d1,c.d2,pi.vt,pj.vt).


    Parameters
    ----------
    dist : str, optional
        Name of default distribution. The default is 'exp'.
    **kwargs : dict
        Extra parameters.


    Returns
    -------
    fragments : dict
        Dictionary of fragment parameters and lambda function.

    """

    if dist == "exp":
        Dmf = kwargs.pop("Dmf", 0.25)
        muf = 0.0
        var = "size"
        bounds = kwargs.pop("bounds", None)
        cdf_bounds = kwargs.pop("cdf_bounds", None)
        parent_cutoff = kwargs.pop("parent_cutoff", 0.0)

        # IF_func = lambda n,x1,x2: gam_int(n,muf,Dmf,x1,x2)

        IF_func = lambda n, c, pi, pj: gam_int(n, muf, Dmf, c.d1, c.d2)

        Dn = Dmf / (muf + 4.0)

        if isinstance(bounds, tuple):
            d_start = bounds[0]
            d_end = bounds[1]
        else:
            d_start = None
            d_end = None

        if isinstance(cdf_bounds, tuple):

            nuf = muf + 1.0

            d_start = gamma.ppf(cdf_bounds[0], a=nuf, scale=Dn)
            d_end = gamma.ppf(cdf_bounds[1], a=nuf, scale=Dn)

            bounds = None

        else:
            cdf_bounds = None

        fragments = {
            "dist": dist,
            "var": var,
            "Dmf": Dmf,
            "cdf_bounds": cdf_bounds,
            "parent_cutoff": parent_cutoff,
            "d_start": d_start,
            "d_end": d_end,
            "func": IF_func,
        }

    elif dist == "exp_mass":

        lamf = kwargs.pop("lamf", 10.0)
        parent_cutoff = kwargs.pop("parent_cutoff", 0.0)
        var = "mass"

        # IF_func = lambda n,x1,x2: In_int(n,lamf,x1,x2)

        IF_func = lambda n, c, pi, pj: In_int(n, lamf, c.xi1, c.xi2)

        fragments = {
            "dist": dist,
            "var": var,
            "lamf": lamf,
            "cdf_bounds": None,
            "parent_cutoff": parent_cutoff,
            "func": IF_func,
        }

    elif dist == "gamma":

        Dmf = kwargs.pop("Dmf", 0.25)
        muf = kwargs.pop("muf", 3.0)
        var = "size"
        bounds = kwargs.pop("bounds", None)
        cdf_bounds = kwargs.pop("cdf_bounds", None)

        parent_cutoff = kwargs.pop("parent_cutoff", 0.0)

        # IF_func = lambda n,x1,x2: gam_int(n,muf,Dmf,x1,x2)

        IF_func = lambda n, c, pi, pj: gam_int(n, muf, Dmf, c.d1, c.d2)

        Dn = Dmf / (muf + 4.0)

        if isinstance(bounds, tuple):
            d_start = bounds[0]
            d_end = bounds[1]
        else:
            d_start = None
            d_end = None

        if isinstance(cdf_bounds, tuple):

            nuf = muf + 1.0

            d_start = gamma.ppf(cdf_bounds[0], a=nuf, scale=Dn)
            d_end = gamma.ppf(cdf_bounds[1], a=nuf, scale=Dn)

            # cdf_bounds = (d_start,d_end)
            bounds = None

        else:
            cdf_bounds = None

        fragments = {
            "dist": dist,
            "var": var,
            "Dmf": 0.25,
            "muf": 3.0,
            "cdf_bounds": cdf_bounds,
            "parent_cutoff": parent_cutoff,
            "d_start": d_start,
            "d_end": d_end,
            "func": IF_func,
        }

    elif dist == "LGN":

        bounds = kwargs.pop("bounds", None)
        # cdf_bounds = kwargs.pop('cdf_bounds',(0.5,0.95))
        cdf_bounds = kwargs.pop("cdf_bounds", None)
        # pbounds = kwargs.pop('pbounds',(0.5,1.0))

        pbounds = kwargs.pop("pbounds", None)
        Df_med = kwargs.pop("Df_med", 0.25)
        Df_mode = kwargs.pop("Df_mode", 0.2)
        var = "size"

        muf = np.log(Df_med)
        sig2f = muf - np.log(Df_mode)

        if isinstance(bounds, tuple):
            d_start = bounds[0]
            d_end = bounds[1]
        else:
            d_start = None
            d_end = None

        if isinstance(cdf_bounds, tuple):
            d_start = np.exp(muf + np.sqrt(2 * sig2f) * erfinv(2.0 * cdf_bounds[0] - 1))
            d_end = np.exp(muf + np.sqrt(2 * sig2f) * erfinv(2.0 * cdf_bounds[1] - 1))

        else:
            cdf_bounds = None

        # IF_func = lambda n,x1,x2:LGN_int(n,muf,sig2f,x1,x2)

        IF_func = lambda n, c, pi, pj: LGN_int(n, muf, sig2f, c.d1, c.d2)

        fragments = {
            "dist": dist,
            "var": var,
            "Df_med": Df_med,
            "Df_mode": Df_mode,
            "cdf_bounds": cdf_bounds,
            "pbounds": pbounds,
            "d_start": d_start,
            "d_end": d_end,
            "func": IF_func,
        }

    elif dist == "Straub":

        # bounds = kwargs.pop('bounds',(0.1,0.25))
        bounds = kwargs.pop("bounds", None)
        # cdf_bounds = kwargs.pop('cdf_bounds',None)
        # cdf_bounds = kwargs.pop('cdf_bounds',(0.5,0.95))

        # cdf_bounds = kwargs.pop('cdf_bounds',(0.5,0.95))
        cdf_bounds = kwargs.pop("cdf_bounds", None)
        # ORIG
        # pbounds = kwargs.pop('pbounds',(1.5,2.5))

        # Gradual
        pbounds = kwargs.pop("pbounds", (1.2, 2.8))

        # pbounds = kwargs.pop('pbounds',(2.2,2.5))
        # pbounds = kwargs.pop('pbounds',None)
        # pbounds = kwargs.pop('pbounds',(0.1,0.15))
        var = "size"

        if isinstance(bounds, tuple):
            d_start = bounds[0]
            d_end = bounds[1]
        else:
            d_start = None
            d_end = None

        # Get Straub parameters. Note, for simplicity just use bin midpoints
        # To get Straub's four fragment distribution parameters. Ideally, this
        # would be done in some clever way by taking into account all mass
        # combinations between the bin limits.

        state = {}

        params = lambda pi, pj: Straub_params(
            pi.d, pj.d, pi.vt, pj.vt, cdf_bounds=cdf_bounds, state=state
        )

        IF_func = lambda n, c, pi, pj: straub_wrapper(n, c, pi, pj)

        fragments = {
            "dist": dist,
            "var": var,
            "cdf_bounds": cdf_bounds,
            "pbounds": pbounds,
            "d_start": d_start,
            "d_end": d_end,
            "state": state,
            "params": params,
            "func": IF_func,
        }

    else:
        raise RuntimeError("Distribution not currently implemented!")

    return fragments


# --- Define the unified wrapper function ---
def straub_wrapper(n, c, pi, pj, cdf_bounds=None, state=None):
    """
    Evaluates the full 4-part Straub distribution efficiently in 4D.
    """
    # 1. Calculate parameters exactly ONCE
    sp = Straub_params(pi.d, pj.d, pi.vt, pj.vt)

    # CHANGE THIS FROM 0.1!
    sig2_min = 1e-5

    n1 = sp["dist1"]["N"]
    muf1 = sp["dist1"]["muf"]
    sig2f = np.maximum(sp["dist1"]["sig2f"], sig2_min)

    n2 = sp["dist2"]["N"]
    mu2 = sp["dist2"]["mu"]
    sig2_2 = np.maximum(sp["dist2"]["sig2"], sig2_min)

    n3 = sp["dist3"]["N"]
    mu3 = sp["dist3"]["mu"]
    sig2_3 = np.maximum(sp["dist3"]["sig2"], sig2_min)

    # n1[sig2f==sig2_min]  = 0.
    # n2[sig2_2==sig2_min] = 0.
    # n3[sig2_3==sig2_min] = 0.

    d0 = np.min(c.d1)
    # ds   = np.minimum(pi.d,pj.d)
    dl = np.maximum(pi.d, pj.d)

    if cdf_bounds is not None:
        p_left = cdf_bounds[0]
        # Calculate dynamic 4D left-bound using the inverse CDF
        d_start_1 = np.exp(muf1 + np.sqrt(2.0 * sig2f) * erfinv(2.0 * p_left - 1.0))
        # Clamp it between your absolute grid minimum and the physical maximum (dl)
        d_start_1 = np.clip(d_start_1, d0, dl)
    else:
        d_start_1 = np.min(c.d1)

    bm_2 = np.broadcast_to(c.bm, sig2_2.shape)
    bm_3 = np.broadcast_to(c.bm, sig2_3.shape)

    ms = np.minimum(pi.am * pi.d**pi.bm, pj.am * pj.d**pj.bm)
    ml = np.maximum(pi.am * pi.d**pi.bm, pj.am * pj.d**pj.bm)

    # Find total mass (without prefactor) of all fragments for the three breakup modes
    M31 = n1 * c.am * LGN_int(c.bm, muf1, sig2f, d0, dl)
    M32 = n2 * c.am * GAU_int(bm_2, mu2, sig2_2, d0, dl)
    M33 = n3 * c.am * GAU_int(bm_3, mu3, sig2_3, d0, dl)

    # Residual mass (without prefactors; just need to map to original grid)
    M_frag_total = M31 + M32 + M33

    M_parent_total = ml + ms

    mass_overshoot = (
        M_frag_total > ms
    )  # Fragment total can't exceed the smaller i,j mass.

    scale_factor = np.ones_like(M_parent_total)

    scale_factor[mass_overshoot] = (
        M_parent_total[mass_overshoot] / M_frag_total[mass_overshoot]
    )

    M_frag_scaled = M_frag_total * scale_factor

    shed_fraction = M_frag_scaled / M_parent_total

    f_min = 1e-4
    f_max = 5e-4

    # OLD
    # f_min = 0.01
    # f_max = 0.015

    is_significant = np.clip((shed_fraction - f_min) / (f_max - f_min), 0.0, 1.0)

    x_res = np.maximum(0.0, M_parent_total - M_frag_scaled)

    # 2. Evaluate Continuous Distributions (LGN and 2 Gaussians)
    N1 = n1 * LGN_int(n, muf1, sig2f, c.d1, c.d2)
    N2 = n2 * GAU_int(n, mu2, sig2_2, c.d1, c.d2)
    N3 = n3 * GAU_int(n, mu3, sig2_3, c.d1, c.d2)

    # 3. Handle the Residual Drop (Dirac Delta)
    # Calculate residual mass
    in_bin_mask = (x_res >= c.xi1) & (x_res < c.xi2) & (x_res > 1e-16)

    D_res = (x_res / c.am) ** (1.0 / c.bm)

    # Return D_res^n inside the correct bin
    N4 = (D_res**n) * in_bin_mask

    # sig2_res = np.maximum((0.02*D_res)**2,1e-12)

    # N4 = GAU_int(n,D_res,sig2_res,c.d1,c.d2)

    # valid_res_mask = (x_res>1e-12)

    # N4 *= valid_res_mask

    if state is not None:
        state["is_significant"] = is_significant

    N1 *= scale_factor * is_significant
    N2 *= scale_factor * is_significant
    N3 *= scale_factor * is_significant
    N4 *= is_significant

    # NTotal = N1+N2+N3+N4

    # 4. Sum and return
    return N1 + N2 + N3 + N4

    # return N1 + N2 + N3

    # return N4
