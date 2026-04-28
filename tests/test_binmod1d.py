import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import os

# Import modules from your package
from binmod1d.spectral_model import spectral_1d

# Define the setups we want to test: 
# 1. A single rain category
# 2. A two-category snow/fragment setup
CATEGORY_SETUPS = [
    ['rain'], 
    ['snow', 'fragments']
]

def test_spectral1d_grid_initialization():
    """
    UNIT TEST 1: Tests the domain grid generation.
    """
    model = spectral_1d(ztop=3000.0, zbot=0.0, dz=50.0, tmax=100.0, dt = 2.,progress=False)
    expected_hlen = int((3000.0 - 0.0) / 50.0) + 1
    expected_tlen = int((100.0 - 0.0) / 2.0) + 1
    assert_equal(model.Tlen, expected_tlen, 
                 err_msg=f"Time grid initialization failed! Expected {expected_tlen} levels.")
    assert_equal(model.Hlen, expected_hlen, 
                 err_msg=f"Height grid initialization failed! Expected {expected_hlen} levels.")


def test_initial_mass_positivity():
    """
    UNIT TEST 2: Tests the mass bin initialization.
    """
    model = spectral_1d(sbin=1, bins=30, tmax=0.0, progress=False)
    assert np.all(model.Mbins >= 0.0), "Negative mass detected in initialized Mbins!"

# =====================================================================
# 2. PHYSICAL CONSERVATION TESTS
# =====================================================================

@pytest.mark.parametrize("habit_list", CATEGORY_SETUPS)
def test_box_model_mass_conservation(habit_list):
    """
    UNIT TEST 3: Box Model Mass Conservation.
    Automatically runs twice: once for 1-category, once for 2-category.
    """
    dist_num = len(habit_list)
    # If 2 categories, route breakup to category 2. Otherwise route to 1.
    br_dest = 2 if dist_num > 1 else 1 
    
    # Initialize the model with the parameterized habit list
    model = spectral_1d(ztop=0., zbot=0., dt=1.0, tmax=5.0, progress=False,
                        habit_params=habit_list, dist_num=dist_num, 
                        cc_dest=1, br_dest=br_dest, kernel='Hydro', Eb=0.05) # Add Eb to test breakup transfers
    
    # Sum mass across ALL categories (axis=0) and all bins (axis=2)
    initial_total_mass = np.sum(model.Mbins[:, :, :, 0])
    
    model.run()
    
    final_total_mass = np.sum(model.Mbins[:, :, :, -1])
    
    assert_allclose(final_total_mass, initial_total_mass, rtol=1e-5,
                    err_msg=f"Box model leaked mass for habit setup: {habit_list}")

@pytest.mark.parametrize("habit_list", CATEGORY_SETUPS)
def test_steady_state_mass_conservation(habit_list):
    """
    UNIT TEST 4: Steady-state Mass Conservation.
    Automatically runs twice: once for 1-category, once for 2-category.
    """
    dist_num = len(habit_list)
    # If 2 categories, route breakup to category 2. Otherwise route to 1.
    br_dest = 2 if dist_num > 1 else 1 
    
    # Initialize the model with the parameterized habit list
    model = spectral_1d(ztop=0., zbot=0., dt=1.0, tmax=5.0, progress=False,
                        habit_params=habit_list, dist_num=dist_num, 
                        cc_dest=1, br_dest=br_dest, kernel='Hydro', Eb=0.05) # Add Eb to test breakup transfers
    
    # Sum mass across ALL categories (axis=0) and all bins (axis=2)
    initial_total_mass = np.sum(model.Mbins[:, :, :, 0])
    
    model.run()
    
    final_total_mass = np.sum(model.Mbins[:, :, :, -1])
    
    assert_allclose(final_total_mass, initial_total_mass, rtol=1e-5,
                    err_msg=f"Box model leaked mass for habit setup: {habit_list}")

@pytest.mark.parametrize("habit_list", CATEGORY_SETUPS)
def test_full_1d_mass_conservation(habit_list):
    """
    UNIT TEST 5: Full 1D Time-Height Conservation (Suspended Shaft).
    """
    dist_num = len(habit_list)
    br_dest = 2 if dist_num > 1 else 1 
    
    model = spectral_1d(ztop=500., zbot=0., dz=50., dt=0.1, tmax=0.1, progress=False,
                        habit_params=habit_list, dist_num=dist_num, 
                        cc_dest=1, br_dest=br_dest)
    
    initial_column_mass = np.sum(model.Mbins[:, :, :, 0]) * model.dz
    model.run()
    final_column_mass = np.sum(model.Mbins[:, :, :, -1]) * model.dz
    
    assert_allclose(final_column_mass, initial_column_mass, rtol=1e-5,
                    err_msg=f"1D model leaked mass during advection for {habit_list}")
# =====================================================================
# 3. I/O & DIAGNOSTICS TESTS
# =====================================================================

def test_netcdf_io(tmp_path):
    """
    UNIT TEST 6: NetCDF file writing and loading.
    Uses pytest's built-in `tmp_path` to write a temporary file, 
    load it into a new model, and verify arrays match, then deletes the file.
    """
    file_path = os.path.join(tmp_path, "test_output.nc")
    
    # Create and save model 1
    model1 = spectral_1d(progress=False)
    model1.run()
    model1.write_netcdf(file_path)
    
    # Load into model 2
    model2 = spectral_1d(load=file_path)
    
    # Verify the mass arrays survived the trip to disk and back
    assert_allclose(model1.Mbins, model2.Mbins, 
                    err_msg="NetCDF loaded Mbins do not match original Mbins!")
    assert_equal(model1.Tlen, model2.Tlen, 
                 err_msg="NetCDF loaded Tlen does not match!")

def test_radar_variable_generation():
    """
    UNIT TEST 7: Macroscopic Radar Variables.
    """
    
    model = spectral_1d(ztop=50.0, zbot=0.0, dz=25.0, tmax=0.0, 
                        radar=True, progress=False)
    model.run()
    model.calc_radar()
    
    assert hasattr(model, 'ZH'), "Radar variable ZH was not created!"
    
    # Shape should be (heights, time_out)
    expected_shape = (model.Hlen, model.Tout_len)
    assert_equal(model.ZH.shape, expected_shape, 
                 err_msg=f"ZH shape mismatch. Expected {expected_shape}")