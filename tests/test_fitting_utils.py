import numpy as np
import pytest
import os
import tempfile
from fwdd.fitting_utils import (
    find_C_t_crossing_points,
    find_time_range_for_C_t_bounds,
    analyze_intervals,
    find_widest_contiguous_stretch,
    format_parameters,
    get_bootstrap_sample,
    bootstrap_multiple_samples,
    calculate_total_combinations,
    create_combined_analysis_plot,
)
from fwdd.noise_spectra import noise_spectrum_combination

def test_find_widest_contiguous_stretch():
    arr = np.array([0.1, 0.5, 0.6, 0.2, 0.9, 0.1, 0.5])
    # [0.5, 0.6] -> contiguous stretch inside [0.4, 0.7] is indices 1, 2 (length 2)
    # [0.5] at the end is length 1.
    res = find_widest_contiguous_stretch(arr, lb=0.4, ub=0.7)
    assert res == [1, 2]

def test_format_parameters():
    args_list = [
        {"A": [1.2345], "alpha": [0.8912]},
        {"omega_0": [2.0], "gamma": [1.0], "A": [1.5]},
        {"C": [0.5]},
        {"A": [1.0], "alpha": [1.0], "beta": [2.0], "gamma": [50.0]}
    ]
    res = format_parameters(args_list, break_lines=False)
    assert "A=1.23" in res
    assert "α=0.89" in res
    assert "C=0.50" in res

def test_find_C_t_crossing_points_and_bounds():
    # Simple check on crossing point finding using delta approx
    finite_width_params = {"N": 8, "tau_p": 0.024, "integration_method": "trapezoid"}
    noise_params = ({"A": [1.0], "alpha": [1.0]}, {}, {}, {}) # f_params with A=1.0, alpha=1.0
    
    crossings = find_C_t_crossing_points(
        C_t_max=0.9,
        C_t_min=0.2,
        t_min=0.1,
        t_max=10.0,
        num_scan_points=50,
        noise_spectrum=noise_spectrum_combination,
        noise_params=noise_params,
        delta=True,
        **finite_width_params
    )
    assert "leftmost_C_t_max" in crossings
    assert "rightmost_C_t_min" in crossings

    t_min, t_max = find_time_range_for_C_t_bounds(
        C_t_max=0.9,
        C_t_min=0.2,
        finite_width_params=finite_width_params,
        noise_spectrum=noise_spectrum_combination,
        noise_params=noise_params,
        delta=True
    )
    assert t_min is not None
    assert t_max is not None
    assert t_max >= t_min

def test_analyze_intervals():
    intervals = np.array([[1.0, 5.0], [2.0, 6.0]])
    res = analyze_intervals(intervals, print_results=False)
    assert len(res) > 0
    # res is list of tuples: (start, end, containing_intervals_list)
    assert res[0][0] == 1.0
    assert res[0][1] == 2.0
    assert res[0][2] == [0]

def test_bootstrapping():
    # Setup mock data dictionary
    data = {
        8: {
            "run0": {"A": 1.0, "alpha": 1.1},
            "run1": {"A": 1.2, "alpha": 0.9}
        },
        128: {
            "run0": {"A": 2.0, "alpha": 1.0}
        }
    }
    
    total = calculate_total_combinations(data)
    assert total == 2 # 2 runs for 8, 1 run for 128 -> 2 * 1 = 2
    
    sample = get_bootstrap_sample(data)
    assert 8 in sample
    assert 128 in sample
    assert len(sample[8]) == 2
    
    samples = bootstrap_multiple_samples(data, num_bootstrap=5)
    # Since total combinations is 2, it should generate max 2 unique samples and break
    assert len(samples) <= 2

def test_create_combined_analysis_plot():
    # Basic mock result structure to run without throwing errors
    finite_width_params = {"N": 8, "tau_p": 0.024, "integration_method": "trapezoid"}
    results_dict = {
        8: {
            "t_points": np.array([1.0, 2.0]),
            "finite_width_params": finite_width_params,
            "optimized_args": [{"A": [1.0], "alpha": [1.0]}, {}, {}, {}],
            "C(t)_true": np.array([0.9, 0.7])
        }
    }
    
    # We save plot to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        create_combined_analysis_plot(
            results_dict,
            n_values=[8],
            combined_loss=False,
            save_dir=tmp_dir,
            experimental_data=True,
            delta_approx=True,
            include_inversion=False
        )
        # Check that plot filename is saved
        expected_file = os.path.join(tmp_dir, "combined_analysis_delta_approx.png")
        assert os.path.exists(expected_file)
