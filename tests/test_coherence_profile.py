import numpy as np

from fwdd.coherence_profile import (
    coherence_decay_profile_delta,
    coherence_decay_profile_finite_peaks_with_widths,
    noise_inversion_delta,
)


def test_coherence_decay_profile_delta():
    t = 2.0
    noise_profile = 0.5
    # chi_t = t * noise_profile / pi = 2.0 * 0.5 / pi = 1.0 / pi
    # out = exp(-1.0 / pi)
    out = coherence_decay_profile_delta(t, noise_profile)
    assert np.allclose(out, np.exp(-1.0 / np.pi))

def test_noise_inversion_delta():
    t = 2.0
    C_t = np.exp(-1.0 / np.pi)
    out = noise_inversion_delta(t, C_t)
    assert np.allclose(out, 0.5)

def test_coherence_decay_profile_finite_peaks_with_widths():
    # Simple test for integration execution
    def noise_profile(omega):
        return 0.1 / (omega**1.0)

    # Let's run a simple point integration
    C_t, _, _, _ = coherence_decay_profile_finite_peaks_with_widths(
        t=1.0,
        N=8,
        tau_p=0.024,
        method="trapezoid",
        noise_profile=noise_profile,
        omega_resolution=1000,
        omega_range=(1e-3, 1e2),
        num_peaks_cutoff=5,
        peak_resolution=10
    )
    assert 0.0 <= C_t <= 1.0
