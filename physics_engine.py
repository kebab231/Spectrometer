import numpy as np
from scipy.ndimage import gaussian_filter1d

H_PLANCK = 6.62607015e-34  
C_LIGHT = 299792458        
K_BOLTZ = 1.380649e-23     
Q_ELEC = 1.60217663e-19    
NUM_PX = 2048

def generate_static_defect_maps():
    """Generates repeatable spatial pixel defects for detector uniformity mapping."""
    np.random.seed(101)  
    prnu = np.random.normal(1.0, 0.008, size=NUM_PX)
    dsnu = np.random.lognormal(mean=0.0, sigma=0.15, size=NUM_PX)
    return prnu, dsnu

PRNU_MAP, DSNU_MAP = generate_static_defect_maps()

def get_silicon_qe(wl_nm):
    """Models a back-thinned Silicon CCD Quantum Efficiency response down to 150 nm."""
    qe = 0.94 * np.exp(-0.5 * ((wl_nm - 600) / 190)**2)
    nir = 1.0 / (1.0 + np.exp((wl_nm - 1020) / 28))
    uv = 1.0 / (1.0 + np.exp(-(wl_nm - 280) / 35))
    base_qe = qe * nir * uv
    uv_angle = np.clip((wl_nm - 150) / 150, 0.0, 1.0) * (np.pi / 2)
    uv_enhancement = np.where(wl_nm < 300, 0.15 + 0.3 * np.sin(uv_angle), 0.0)
    return np.clip(np.where(wl_nm < 300, np.maximum(base_qe, uv_enhancement), base_qe), 0.0, 0.95)

def simulate_instrument(fl_mm, lines_mm, slit_um, wl_min_target, wl_max_target,
                        temp_k, t_int_ms, sensor_c, vacuum_env, read_noise_e, 
                        adc_bits, full_well_e):
    focal_length = fl_mm * 1e-3
    grating_d = 1e-3 / lines_mm
    slit_w = slit_um * 1e-6
    etendue = (slit_w * 1e-3) * (np.pi * (0.025 / 2)**2) / (focal_length**2)
    px_pitch = 14e-6  
    focal_x = (np.arange(NUM_PX) - NUM_PX // 2) * px_pitch
    lam_min, lam_max = wl_min_target * 1e-9, wl_max_target * 1e-9
    
    sin_alpha_p_b_min = lam_min / grating_d
    sin_alpha_p_b_max = lam_max / grating_d
    if np.any(np.abs([sin_alpha_p_b_min, sin_alpha_p_b_max]) >= 1.0):
        return None
        
    beta_min_offset = np.arctan(focal_x[0] / focal_length)
    beta_max_offset = np.arctan(focal_x[-1] / focal_length)
    beta_tilt_rad = 0.5 * (np.arcsin(sin_alpha_p_b_max) + np.arcsin(sin_alpha_p_b_min))
    alpha_rad = 0.5 * (np.arcsin(sin_alpha_p_b_max) - np.arcsin(sin_alpha_p_b_min)) - 0.5 * (beta_min_offset + beta_max_offset)
    
    beta_rad = beta_tilt_rad + np.arctan(focal_x / focal_length)
    wl_m = grating_d * (np.sin(alpha_rad) + np.sin(beta_rad))
    if np.any(wl_m <= 0):
        return None
    wl_nm = wl_m * 1e9
    
    m1 = 2 * H_PLANCK * C_LIGHT**2
    with np.errstate(over='ignore', divide='ignore'):
        m2 = (H_PLANCK * C_LIGHT) / (wl_m * K_BOLTZ * temp_k)
        radiance = np.nan_to_num(m1 / (wl_m**5 * (np.exp(m2) - 1.0)))
        
    ideal_spec = radiance.copy()
    for center, fwhm, depth in [(280.0, 4.0, 0.75), (450.0, 3.0, 0.6), (600.0, 8.0, 0.7), (850.0, 15.0, 0.5)]:
        ideal_spec *= (1.0 - depth * np.exp(-0.5 * ((wl_nm - center) / (fwhm / 2.355))**2))
    if not vacuum_env:
        ideal_spec *= (1.0 / (1.0 + np.exp(-(wl_nm - 210.0) / 4.5)))

    mag_slit = slit_w * (np.cos(alpha_rad) / np.cos(beta_rad))
    lsf_width_m = np.sqrt(mag_slit**2 + (wl_m * (focal_length / 0.025))**2)
    sigma_pixels = np.mean(lsf_width_m / 2.355) / np.mean(np.diff(focal_x))
    convolved = gaussian_filter1d(ideal_spec, sigma=sigma_pixels, mode='nearest')

    rld = (grating_d * np.cos(beta_rad)) / focal_length
    delta_wl = rld * px_pitch
    photons = (convolved * etendue * delta_wl * (t_int_ms * 1e-3)) / ((H_PLANCK * C_LIGHT) / wl_m)
    qe = get_silicon_qe(wl_nm)
    
    electrons_prnu = (photons * qe) * PRNU_MAP
    tk = sensor_c + 273.15
    dark_rate_base = 0.5 * (tk / 298.15)**3 * np.exp(-(1.12 * Q_ELEC) / (2 * K_BOLTZ) * (1.0 / tk - 1.0 / 298.15))
    dark_e = (dark_rate_base * DSNU_MAP) * (t_int_ms * 1e-3)
    
    total_mean_signal = np.maximum(0.0, electrons_prnu + dark_e)
    noisy_e = np.zeros_like(total_mean_signal)
    large_mask = total_mean_signal > 1000
    small_mask = ~large_mask
    
    if np.any(small_mask):
        noisy_e[small_mask] = np.random.poisson(total_mean_signal[small_mask])
    if np.any(large_mask):
        noisy_e[large_mask] = np.random.normal(total_mean_signal[large_mask], np.sqrt(total_mean_signal[large_mask]))
        
    signal_with_read_noise = noisy_e + np.random.normal(0.0, read_noise_e, size=NUM_PX)
    clipped_electrons = np.clip(signal_with_read_noise, 0.0, full_well_e)
    
    gain_e_per_dn = full_well_e / ((2**adc_bits) - 1)
    telemetry_e_equivalent = np.round(clipped_electrons / gain_e_per_dn) * gain_e_per_dn
    
    return {
        'wl': wl_nm, 'id': photons * qe, 'el': electrons_prnu, 'dn': telemetry_e_equivalent, 
        'qe': qe, 'dk': dark_e, 'ns': np.sqrt(total_mean_signal + read_noise_e**2),
        'alpha': float(np.degrees(alpha_rad)), 'beta_tilt': float(np.degrees(beta_tilt_rad)),
        'res_avg': float(np.mean(delta_wl * 1e9))
    }
