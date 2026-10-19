import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.ndimage import gaussian_filter1d

# --- 1. PAGE LAYOUT CONFIGURATION ---
st.set_page_config(
    page_title="JHU/APL ASPIRE Instrument Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Precise Metrological Constants
H_PLANCK, C_LIGHT, K_BOLTZ, Q_ELEC = 6.62607015e-34, 299792458, 1.380649e-23, 1.60217663e-19
NUM_PX = 2048

# Seed standard calibration tables for persistent instrument hardware defects
@st.cache_data
def generate_static_defect_maps():
    np.random.seed(42)
    prnu = np.random.normal(1.0, 0.012, size=NUM_PX)
    dsnu = np.random.lognormal(mean=0.0, sigma=0.25, size=NUM_PX)
    return prnu, dsnu

PRNU_MAP, DSNU_MAP = generate_static_defect_maps()

# --- 2. CORE PHYSICS & INSTRUMENT SIMULATION ENGINE ---
def get_silicon_qe(wl_nm):
    """Models a back-thinned Silicon CCD Quantum Efficiency response profile down to 150 nm."""
    qe = 0.94 * np.exp(-0.5 * ((wl_nm - 600) / 190)**2)
    nir = 1.0 / (1.0 + np.exp((wl_nm - 1020) / 28))
    uv = 1.0 / (1.0 + np.exp(-(wl_nm - 280) / 35))
    base_qe = qe * nir * uv
    
    # Enforce strict positive UV enhancement bounds using safe angular transforms
    uv_angle = np.clip((wl_nm - 150) / 150, 0.0, 1.0) * (np.pi / 2)
    uv_enhancement = np.where(wl_nm < 300, 0.15 + 0.3 * np.sin(uv_angle), 0.0)
    
    return np.clip(np.where(wl_nm < 300, np.maximum(base_qe, uv_enhancement), base_qe), 0.0, 0.95)

def simulate_instrument(fl_mm=100.0, lines_mm=300.0, slit_um=25.0, 
                        wl_min_target=150.0, wl_max_target=1000.0,
                        temp_k=4500.0, t_int_ms=50.0, sensor_c=25.0, vacuum_env=False):
    
    focal_length = fl_mm * 1e-3
    grating_d = 1e-3 / lines_mm
    slit_w = slit_um * 1e-6
    etendue = (slit_w * 1e-3) * (np.pi * (0.025 / 2)**2) / (focal_length**2)
    
    px_pitch = 14e-6  
    focal_x = (np.arange(NUM_PX) - NUM_PX // 2) * px_pitch
    
    # Validate optical geometric bounds via scalar inputs
    lam_min, lam_max = wl_min_target * 1e-9, wl_max_target * 1e-9
    sin_alpha_plus_beta_min = lam_min / grating_d
    sin_alpha_plus_beta_max = lam_max / grating_d
    
    if np.any(np.abs([sin_alpha_plus_beta_min, sin_alpha_plus_beta_max]) >= 1.0):
        return None
        
    beta_min_offset = np.arctan(focal_x / focal_length)
    beta_max_offset = np.arctan(focal_x[-1] / focal_length)
    
    # Process core optical angles
    beta_tilt_rad = 0.5 * (np.arcsin(sin_alpha_plus_beta_max) + np.arcsin(sin_alpha_plus_beta_min))
    alpha_rad = 0.5 * (np.arcsin(sin_alpha_plus_beta_max) - np.arcsin(sin_alpha_plus_beta_min)) - 0.5 * (beta_min_offset + beta_max_offset)
    
    beta_rad = beta_tilt_rad + np.arctan(focal_x / focal_length)
    sin_beta = np.sin(beta_rad)
    sin_alpha = np.sin(alpha_rad)
    
    if np.any(np.abs(sin_alpha + sin_beta) > 1.0):
        return None
        
    wl_m = grating_d * (sin_alpha + sin_beta)
    if np.any(wl_m <= 0):
        return None
    wl_nm = wl_m * 1e9
    
    # Optimized Planck Radiance Engine
    m1 = 2 * H_PLANCK * C_LIGHT**2
    with np.errstate(over='ignore', divide='ignore'):
        m2 = (H_PLANCK * C_LIGHT) / (wl_m * K_BOLTZ * temp_k)
        radiance = np.nan_to_num(m1 / (wl_m**5 * (np.exp(m2) - 1.0)))
        
    ideal_spec = radiance.copy()
    absorption_lines = [(280.0, 4.0, 0.75), (450.0, 3.0, 0.6), (600.0, 8.0, 0.7), (850.0, 15.0, 0.5)]
    for center, fwhm, depth in absorption_lines:
        ideal_spec *= (1.0 - depth * np.exp(-0.5 * ((wl_nm - center) / (fwhm / 2.355))**2))

    if not vacuum_env:
        air_transmission = 1.0 / (1.0 + np.exp(-(wl_nm - 210.0) / 4.5))
        ideal_spec *= air_transmission

    # Optimized Wave-Mechanical 1D Gaussian Convolution Matrix
    mag_slit = slit_w * (np.cos(alpha_rad) / np.cos(beta_rad))
    lsf_width_m = np.sqrt(mag_slit**2 + (wl_m * (focal_length / 0.025))**2)
    sigma_m = lsf_width_m / 2.355
    
    mean_dispersion = np.mean(np.diff(focal_x))
    sigma_pixels = np.mean(sigma_m) / mean_dispersion
    convolved = gaussian_filter1d(ideal_spec, sigma=sigma_pixels, mode='nearest')

    # Metrological Electron Transport Mapping
    rld = (grating_d * np.cos(beta_rad)) / focal_length
    delta_wl = rld * px_pitch
    power = convolved * etendue * delta_wl
    photons = (power * (t_int_ms * 1e-3)) / ((H_PLANCK * C_LIGHT) / wl_m)
    qe = get_silicon_qe(wl_nm)
    
    ideal_electrons = photons * qe
    electrons = ideal_electrons * PRNU_MAP
    
    # Thermal Dark Current Defects
    tk = sensor_c + 273.15
    dark_rate_base = 0.5 * (tk / 298.15)**3 * np.exp(-(1.12 * Q_ELEC) / (2 * K_BOLTZ) * (1.0 / tk - 1.0 / 298.15))
    dark_e = (dark_rate_base * DSNU_MAP) * (t_int_ms * 1e-3)
    
    # Fast vectorized Poisson-Gaussian statistical approximation channel
    lam = np.maximum(0.0, electrons + dark_e)
    noisy_e = np.zeros_like(lam)
    
    large_thresh = 1000
    large_mask = lam > large_thresh
    small_mask = ~large_mask
    
    if np.any(small_mask):
        noisy_e[small_mask] = np.random.poisson(lam[small_mask])
    if np.any(large_mask):
        noisy_e[large_mask] = np.random.normal(lam[large_mask], np.sqrt(lam[large_mask]))
        
    total_e = np.clip(noisy_e + np.random.normal(0.0, 3.5, size=NUM_PX), 0.0, 50000.0)
    gain = 16383.0 / 50000.0
    
    return {
        'wl': wl_nm, 'id': ideal_electrons, 'el': electrons, 'dn': np.round(total_e * gain) / gain, 
        'qe': qe, 'dk': dark_e, 
        'alpha': float(np.degrees(alpha_rad)), 
        'beta_tilt': float(np.degrees(beta_tilt_rad)),
        'res_avg': float(np.mean(delta_wl * 1e9))
    }

# --- 3. STREAMLIT CONTROL PANEL SIDEBAR ---
st.title("JHU/APL ASPIRE Dispersive Electro-Optical Telemetry Instrument Response Model")
st.markdown("---")

st.sidebar.header("🎛️ Instrument Controls")

fl = st.sidebar.slider("Focal Length (mm)", 50.0, 250.0, 100.0, step=10.0)
l_mm = st.sidebar.slider("Grating Pitch (lines/mm)", 150.0, 1000.0, 300.0, step=50.0)
slit = st.sidebar.slider("Slit Width (µm)", 5.0, 100.0, 25.0, step=5.0)

st.sidebar.subheader("Spectral Targets")
wl_min = st.sidebar.slider("Target Min λ (nm)", 100.0, 400.0, 150.0, step=10.0)
wl_max = st.sidebar.slider("Target Max λ (nm)", 700.0, 1200.0, 1000.0, step=25.0)

st.sidebar.subheader("Environment Settings")
tk = st.sidebar.slider("Source Temperature (K)", 2000.0, 6000.0, 4500.0, step=100.0)
t_int = st.sidebar.slider("Integration Time (ms)", 1.0, 500.0, 50.0, step=5.0)
tc = st.sidebar.slider("Sensor Temperature (°C)", -40.0, 60.0, 25.0, step=5.0)
vacuum = st.sidebar.toggle("Vacuum Environment Mode", value=False)

# Guard against user moving sliders into a logical collision
if wl_min >= wl_max:
    st.error("Invalid Configuration: Target Minimum λ cannot be greater than or equal to Target Maximum λ.")
else:
    # Run simulation based on active slider positions
    data = simulate_instrument(fl, l_mm, slit, wl_min, wl_max, tk, t_int, tc, vacuum)

    if data is None:
        st.error("Error: Target wavelength boundaries exceed physical grating limits for this line pitch.")
    else:
        # --- 4. GRAPHICAL CANVAS RENDERING LAYER ---
        with plt.rc_context({'axes.grid': True, 'grid.linestyle': ':', 'font.size': 8.5}):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.8), sharex=True,
                                           gridspec_kw={'height_ratios': [2.0, 1]})
            
            fig.subplots_adjust(left=0.12, right=0.78, bottom=0.14, top=0.88, hspace=0.18)
            
            # Primary Signals Subplot
            ax1.plot(data['wl'], data['id'], 'gray', linestyle='--', alpha=0.5, label='Ideal Photon Envelope')
            ax1.plot(data['wl'], data['el'], '#1a5f7a', linewidth=1.1, alpha=0.7, label=r'Optical Convolved Signal ($e^-$)')
            ax1.step(data['wl'], data['dn'], '#9a0680', where='mid', linewidth=0.4, alpha=0.8, label='Quantized Telemetry Output')
            ax1.set_ylabel('Focal Plane Amplitude (e- / Pixel)', weight='bold')
            
            ax1.set_title('JHU/APL ASPIRE Dispersive Electro-Optical Telemetry Instrument Response Model', weight='bold', size=10, pad=10)
            ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=7.5)
            
            # Hardware Metrics Subplot
            ax2_t = ax2.twinx()
            p1, = ax2.plot(data['wl'], data['qe'], 'g', linewidth=1.1, label='Silicon CCD QE Profile')
            p2, = ax2_t.plot(data['wl'], data['dk'], '#d35400', linewidth=0.6, alpha=0.7, label='Fixed-Pattern Dark DSNU')
            ax2.set_xlabel(r'Focal Plane Array Dispersion Grid Alignment ($\lambda$, nm)', weight='bold')
            ax2.set_ylabel('Quantum Efficiency [0-1]', color='g')
            ax2_t.set_ylabel(r'Dark Noise Current ($e^-$)', color='#d35400')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2_t.tick_params(axis='y', labelcolor='#d35400')
            
            # FIXED: Evaluates the scalar index data['wl'][0] safely to avoid array truncation exceptions
            ax2.set_xlim(data['wl'][0], data['wl'][-1])
            ax2.set_ylim(-0.02, 1.05)
            ax2.legend([p1, p2], [p1.get_label(), p2.get_label()], loc='upper left', frameon=True, fontsize=7.5)
            
            # Compact physical telemetry string box layout
            meta_text = (
                f"Calculated Alignment:\n"
                f"  Angle α: {data['alpha']:.2f}°\n"
                f"  Tilt β: {data['beta_tilt']:.2f}°\n"
                f"Performance Metrics:\n"
                f"  Mean Res: {data['res_avg']:.3f} nm/px\n"
                f"  Sensor Temp: {tc}°C"
            )
            ax1.text(0.02, 0.05, meta_text, transform=ax1.transAxes, fontsize=7.5, family='monospace',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f6f8', edgecolor='#bdc3c7', alpha=0.9))
            
# Safely project onto Web framework and prevent background memory leakage
st.pyplot(fig)
plt.close(fig)
