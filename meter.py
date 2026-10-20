import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.ndimage import gaussian_filter1d

# --- 1. APP CONFIGURATION & CONSTANTS ---
st.set_page_config(
    page_title="JHU/APL ASPIRE Spectrometer Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

H_PLANCK = 6.62607015e-34  # J*s
C_LIGHT = 299792458        # m/s
K_BOLTZ = 1.380649e-23     # J/K
Q_ELEC = 1.60217663e-19    # C
NUM_PX = 2048

@st.cache_data
def generate_defect_maps():
    """Generates repeatable sensor pixel variation maps."""
    np.random.seed(42)
    prnu = np.random.normal(1.0, 0.012, size=NUM_PX)
    dsnu = np.random.lognormal(mean=0.0, sigma=0.25, size=NUM_PX) * 5.0
    return prnu, dsnu

PRNU_MAP, DSNU_MAP = generate_defect_maps()


# --- 2. CORE PHYSICS & INSTRUMENT SIMULATION ENGINE ---
def get_silicon_qe(wl_nm):
    """Calculates CCD Quantum Efficiency based on a standard back-thinned silicon profile."""
    qe = 0.94 * np.exp(-0.5 * ((wl_nm - 600) / 190)**2)
    nir = 1.0 / (1.0 + np.exp((wl_nm - 1020) / 28))
    uv = 1.0 / (1.0 + np.exp(-(wl_nm - 280) / 35))
    base_qe = qe * nir * uv
    
    uv_angle = np.clip((wl_nm - 150) / 150, 0.0, 1.0) * (np.pi / 2)
    uv_enhancement = np.where(wl_nm < 300, 0.15 + 0.3 * np.sin(uv_angle), 0.0)
    
    return np.clip(np.where(wl_nm < 300, np.maximum(base_qe, uv_enhancement), base_qe), 0.0, 0.95)


def simulate_instrument(fl_mm, lines_mm, slit_um, wl_min_target, wl_max_target, temp_k, t_int_ms, sensor_c, vacuum_env):
    """Simulates grating diffraction optics, source blackbody radiance, and sensor noise."""
    focal_length = fl_mm * 1e-3
    grating_d = 1e-3 / lines_mm
    slit_w = slit_um * 1e-6
    etendue = (slit_w * 1e-3) * (np.pi * (0.025 / 2)**2) / (focal_length**2)
    
    focal_x = (np.arange(NUM_PX) - NUM_PX // 2) * 14e-6
    
    lam_min, lam_max = wl_min_target * 1e-9, wl_max_target * 1e-9
    sin_alpha_plus_beta_min = lam_min / grating_d
    sin_alpha_plus_beta_max = lam_max / grating_d
    
    if np.any(np.abs([sin_alpha_plus_beta_min, sin_alpha_plus_beta_max]) >= 1.0):
        return None
        
    beta_min_offset = np.arctan(focal_x[0] / focal_length)
    beta_max_offset = np.arctan(focal_x[-1] / focal_length)
    
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
    
    m1 = 2 * H_PLANCK * C_LIGHT**2
    with np.errstate(over='ignore', divide='ignore'):
        m2 = (H_PLANCK * C_LIGHT) / (wl_m * K_BOLTZ * temp_k)
        radiance = np.nan_to_num(m1 / (wl_m**5 * (np.exp(m2) - 1.0)))
        
    ideal_spec = radiance.copy()
    
    absorption_lines = [(280.0, 4.0, 0.75), (450.0, 3.0, 0.6), (600.0, 8.0, 0.7), (850.0, 15.0, 0.5)]
    for center, fwhm, depth in absorption_lines:
        ideal_spec *= (1.0 - depth * np.exp(-0.5 * ((wl_nm - center) / (fwhm / 2.355))**2))

    if not vacuum_env:
        ideal_spec *= 1.0 / (1.0 + np.exp(-(wl_nm - 210.0) / 4.5))

    mag_slit = slit_w * (np.cos(alpha_rad) / np.cos(beta_rad))
    lsf_width_m = np.sqrt(mag_slit**2 + (wl_m * (focal_length / 0.025))**2)
    sigma_pixels = np.mean(lsf_width_m / 2.355) / 14e-6
    convolved = gaussian_filter1d(ideal_spec, sigma=sigma_pixels, mode='nearest')

    delta_wl = ((grating_d * np.cos(beta_rad)) / focal_length) * 14e-6
    photons = (convolved * etendue * delta_wl * (t_int_ms * 1e-3)) / ((H_PLANCK * C_LIGHT) / wl_m)
    qe = get_silicon_qe(wl_nm)
    
    ideal_electrons = photons * qe
    electrons = ideal_electrons * PRNU_MAP
    
    tk = sensor_c + 273.15
    dark_rate_base = 0.5 * (tk / 298.15)**3 * np.exp(-(1.12 * Q_ELEC) / (2 * K_BOLTZ) * (1.0 / tk - 1.0 / 298.15))
    dark_e = (dark_rate_base * DSNU_MAP) * (t_int_ms * 1e-3)
    
    lam = np.maximum(0.0, electrons + dark_e)
    noisy_e = np.zeros_like(lam)
    
    large_mask = lam > 1000
    small_mask = ~large_mask
    if np.any(small_mask): noisy_e[small_mask] = np.random.poisson(lam[small_mask])
    if np.any(large_mask):  noisy_e[large_mask] = np.random.normal(lam[large_mask], np.sqrt(lam[large_mask]))
        
    full_well = (np.max(lam) if np.max(lam) > 0 else 50000.0) * 1.1
    total_e = np.clip(noisy_e + np.random.normal(0.0, 3.5, size=NUM_PX), 0.0, full_well)
    
    telemetry_dn = np.round((total_e / full_well) * 16383.0)

    return {
        'wl': wl_nm, 
        'id': ideal_electrons, 
        'el': total_e,  
        'dn': telemetry_dn, 
        'qe': qe, 
        'dk': dark_e, 
        'alpha': float(np.degrees(alpha_rad)), 
        'beta_tilt': float(np.degrees(beta_tilt_rad)),
        'res_avg': float(np.mean(delta_wl * 1e9)), 
        'full_well': full_well
    }


# --- 3. STREAMLIT CONTROL PANEL ---
st.title("JHU/APL ASPIRE Spectrometer Instrument Model")
st.markdown("---")

st.sidebar.header("Instrument Controls")

def synced_slider_input(label, min_val, max_val, default_val, step, key_prefix):
    st.sidebar.markdown(f"**{label}**")
    with st.sidebar:
        col1, col2 = st.columns([1, 2])
        val_num = col1.number_input("", min_value=float(min_val), max_value=float(max_val), value=float(default_val), step=float(step), label_visibility="collapsed", key=f"{key_prefix}_num")
        val_slider = col2.slider("", min_value=float(min_val), max_value=float(max_val), value=float(val_num), step=float(step), label_visibility="collapsed", key=f"{key_prefix}_slide")
    if val_num != val_slider:
        val_num = val_slider if val_num == st.session_state.get(f"{key_prefix}_last_num", default_val) else val_num
        st.session_state[f"{key_prefix}_last_num"] = val_num
        st.rerun()
    return val_num

fl = synced_slider_input("Focal Length (mm)", 50.0, 250.0, 100.0, 10.0, "fl")
l_mm = synced_slider_input("Grating Pitch (lines/mm)", 150.0, 1000.0, 300.0, 50.0, "l_mm")
slit = synced_slider_input("Slit Width (µm)", 5.0, 100.0, 25.0, 5.0, "slit")

st.sidebar.subheader("Spectral Spectrum Targets")
wl_min = synced_slider_input("Minimum Target Wavelength (nm)", 100.0, 400.0, 150.0, 10.0, "wl_min")
wl_max = synced_slider_input("Maximum Target Wavelength (nm)", 700.0, 1200.0, 1000.0, 25.0, "wl_max")

st.sidebar.subheader("Environment Settings")
tk = synced_slider_input("Source Temperature (K)", 2000.0, 6000.0, 4500.0, 100.0, "tk")
t_int = synced_slider_input("Integration Time (ms)", 1.0, 500.0, 50.0, 5.0, "t_int")
tc = synced_slider_input("Sensor Temperature (°C)", -40.0, 60.0, 25.0, 5.0, "tc")
vacuum = st.sidebar.toggle("Vacuum Environment Mode", value=False)

if wl_min >= wl_max:
    st.error("Invalid Configuration: Target Minimum wavelength cannot be greater than or equal to Target Maximum wavelength.")
else:
    data = simulate_instrument(fl, l_mm, slit, wl_min, wl_max, tk, t_int, tc, vacuum)
    if data is None:
        st.error("Error: Target wavelength boundaries exceed physical grating limits for this line pitch.")
    else:
        # --- 4. MATPLOTLIB MULTI-GRAPH COMPONENT ---
        with plt.rc_context({'axes.grid': True, 'grid.linestyle': ':', 'font.size': 8.5}):
            # Setup 3 independent stacked subplots with individual x-axes
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.0, 9.5))
            fig.subplots_adjust(left=0.12, right=0.86, bottom=0.08, top=0.94, hspace=0.45)
            
            # --- PLOT 1: OPTICAL SPECTRUM (Wavelength Scale) ---
            y_scale = 1e11
            ax1.plot(data['wl'], data['id'] / y_scale, 'gray', linestyle='--', alpha=0.6, label='Ideal Photon Envelope')
            ax1.plot(data['wl'], data['el'] / y_scale, '#1a5f7a', linewidth=1.3, label='Optical Convolved Signal')
            
            ax1.set_xlabel(r'Focal Plane Target Wavelength ($\lambda$, nm)', weight='bold')
            ax1.set_ylabel(r'Amplitude ($\times 10^{11}\ e^-$)', color='#1a5f7a', weight='bold')
            ax1.set_xlim(data['wl'][0], data['wl'][-1])
            ax1.set_ylim(-0.05 * (data['full_well'] / y_scale), 1.05 * (data['full_well'] / y_scale))
            ax1.set_title('JHU/APL ASPIRE Spectrometer Response Model', weight='bold', size=11, pad=10)
            ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=8)
            
            # Geometry Meta Box
            meta_text = (
                f"Calculated Geometry:\n"
                f"  Incidence α: {data['alpha']:.2f}°\n"
                f"  Grating Tilt β: {data['beta_tilt']:.2f}°\n"
                f"Performance:\n"
                f"  Mean Dispersion: {data['res_avg']:.3f} nm/px\n"
                f"  Sensor Array Temp: {tc:.1f}°C"
            )
           ax2.set_xlabel('Telemetry Array Channel (Pixel Bin Index)', weight='bold')
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.95))
            
                      # --- PLOT 2: QUANTIZED TELEMETRY DATA (Pixel Scale) ---
            # Create pixel indices matching the size of the array (0 to 2047)
            pixel_indices = np.arange(NUM_PX)
            
            # Change the x-data here from data['wl'] to pixel_indices
            ax2.step(pixel_indices, data['dn'], '#9a0680', where='mid', linewidth=1.1, label='Quantized Telemetry Output')
            
            # This is your corrected label
            ax2.set_xlabel('Telemetry Array Channel (Pixel Bin Index)', weight='bold')
            ax2.set_ylabel('Telemetry Output (DN)', color='#9a0680', weight='bold')
            
            # Adjust the limits to clip from pixel 0 to the maximum pixel index
            ax2.set_xlim(0, NUM_PX - 1)
            ax2.set_ylim(-500, 17000)
            ax2.legend(loc='upper right', frameon=True, fontsize=8)
            
            # --- PLOT 3: PHYSICAL HARDWARE PERFORMANCE (Pixel Scale) ---
            pixel_indices = np.arange(NUM_PX)
p1, = ax3.plot(pixel_indices, data['qe'], 'g', linewidth=1.2, label='Silicon CCD QE Profile')
ax3.set_xlabel('Physical Sensor Array Element Index (Pixel Number)', weight='bold')
ax3.set_ylabel('Quantum Efficiency [0-1]', color='g', weight='bold')
ax3.tick_params(axis='y', labelcolor='g')
ax3.set_ylim(-0.05, 1.05)
ax3_twin = ax3.twinx()
p2, = ax3_twin.plot(pixel_indices, data['dk'], '#d35400', linewidth=0.5, alpha=0.8, label='Spatial Dark DSNU Floor')
ax3_twin.set_ylabel(r'Dark Noise ($e^-$ / pixel / sec)', color='#d35400', weight='bold')
ax3_twin.tick_params(axis='y', labelcolor='#d35400')
ax3.set_xlim(0, NUM_PX - 1)
ax3.legend([p1, p2], [p1.get_label(), p2.get_label()], loc='upper right', frameon=True, fontsize=8)
st.pyplot(fig)
plt.close(fig)
