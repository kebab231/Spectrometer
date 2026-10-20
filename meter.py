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
    import numpy as np
    
    # 1. Geometry Setup (Fixed-Deviation Czerny-Turner representation)
    NUM_PX = 2048
    grating_d = (1e-3 / lines_mm) if lines_mm > 0 else 1e-6
    focal_length = fl_mm * 1e-3
    
    # Anchor the center wavelength step
    wl_center = 0.5 * (wl_min_target + wl_max_target) * 1e-9
    theta_rad = np.radians(12.0)
    sin_phi = wl_center / (2.0 * grating_d * np.cos(theta_rad))
    phi_rad = np.arcsin(np.clip(sin_phi, -0.99, 0.99))
    alpha_rad = phi_rad + theta_rad
    beta_zero = phi_rad - theta_rad
    
    # Generate linear physical wavelength array across the pixels
    wl_nm = np.linspace(wl_min_target, wl_max_target, NUM_PX)
    lam = wl_nm * 1e-9
    
    # Calculate dispersion metrics
    delta_wl = (wl_max_target - wl_min_target) / NUM_PX
    
    # 2. Spectral Engine - Ideal Blackbody Planck Law
    h = 6.62607015e-34
    c = 299792458
    k_b = 1.380649e-23
    
    # Prevent divide by zero or extreme exponential overflows
    temp_k = max(300.0, float(temp_k))
    exponent = (h * c) / (lam * k_b * temp_k)
    exponent = np.clip(exponent, 1e-5, 700.0)
    bb_radiance = (2.0 * h * (c**2)) / ((lam**5) * (np.exp(exponent) - 1.0))
    
    # Inject atmospheric or sample absorption valleys (gives your graph shape)
    absorption = 1.0 - 0.45 * np.exp(-((wl_nm - 400)/15)**2) - 0.6 * np.exp(-((wl_nm - 760)/25)**2)
    ideal_photon_envelope = bb_radiance * absorption
    
    # 3. Slit Width Convolution (Smears signal out if slit is wide)
    pixel_scale_nm = delta_wl
    slit_width_nm = slit_um * (pixel_scale_nm / 10.0)
    sigma_px = max(0.5, slit_width_nm / pixel_scale_nm)
    
    # Generate Gaussian smoothing kernel
    kernel_radius = int(ceil(3 * sigma_px)) if 'ceil' in globals() else int(3 * sigma_px) + 1
    k_x = np.arange(-kernel_radius, kernel_radius + 1)
    kernel = np.exp(-(k_x**2) / (2 * sigma_px**2))
    kernel /= np.sum(kernel)
    
    # Apply convolution
    convolved_signal = np.convolve(ideal_photon_envelope, kernel, mode='same')
    
    # 4. Silicon Detector Quantum Efficiency Profile
    qe = 0.85 * np.exp(-((wl_nm - 650)/300)**2)
    qe = np.clip(qe, 0.02, 0.90)
    
    # 5. Thermodynamics and Photo-Electron Conversion
    # Scale photons to reasonable dynamic range below saturation thresholds
    base_scalar = 1e16 * (t_int_ms * 1e-3)
    ideal_electrons = convolved_signal * base_scalar * qe
    
    # Arrhenius Dark Current Noise implementation
    tk = sensor_c + 273.15
    E_g = 1.12 * 1.60217663e-19
    dark_rate = 1.5 * ((tk / 298.15)**3) * np.exp(-E_g / (2.0 * k_b * tk))
    dark_e = np.ones(NUM_PX) * dark_rate * 5000.0 * (t_int_ms * 1e-3)
    
    # Total accumulated charge matrix
    total_mean_signal = np.maximum(0.0, ideal_electrons + dark_e)
    
    # 6. Hardware Multi-bit ADC Quantization Dynamic Bounds
    full_well = 100000.0
    
    # Scale signal down safely to avoid flatlining, then apply Poisson noise distributions
    scale_to_well = np.max(total_mean_signal) if np.max(total_mean_signal) > 0 else 1.0
    electrons_in_well = (total_mean_signal / scale_to_well) * (full_well * 0.75)
    
    # Shot noise loop
    noisy_electrons = np.random.poisson(np.clip(electrons_in_well, 0, full_well)).astype(float)
    read_noise = np.random.normal(0.0, 4.5, size=NUM_PX)
    final_clipped_electrons = np.clip(noisy_electrons + read_noise, 0.0, full_well)
    
    # Map to true 14-bit data integers (0 to 16383)
    telemetry_dn = np.round((final_clipped_electrons / full_well) * 16383.0)
    
    return {
        'wl': wl_nm,
        'id': ideal_electrons,
        'el': final_clipped_electrons,
        'dn': telemetry_dn,
        'qe': qe,
        'dk': dark_e,
        'alpha': float(np.degrees(alpha_rad)),
        'beta_tilt': float(np.degrees(beta_zero)),
        'res_avg': float(pixel_scale_nm),
        'full_well': full_well
    }

# --- 3. STREAMLIT CONTROL PANEL ---
st.title("JHU/APL ASPIRE Spectrometer Instrument Model")
st.markdown("---")

st.sidebar.header("Instrument Controls")

def synced_slider_input(label, min_val, max_val, default_val, step, key_prefix):
    st.sidebar.markdown(f"**{label}**")
    with st.sidebar:
        col1, col2 = st.columns(2)
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
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.0, 9.5))
            fig.subplots_adjust(left=0.12, right=0.86, bottom=0.08, top=0.94, hspace=0.45)
            
            # --- PLOT 1: OPTICAL SPECTRUM (Wavelength Scale) ---
            y_scale = 1e11
            ax1.plot(data['wl'], data['id'] / y_scale, 'gray', linestyle='--', alpha=0.6, label='Ideal Photon Envelope')
            ax1.plot(data['wl'], data['el'] / y_scale, '#1a5f7a', linewidth=1.3, label='Optical Convolved Signal')
            
            ax1.set_xlabel(r'Focal Plane Target Wavelength ($\lambda$, nm)', weight='bold')
            ax1.set_ylabel(r'Amplitude ($\times 10^{11}\ e^-$)', color='#1a5f7a', weight='bold')
            ax1.set_xlim(wl_min, wl_max)
            ax1.set_ylim(-0.05 * (data['full_well'] / y_scale), 1.05 * (data['full_well'] / y_scale))
            ax1.set_title('JHU/APL ASPIRE Spectrometer Response Model', weight='bold', size=11, pad=10)
            ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=8)
            
            meta_text = (
                f"Calculated Geometry:\n"
                f"  Incidence α: {data['alpha']:.2f}°\n"
                f"  Grating Tilt β: {data['beta_tilt']:.2f}°\n"
                f"Performance:\n"
                f"  Mean Dispersion: {data['res_avg']:.3f} nm/px\n"
                f"  Sensor Array Temp: {tc:.1f}°C"
            )
            ax1.text(0.02, 0.05, meta_text, transform=ax1.transAxes, fontsize=8, family='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.95))
            
            # --- PLOT 2: QUANTIZED TELEMETRY DATA (Pixel Scale) ---
            pixel_indices = np.arange(NUM_PX)
            ax2.step(pixel_indices, data['dn'], '#9a0680', where='mid', linewidth=1.1, label='Quantized Telemetry Output')
            ax2.set_xlabel('Telemetry Array Channel (Pixel Bin Index)', weight='bold')
            ax2.set_ylabel('Telemetry Output (DN)', color='#9a0680', weight='bold')
            ax2.set_xlim(0, 2048)
            ax2.set_ylim(-500, 17000)
            ax2.legend(loc='upper right', frameon=True, fontsize=8)
            
            # --- PLOT 3: PHYSICAL HARDWARE PERFORMANCE (Pixel Scale) ---
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
