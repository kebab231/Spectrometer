import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="JHU/APL ASPIRE Spectrometer Engine", layout="wide")

# Constants
K_BOLTZ = 1.380649e-23
Q_ELEC = 1.60217663e-19

# --- 2. CORE PHYSICS & INSTRUMENT SIMULATION ENGINE ---
def simulate_instrument(fl_mm, lines_mm, slit_um, wl_min_target, wl_max_target, temp_k, t_int_ms, sensor_c, vacuum_env):
    NUM_PX = 2048
    grating_d = (1e-3 / lines_mm) if lines_mm > 0 else 1e-6
    focal_length = fl_mm * 1e-3
    
    # Czerny-Turner Fixed-Deviation Configuration Layout
    wl_center = 0.5 * (wl_min_target + wl_max_target) * 1e-9
    theta_rad = np.radians(12.0) # Fixed half-inclusion angle
    
    sin_phi = wl_center / (2.0 * grating_d * np.cos(theta_rad))
    phi_rad = np.arcsin(np.clip(sin_phi, -0.99, 0.99))
    
    alpha_rad = phi_rad + theta_rad   # Fixed incidence angle
    beta_zero = phi_rad - theta_rad   # Nominal center reflection angle
    
    # Establish dynamic linear pixel array footprint
    wl_nm = np.linspace(wl_min_target, wl_max_target, NUM_PX)
    lam = wl_nm * 1e-9
    pixel_scale_nm = (wl_max_target - wl_min_target) / NUM_PX
    
    # Planck's Blackbody Radiance Engine
    h = 6.62607015e-34
    c = 299792458
    exponent = (h * c) / (lam * K_BOLTZ * float(temp_k))
    exponent = np.clip(exponent, 1e-5, 700.0)
    bb_radiance = (2.0 * h * (c**2)) / ((lam**5) * (np.exp(exponent) - 1.0))
    
    # Introduce optical absorption valleys to give the spectrum dynamic shapes
    absorption = 1.0 - 0.45 * np.exp(-((wl_nm - 400)/15)**2) - 0.6 * np.exp(-((wl_nm - 760)/25)**2)
    ideal_photon_envelope = bb_radiance * absorption
    
    # Slit Width Convolution (Line Spread Function Smearing)
    slit_width_nm = slit_um * (pixel_scale_nm / 10.0)
    sigma_px = max(0.5, slit_width_nm / pixel_scale_nm)
    k_radius = int(np.ceil(3 * sigma_px))
    k_x = np.arange(-k_radius, k_radius + 1)
    kernel = np.exp(-(k_x**2) / (2 * sigma_px**2))
    kernel /= np.sum(kernel)
    convolved_signal = np.convolve(ideal_photon_envelope, kernel, mode='same')
    
    # Sensor Quantum Efficiency Profile
    qe = 0.85 * np.exp(-((wl_nm - 650)/300)**2)
    qe = np.clip(qe, 0.02, 0.90)
    
    # Normalize photon scales to cleanly match your original plot multiplier scales (1e-11)
    ideal_electrons = (ideal_photon_envelope / np.max(ideal_photon_envelope)) * 0.8 if np.max(ideal_photon_envelope) > 0 else ideal_photon_envelope
    convolved_electrons = (convolved_signal / np.max(convolved_signal)) * 0.8 if np.max(convolved_signal) > 0 else convolved_signal

    
    # Arrhenius Thermodynamic Noise Properties
    tk = sensor_c + 273.15
    E_g = 1.12 * Q_ELEC
    dark_rate = 1.5 * ((tk / 298.15)**3) * np.exp(-E_g / (2.0 * K_BOLTZ * tk))
    dark_e = np.ones(NUM_PX) * dark_rate * 0.05 * (t_int_ms * 1e-3)
    
    # Total charge composition
    total_mean_signal = np.maximum(0.0, convolved_electrons * qe + dark_e)
    
    # 14-Bit Quantization Dynamics
    full_well = 1.0  # Synced exactly with your top chart's 1.0 Max limit
    read_noise = np.random.normal(0.0, 0.005, size=NUM_PX)
    noisy_electrons = np.random.poisson(np.clip(total_mean_signal * 1000, 0, None)) / 1000.0
    final_electrons = np.clip(noisy_electrons + read_noise, 0.0, full_well)
    
    # Convert into precise 14-bit Telemetry numbers
    telemetry_dn = np.round((final_electrons / full_well) * 16383.0)
    
    return {
        'wl': wl_nm,
        'id': ideal_electrons,
        'el': final_electrons,
        'dn': telemetry_dn,
        'qe': qe,
        'dk': dark_e,
        'alpha': float(np.degrees(alpha_rad)),
        'beta_tilt': float(np.degrees(beta_zero)),
        'res_avg': float(pixel_scale_nm),
        'full_well': full_well
    }

# --- 3. STREAMLIT CONTROL PANEL SIDEBAR ---
st.sidebar.title("Instrument Controls")
fl = st.sidebar.slider("Focal Length (mm)", 50.0, 500.0, 100.0, 10.0)
l_mm = st.sidebar.slider("Grating Pitch (lines/mm)", 100.0, 1200.0, 300.0, 50.0)
slit = st.sidebar.slider("Slit Width (μm)", 5.0, 200.0, 25.0, 5.0)

st.sidebar.title("Spectral Spectrum Targets")
wl_min = st.sidebar.slider("Minimum Target Wavelength (nm)", 150.0, 500.0, 150.0, 10.0)
wl_max = st.sidebar.slider("Maximum Target Wavelength (nm)", 600.0, 1100.0, 1000.0, 10.0)

st.sidebar.title("Environment Settings")
tk = st.sidebar.slider("Source Temperature (K)", 1000.0, 6000.0, 4500.0, 100.0)
t_int = st.sidebar.slider("Integration Time (ms)", 1.0, 500.0, 50.0, 5.0)
tc = st.sidebar.slider("Sensor Temperature (°C)", -40.0, 60.0, 25.0, 5.0)
vacuum = st.sidebar.checkbox("Vacuum Environment Mode", value=False)

# Execute updated physics pipeline
data = simulate_instrument(fl, l_mm, slit, wl_min, wl_max, tk, t_int, tc, vacuum)

# --- 4. DATA VISUALIZATION DASHBOARD ---
st.title("JHU/APL ASPIRE Spectrometer Response Model")

#new stuff--------------------------
import numpy as np

# Assuming 'telemetry_output' is your current 1D NumPy array representing the smooth curve
# Convert your signal to a temporary photon/electron count if it isn't already, 
# because Poisson noise applies strictly to discrete counts.
signal_counts = telemetry_dn 

# Generate Poisson (Shot) Noise
# np.random.poisson requires inputs >= 0; clip negative values if any exist from calculations
shot_noise_signal = np.random.poisson(np.maximum(0, signal_counts)).astype(float)



if data is not None:
    # Chart 1: Optical Model Curves
    fig1, ax1 = plt.subplots(figsize=(10, 3.5))
    ax1.plot(data['wl'], data['id'], '--', color='gray', alpha=0.7, label='Ideal Photon Envelope')
    ax1.plot(data['wl'], data['el'], color='steelblue', label='Optical Convolved Signal')
    ax1.set_xlabel("Focal Plane Target Wavelength ($\lambda$, nm)", fontweight='bold')
    ax1.set_ylabel("Amplitude (x$1^{11}$ $e^-$)", color='steelblue', fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')
    ax1.set_xlim(wl_min, wl_max)
    ax1.set_ylim(-0.05, 1.05)
    
    # Physical Dashboard Statistics overlay box
    textstr = (
        f"Calculated Geometry:\n"
        f"  Incidence $\\alpha$: {data['alpha']:.2f}°\n"
        f"  Grating Tilt $\\beta$: {data['beta_tilt']:.2f}°\n"
        f"Performance:\n"
        f"  Mean Dispersion: {data['res_avg']:.3f} nm/px\n"
        f"  Sensor Array Temp: {tc:.1f}°C"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray')
    ax1.text(0.03, 0.08, textstr, transform=ax1.transAxes, fontsize=9, verticalalignment='bottom', bbox=props)
    st.pyplot(fig1)
    
       # Chart 2: Digitized Telemetry Curves
    fig2, ax2 = plt.subplots(figsize=(10, 2.5))
    ax2.plot(np.arange(2048), data['dn'], color='purple', alpha=0.8, label='Quantized Telemetry Output')
    ax2.set_xlabel("Telemetry Array Channel (Pixel Bin Index)", fontweight='bold')
    ax2.set_ylabel("Telemetry Output (DN)", color='purple', fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 2048)
    ax2.set_ylim(-500, 17500)
    st.pyplot(fig2)

    # --- Chart 3: Sensor Quantum Efficiency Curve (MAKE SURE THIS IS INDENTED 4 SPACES) ---
    fig3, ax3 = plt.subplots(figsize=(10, 2.0))
    ax3.plot(data['wl'], data['qe'] * 100, color='forestgreen', alpha=0.8, label='Silicon CCD QE Profile')
    ax3.set_xlabel("Wavelength ($\lambda$, nm)", fontweight='bold')
    ax3.set_ylabel("Efficiency (%)", color='forestgreen', fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right')
    ax3.set_xlim(wl_min, wl_max)
    ax3.set_ylim(-5, 105)
    st.pyplot(fig3)

else:
    st.error("Invalid geometric bounds. Adjust focal parameters or reduce grating line pitch settings.")



