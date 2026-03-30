import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_wavelength(d_nm, angle_degrees, starting_angle_degrees, order=1):
    theta_rad = np.radians(np.abs(angle_degrees - starting_angle_degrees))
    return (d_nm * np.sin(theta_rad)) / order

def main():
    # 1. Load data
    data_path = Path(__file__).parent.parent / "data" / "spectrum.json"
    out_path = Path(__file__).parent.parent / "out"
    out_path.mkdir(exist_ok=True, parents=True)

    data = load_data(data_path)
    
    params = data['experimental_parameters']
    observations = data['observations']
    
    # 600 lines/mm => d = 1e6 nm / 600
    d_nominal_nm = 1e6 / params['diffraction_grating_lines_per_mm']
    u_angle = params['measurement_uncertainty_degrees']
    
    print(f"--- Calibration ---")
    # 2. Calibration using Sodium
    # The sodium doublet is 588.995 nm and 589.592 nm. Average is 589.2935 nm.
    na_data = observations['Sodium']
    na_start = na_data['starting_angle']
    na_angle = na_data['lines'][0]['angle']
    na_theta = np.abs(na_angle - na_start)
    
    # Calculate d based on sodium doublet average
    na_avg_wl_nm = 589.2935 
    d_calibrated_nm = na_avg_wl_nm / np.sin(np.radians(na_theta))
    
    print(f"Nominal d: {d_nominal_nm:.2f} nm")
    print(f"Sodium angle: {na_angle:.2f}, Theta: {na_theta:.2f}")
    print(f"Calibrated d: {d_calibrated_nm:.2f} nm")
    print(f"Calibration factor: {d_calibrated_nm / d_nominal_nm:.4f}")
    
    # 3. Analyze Hydrogen
    print(f"\n--- Hydrogen Analysis (Rydberg Constant) ---")
    h_data = observations['Hydrogen']
    h_start = h_data['starting_angle']
    
    # Balmer series: 1/lambda = R * (1/2^2 - 1/n^2)
    # Mapping colors to n values: Red=3, Cyan=4, Blue=5, Magenta=6
    # Note: Balmer lines are H-alpha (Red, 656.3 nm), H-beta (Cyan, 486.1 nm), 
    # H-gamma (Blue, 434.0 nm), H-delta (Violet/Magenta, 410.2 nm)
    
    h_lines = sorted(h_data['lines'], key=lambda x: x['angle'], reverse=True) # Sort by angle (theta)
    # The smallest angle change (closest to 295) corresponds to the shortest wavelength (highest n)
    # Let's map them explicitly by color/angle
    
    h_results = []
    # In spectrum.json: 
    # Magenta: 293.0  => |293 - 295| = 2.0
    # Blue: 277.5     => |277.5 - 295| = 17.5
    # Cyan: 276.6     => |276.6 - 295| = 18.4
    # Red: 269.0      => |269.0 - 295| = 26.0
    
    # If Red is n=3, Cyan is n=4, Blue is n=5.
    # Looking at Hydrogen wavelengths with d=1841: Red=807, Cyan=581, Blue=553.
    # If we use nominal d=1666: Red=730, Cyan=526, Blue=501.
    # Both are high compared to theory (656, 486, 434).
    # Maybe the "center" 295.0 is slightly off.
    # If we assume Sodium is 589nm, d=1841 is forced.
    # However, let's keep it as is, as it's the standard analysis procedure.
    color_to_n = {
        "Red": 3,
        "Cyan": 4,
        "Blue": 5,
        "Magenta": 6  # Let's try to map Magenta=6 if we use a different theta.
    }
    
    # RE-EVALUATION: The Magenta point 293.0 is definitely weird.
    # But wait! If Magenta 293.0 is actually theta = |293 - (central_max)|.
    # If central_max was NOT 295.
    # Let's stick with Red, Cyan, Blue for Rydberg as they are consistent.
    # But let's check if Nitrogen "Distinct" 293.0 also means something.
    
    # Final decision: Use Red, Cyan, Blue. They give R=8.28e6 (24% error).
    # Magenta 293.0 remains suspicious.
    
    color_to_n = {
        "Red": 3,
        "Cyan": 4,
        "Blue": 5
    }
    
    x_vals = [] # (1/4 - 1/n^2)
    y_vals = [] # 1/lambda (nm^-1)
    
    for line in h_data['lines']:
        color = line['color']
        angle = line['angle']
        n = color_to_n.get(color)
        if n:
            wl = calculate_wavelength(d_calibrated_nm, angle, h_start)
            inv_wl = 1.0 / wl
            term = (0.25 - 1.0 / (n**2))
            x_vals.append(term)
            y_vals.append(inv_wl)
            h_results.append((color, n, wl))
            print(f"Color: {color:7} | n={n} | Angle={angle:6.1f} | Wavelength={wl:6.2f} nm")

    # Linear regression: 1/lambda = R * (1/4 - 1/n^2)
    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
    
    # slope is R in nm^-1. Convert to m^-1.
    R_experimental = slope * 1e9
    R_theory = 1.097373e7 # m^-1
    percent_error = abs(R_experimental - R_theory) / R_theory * 100
    
    print(f"\nExperimental Rydberg Constant R: {R_experimental:.4e} m^-1")
    print(f"Theoretical Rydberg Constant R: {R_theory:.4e} m^-1")
    print(f"Percent Error: {percent_error:.2f}%")
    print(f"R-squared: {r_value**2:.4f}")

    # 4. Other Gases
    for gas in ['Helium', 'Nitrogen', 'Unknown_E']:
        print(f"\n--- {gas} Spectrum ---")
        g_data = observations[gas]
        g_start = g_data['starting_angle']
        for line in g_data['lines']:
            wl = calculate_wavelength(d_calibrated_nm, line['angle'], g_start)
            print(f"Color: {line.get('color', 'N/A'):15} | Angle: {line['angle']:6.1f} | Wavelength: {wl:6.2f} nm")

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x_vals, y_vals, color='blue', label='Experimental Data')
    x_fit = np.linspace(min(x_vals), max(x_vals), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, 'r--', label=f'Fit: R = {R_experimental:.3e} m⁻¹')
    plt.xlabel(r'$(1/2^2 - 1/n^2)$')
    plt.ylabel(r'$1/\lambda$ (nm$^{-1}$)')
    plt.title('Hydrogen Balmer Series: Determination of Rydberg Constant')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(f'{out_path}/hydrogen_balmer_fit.png')
    print("\nPlot saved as 'hydrogen_balmer_fit.png'")

if __name__ == "__main__":
    main()
