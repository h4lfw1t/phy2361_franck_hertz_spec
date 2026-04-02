import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Sequence
from scipy.stats import linregress


def load_data(filepath: Path | str) -> dict[str, Any]:
    """Load spectroscopy observations from a JSON file.

    :param filepath: Path to the JSON data file.
    :returns: Parsed JSON content.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def calibrate_zero_angle(
        d_nm: float,
        ref_wavelength_nm: float,
        measured_angle_deg: float,
        start_angle_deg: float,
        order: int = 1,
) -> float:
    """
    Calculates the true zero angle of the rotating stage using a reference wavelength.

    :param d_nm: The grating constant in nanometers.
    :param ref_wavelength_nm: Reference wavelength used for calibration in
        nanometers.
    :param measured_angle_deg: Measured angle for the reference wavelength in
        degrees.
    :param start_angle_deg: Initial recorded stage angle in degrees.
    :param order: Diffraction order (default = 1).
    :returns: Calibrated zero-angle position in degrees.
    """
    # Calculate theoretical diffraction angle for the reference wavelength
    theta_ref_rad = np.arcsin((order * ref_wavelength_nm) / d_nm)
    theta_ref_deg = float(np.degrees(theta_ref_rad))

    # Determine true zero angle based on which side of the central max it was measured
    if measured_angle_deg > start_angle_deg:
        true_zero_angle = measured_angle_deg - theta_ref_deg
    else:
        true_zero_angle = measured_angle_deg + theta_ref_deg

    return float(true_zero_angle)


def calculate_wavelength(
    d_nm: float,
    angle_degrees: float,
    true_zero_angle_degrees: float,
    order: int = 1,
) -> float:
    """
    Calculates wavelength using the calibrated zero angle.

    :param d_nm: The grating constant in nanometers.
    :param angle_degrees: Measured diffraction angle in degrees.
    :param true_zero_angle_degrees: Calibrated zero-angle position in degrees.
    :param order: Diffraction order (default = 1).
    :returns: Calculated wavelength in nanometers.
    """
    theta_rad = np.radians(np.abs(angle_degrees - true_zero_angle_degrees))
    return float((d_nm * np.sin(theta_rad)) / order)


def analyze_rydberg_constant(
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    out_path: Path,
) -> None:
    """Fit the Balmer relation and save a regression plot.

    :param x_vals: Balmer-series term values ``(1/2^2 - 1/n^2)``.
    :param y_vals: Inverse wavelengths in ``nm^-1``.
    :param out_path: Output directory for generated figures.
    :returns: None.
    """
    # Linear regression: 1/lambda = R * (1/4 - 1/n^2)
    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)

    R_experimental = slope * 1e9
    R_theory = 1.097373e7  # m^-1
    percent_error = abs(R_experimental - R_theory) / R_theory * 100

    print(f"\nExperimental Rydberg Constant R: {R_experimental:.4e} m^-1")
    print(f"Theoretical Rydberg Constant R: {R_theory:.4e} m^-1")
    print(f"Percent Error: {percent_error:.2f}%")

    # Plot linear regression
    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, y_vals, color='red', zorder=5, label='Experimental Data')

    # Calculate line of best fit
    fit_x = np.linspace(0, max(x_vals) * 1.1, 100)
    fit_y = slope * fit_x + intercept

    plt.plot(fit_x, fit_y, color='blue', linestyle='--', label=f'Linear Fit\n$y = {slope:.4e}x + {intercept:.4e}$')

    plt.title('Hydrogen Balmer Series - Rydberg Constant Regression')
    plt.xlabel(r'$\left(\frac{1}{2^2} - \frac{1}{n^2}\right)$')
    plt.ylabel(r'$\frac{1}{\lambda}$ (nm$^{-1}$)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plot_path = out_path / "rydberg_regression.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved regression plot to {plot_path}")


def perform_calibration(observations: dict[str, Any], d_nominal_nm: float) -> float:
    """Compute stage zero-angle from the sodium reference line.

    :param observations: Observation data grouped by gas name.
    :param d_nominal_nm: Nominal grating spacing in nanometers.
    :returns: Calibrated zero angle in degrees.
    """
    print(f"--- Calibration ---")
    na_data = observations['Sodium']
    na_start = na_data['starting_angle']
    na_angle = na_data['lines'][0]['angle']

    # Average wavelength of the Sodium doublet
    na_avg_wl_nm = (588.995 + 589.592) / 2.0

    # Calculate the true zero angle
    true_zero_angle = calibrate_zero_angle(d_nominal_nm, na_avg_wl_nm, na_angle, na_start)
    angle_offset = np.abs(true_zero_angle - na_start)

    print(f"Nominal d: {d_nominal_nm:.2f} nm")
    print(f"Recorded Start Angle: {na_start:.2f}°")
    print(f"True Zero Angle: {true_zero_angle:.2f}°")
    print(f"Zero Angle Error: {angle_offset:.2f}°")

    return true_zero_angle


def analyze_hydrogen(
    observations: dict[str, Any],
    d_nominal_nm: float,
    true_zero_angle: float,
    out_path: Path,
) -> None:
    """Analyze hydrogen Balmer lines and estimate the Rydberg constant.

    :param observations: Observation data grouped by gas name.
    :param d_nominal_nm: Nominal grating spacing in nanometers.
    :param true_zero_angle: Calibrated zero-angle position in degrees.
    :param out_path: Output directory for generated plots.
    :returns: None.
    """
    print(f"\n--- Hydrogen Analysis (Rydberg Constant) ---")
    h_data = observations['Hydrogen']

    color_to_n = {
        "Red": 3,
        "Cyan": 4,
        "Blue": 5,
        # NOTE: Omit violet due to anomalous results
        # "Violet": 6
    }

    x_vals = []
    y_vals = []

    for line in h_data['lines']:
        color = line['color']
        measured_angle = line['angle']
        n = color_to_n.get(color)
        if n:
            # Pass the true_zero_angle instead of the recorded start angle
            wl = calculate_wavelength(d_nominal_nm, measured_angle, true_zero_angle)
            true_angle = np.abs(measured_angle - true_zero_angle)
            inv_wl = 1.0 / wl
            term = (0.25 - 1.0 / (n ** 2))
            x_vals.append(term)
            y_vals.append(inv_wl)
            print(f"Color: {color:7} | n={n} | Measured Angle={measured_angle:6.1f}° | True Angle={true_angle:3.1f}° | Wavelength={wl:6.2f} nm")

    analyze_rydberg_constant(x_vals, y_vals, out_path)


def analyze_helium(
    observations: dict[str, Any],
    d_nominal_nm: float,
    true_zero_angle: float,
) -> None:
    """Compute wavelengths for measured helium emission lines.

    :param observations: Observation data grouped by gas name.
    :param d_nominal_nm: Nominal grating spacing in nanometers.
    :param true_zero_angle: Calibrated zero-angle position in degrees.
    :returns: None.
    """
    print(f"\n--- Helium Analysis ---")
    he_data = observations['Helium']

    for line in he_data['lines']:
        color = line['color']
        measured_angle = line['angle']

        wl = calculate_wavelength(d_nominal_nm, measured_angle, true_zero_angle)
        true_angle = np.abs(measured_angle - true_zero_angle)
        print(f"Color: {color:7} | Measured Angle={measured_angle:6.1f}° | True Angle={true_angle:3.1f}° | Wavelength={wl:6.2f} nm")


def analyze_unknown_e(
    observations: dict[str, Any],
    d_nominal_nm: float,
    true_zero_angle: float,
) -> None:
    """Compute wavelengths for measured lines of the unknown gas sample.

    :param observations: Observation data grouped by gas name.
    :param d_nominal_nm: Nominal grating spacing in nanometers.
    :param true_zero_angle: Calibrated zero-angle position in degrees.
    :returns: None.
    """
    print(f"\n--- Unknown_E Analysis ---")
    unk_data = observations['Unknown_E']

    for line in unk_data['lines']:
        color = line['color']
        measured_angle = line['angle']

        wl = calculate_wavelength(d_nominal_nm, measured_angle, true_zero_angle)
        true_angle = np.abs(measured_angle - true_zero_angle)
        print(f"Color: {color:10} | Measured Angle={measured_angle:6.1f}° | True Angle={true_angle:3.1f}° | Wavelength={wl:6.2f} nm")


def main() -> None:
    """Run calibration and spectroscopy analysis pipeline.

    :returns: None.
    """
    data_path = Path(__file__).parent.parent / "data" / "spectrum.json"
    out_path = Path(__file__).parent.parent / "out"
    out_path.mkdir(exist_ok=True, parents=True)

    data = load_data(data_path)

    params = data['experimental_parameters']
    observations = data['observations']

    # Get calibration
    d_nominal_nm = 1e6 / params['diffraction_grating_lines_per_mm']
    true_zero_angle = perform_calibration(observations, d_nominal_nm)

    # Analyze gasses
    analyze_hydrogen(observations, d_nominal_nm, true_zero_angle, out_path)
    analyze_helium(observations, d_nominal_nm, true_zero_angle)
    analyze_unknown_e(observations, d_nominal_nm, true_zero_angle)


if __name__ == "__main__":
    main()