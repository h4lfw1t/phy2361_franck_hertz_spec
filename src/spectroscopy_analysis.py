import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Sequence
from scipy.stats import linregress
from uncertainties import ufloat
from uncertainties.umath import asin, degrees, radians, sin
import uncertainties.unumpy as unp


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
        measured_angle_deg: Any,
        start_angle_deg: Any,
        order: int = 1,
) -> Any:
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
    theta_ref_rad = asin((order * ref_wavelength_nm) / d_nm)
    theta_ref_deg = degrees(theta_ref_rad)

    # Determine true zero angle based on which side of the central max it was measured
    if measured_angle_deg > start_angle_deg:
        true_zero_angle = measured_angle_deg - theta_ref_deg
    else:
        true_zero_angle = measured_angle_deg + theta_ref_deg

    return true_zero_angle


def calculate_wavelength(
    d_nm: float,
    angle_degrees: Any,
    true_zero_angle_degrees: Any,
    order: int = 1,
) -> Any:
    """
    Calculates wavelength using the calibrated zero angle.

    :param d_nm: The grating constant in nanometers.
    :param angle_degrees: Measured diffraction angle in degrees.
    :param true_zero_angle_degrees: Calibrated zero-angle position in degrees.
    :param order: Diffraction order (default = 1).
    :returns: Calculated wavelength in nanometers.
    """
    theta_rad = radians(abs(angle_degrees - true_zero_angle_degrees))
    return (d_nm * sin(theta_rad)) / order


def analyze_rydberg_constant(
    x_vals: Sequence[Any],
    y_vals: Sequence[Any],
    out_path: Path,
) -> None:
    """Fit the Balmer relation and save a regression plot.

    :param x_vals: Balmer-series term values ``(1/2^2 - 1/n^2)``.
    :param y_vals: Inverse wavelengths in ``nm^-1``.
    :param out_path: Output directory for generated figures.
    :returns: None.
    """
    # Extract nominal values for plotting/regression
    x_nom = unp.nominal_values(x_vals)
    y_nom = unp.nominal_values(y_vals)
    y_std = unp.std_devs(y_vals)

    # Linear regression: 1/lambda = R * (1/4 - 1/n^2)
    if any(y_std > 0):
        weights = 1 / y_std
        idx = np.isfinite(weights)
        slope, intercept = np.polyfit(np.array(x_nom)[idx], np.array(y_nom)[idx], 1, w=weights[idx])
        cov = np.polyfit(np.array(x_nom)[idx], np.array(y_nom)[idx], 1, w=weights[idx], cov=True)[1]
        slope_err = np.sqrt(cov[0, 0])
        slope_u = ufloat(slope, slope_err)
    else:
        slope, intercept, r_value, p_value, std_err = linregress(x_nom, y_nom)
        slope_u = ufloat(slope, std_err)

    R_experimental = slope_u * 1e9
    R_theory = 1.097373e7  # m^-1

    print(f"\nExperimental Rydberg Constant R: {R_experimental} m^-1")
    print(f"Theoretical Rydberg Constant R: {R_theory:.4e} m^-1")

    # Plot linear regression
    plt.figure(figsize=(8, 6))
    if any(y_std > 0):
        plt.errorbar(x_nom, y_nom, yerr=y_std, fmt='o', color='red', zorder=5, label='Experimental Data (with uncertainty)')
    else:
        plt.scatter(x_nom, y_nom, color='red', zorder=5, label='Experimental Data')

    # Calculate line of best fit
    fit_x = np.linspace(0, max(x_nom) * 1.1, 100)
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


def perform_calibration(observations: dict[str, Any], d_nominal_nm: float) -> Any:
    """Compute stage zero-angle from the sodium reference line.

    :param observations: Observation data grouped by gas name.
    :param d_nominal_nm: Nominal grating spacing in nanometers.
    :returns: Calibrated zero angle in degrees.
    """
    print(f"--- Calibration ---")
    na_data = observations['Sodium']
    na_start = ufloat(na_data['starting_angle'], 1/6)
    na_angle = ufloat(na_data['lines'][0]['angle'], 1/6)

    # Average wavelength of the Sodium doublet
    na_avg_wl_nm = (588.995 + 589.592) / 2.0

    # Calculate the true zero angle
    true_zero_angle = calibrate_zero_angle(d_nominal_nm, na_avg_wl_nm, na_angle, na_start)
    angle_offset = abs(true_zero_angle - na_start)

    print(f"Nominal d: {d_nominal_nm:.2f} nm")
    print(f"Recorded Start Angle: {na_start} deg")
    print(f"True Zero Angle: {true_zero_angle} deg")
    print(f"Zero Angle Error: {angle_offset} deg")

    return true_zero_angle


def analyze_hydrogen(
    observations: dict[str, Any],
    d_nominal_nm: float,
    true_zero_angle: Any,
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
        measured_angle = ufloat(line['angle'], 1/6)
        n = color_to_n.get(color)
        if n:
            wl = calculate_wavelength(d_nominal_nm, measured_angle, true_zero_angle)
            true_angle = abs(measured_angle - true_zero_angle)
            inv_wl = 1.0 / wl
            term = (0.25 - 1.0 / (n ** 2))
            x_vals.append(term)
            y_vals.append(inv_wl)
            print(f"Color: {color:7} | n={n} | Measured Angle={measured_angle} deg | True Angle={true_angle} deg | Wavelength={wl} nm")

    analyze_rydberg_constant(x_vals, y_vals, out_path)


def analyze_helium(
    observations: dict[str, Any],
    d_nominal_nm: float,
    true_zero_angle: Any,
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
        measured_angle = ufloat(line['angle'], 1/6)

        wl = calculate_wavelength(d_nominal_nm, measured_angle, true_zero_angle)
        true_angle = abs(measured_angle - true_zero_angle)
        print(f"Color: {color:7} | Measured Angle={measured_angle} deg | True Angle={true_angle} deg | Wavelength={wl} nm")


def analyze_unknown_e(
    observations: dict[str, Any],
    d_nominal_nm: float,
    true_zero_angle: Any,
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
        measured_angle = ufloat(line['angle'], 1/6)

        wl = calculate_wavelength(d_nominal_nm, measured_angle, true_zero_angle)
        true_angle = abs(measured_angle - true_zero_angle)
        print(f"Color: {color:10} | Measured Angle={measured_angle} deg | True Angle={true_angle} deg | Wavelength={wl} nm")


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