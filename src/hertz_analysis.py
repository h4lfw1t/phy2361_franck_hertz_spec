import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
import scipy.constants as const
from uncertainties import ufloat
from typing import Union

# Constants
LAMBDA_M = 253.7e-9
V_PP = 39.0
V_F = 6.6

def load_and_preprocess_data(data_path: Path) -> pd.DataFrame:
    """
    Load the Franck-Hertz experiment data and isolate a single sweep.

    :param data_path: Path to the CSV file containing the experimental data.
    :return: A pandas DataFrame containing the isolated sweep.
    """
    df = pd.read_csv(data_path, header=0, usecols=[0, 1], names=['Time', 'Collector_Signal'])
    df['Signal_Diff'] = df['Collector_Signal'].diff()
    reset_indices = df.index[df['Signal_Diff'] < -30].tolist()

    if len(reset_indices) >= 2:
        return df.iloc[reset_indices[0] + 1:reset_indices[1]].copy()
    return df.copy()

def map_time_to_voltage(sweep: pd.DataFrame, v_pp: float) -> pd.DataFrame:
    """
    Map the time axis to an anode voltage axis using a linear assumption.

    :param sweep: DataFrame containing the sweep data.
    :param v_pp: Peak-to-peak voltage.
    :return: DataFrame with the mapped anode voltage.
    """
    t_min = sweep['Time'].min()
    t_max = sweep['Time'].max()
    sweep['Anode_Voltage'] = ((sweep['Time'] - t_min) / (t_max - t_min)) * v_pp
    return sweep

def detect_peaks(sweep: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect prominent peaks in the collector signal.

    :param sweep: DataFrame containing the sweep data with mapped anode voltage.
    :return: A tuple containing an array of peak voltages and an array of peak signals.
    """
    peaks, _ = find_peaks(sweep['Collector_Signal'], prominence=5, distance=3)
    peak_voltages = sweep['Anode_Voltage'].iloc[peaks].values
    peak_signals = sweep['Collector_Signal'].iloc[peaks].values
    return peak_voltages, peak_signals

def calculate_results(peak_voltages: np.ndarray, lambda_m: float) -> None:
    """
    Calculate and print the physical results including Planck's constant using scipy.constants.

    :param peak_voltages: Array of detected peak voltages.
    :param lambda_m: Wavelength of the mercury resonance line.
    """
    delta_vs = np.diff(peak_voltages)
    avg_delta_v = np.mean(delta_vs)
    std_err_delta_v = np.std(delta_vs, ddof=1) / np.sqrt(len(delta_vs))

    delta_v_u = ufloat(avg_delta_v, std_err_delta_v)

    h_exp_u = (const.e * delta_v_u * lambda_m) / const.c
    h_exp_val = h_exp_u.n
    h_exp_err = h_exp_u.s

    percent_error = (abs(h_exp_val - const.h) / const.h) * 100

    print(f"Average Voltage Separation (dV) = {avg_delta_v:.3f} V +/ {std_err_delta_v:.3f} V")
    print(f"Energy of the excited state ~= {avg_delta_v:.3f} eV")
    print(f"\nCalculated Planck's Constant (h) = {h_exp_val:.4e} +/- {h_exp_err:.4e} J * s")
    print(f"Accepted Planck's Constant (h)   = {const.h:.4e} J * s")
    print(f"Percent Error                    = {percent_error:.2f}%")

def plot_results(sweep: pd.DataFrame, peak_voltages: np.ndarray, peak_signals: np.ndarray, savefig: Union[str, Path]) -> None:
    """
    Plot the collector signal versus anode voltage with detected peaks highlighted.

    :param sweep: DataFrame containing the sweep data.
    :param peak_voltages: Array of detected peak voltages.
    :param peak_signals: Array of detected peak signals.
    :param savefig: Path to save the generated plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sweep['Anode_Voltage'], sweep['Collector_Signal'], label='Collector Current', color='blue')
    plt.plot(peak_voltages, peak_signals, "rx", markersize=10, label='Identified Peaks')

    plt.title('Franck-Hertz Experiment: Collector Current vs. Anode Voltage', fontsize=14)
    plt.xlabel('Anode Voltage (V)', fontsize=12)
    plt.ylabel('Collector Signal (Relative)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefig, bbox_inches='tight', dpi=300)
    plt.show()

def plot_raw_data(df: pd.DataFrame, savefig: Union[str, Path]) -> None:
    """
    Plot the raw collector signal versus time.
    
    :param df: DataFrame containing the raw experimental data.
    :param savefig: Path to save the generated plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time'], df['Collector_Signal'], label='Raw Collector Signal', color='green')
    plt.title('Franck-Hertz Experiment: Raw Collector Signal vs. Time', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Collector Signal (Relative)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefig, bbox_inches='tight', dpi=300)
    plt.show()

def main() -> None:
    """
    Main execution function for Franck-Hertz data analysis.
    """
    data_path = Path(__file__).parent.parent / "data" / "franck_hertz_data.csv"
    out_path = Path(__file__).parent.parent / "out"
    out_path.mkdir(exist_ok=True, parents=True)

    raw_data = pd.read_csv(data_path, header=0, usecols=[0, 1], names=['Time', 'Collector_Signal'])
    plot_raw_data(raw_data, savefig=out_path / "raw_data_plot.png")

    sweep = load_and_preprocess_data(data_path)
    sweep = map_time_to_voltage(sweep, V_PP)

    peak_voltages, peak_signals = detect_peaks(sweep)
    print(f"Identified {len(peak_voltages)} peaks at Anode Voltages: {np.round(peak_voltages, 2)} V\n")

    calculate_results(peak_voltages, LAMBDA_M)
    plot_results(sweep, peak_voltages, peak_signals, savefig=out_path / "franck_hertz_plot.png")

if __name__ == "__main__":
    main()